
// cpp/hybrid_core.cpp
//
// Full Hybrid A* search kernel (pybind11).
//
// Design goals:
//   - Keep all per-plan data local to `run_search` (no global mutable state).
//   - Release the GIL during the heavy search loop to enable future parallel planning with Python threads.
//   - Use contiguous NumPy arrays for inputs to avoid copying
//   - Provide a small, stable ABI surface for Python

#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <cmath>
#include <cstdint>
#include <vector>
#include <limits>
#include <queue>



namespace py = pybind11;

static constexpr double M_PI = 3.14159265358979323846;
static constexpr double TAU = 6.28318530717958647692;

inline double wrap_angle(double th) {
    // [-pi, pi)
    th = std::fmod(th + M_PI, TAU);
    if (th < 0) th += TAU;
    return th - M_PI;
}

inline double wrap_to_2pi(double th) {
    th = std::fmod(th, TAU);
    if (th < 0) th += TAU;
    return th;
}

inline int theta_to_bin(double th, int theta_bins) {
    const double tp = wrap_to_2pi(th);
    const double dtheta = TAU / (double)theta_bins;
    int k = (int)std::floor(tp / dtheta);
    k %= theta_bins;
    if (k < 0) k += theta_bins;
    return k;
}

inline void world_to_grid(double x, double y, double ox, double oy, double res, int& ix, int& iy) {
    ix = (int)std::floor((x - ox) / res);
    iy = (int)std::floor((y - oy) / res);
}

struct Pose3 {
    double x, y, th;
};

inline Pose3 propagate_bicycle(const Pose3& p, double steer, int direction, double ds, double wheelbase) {
    const double sigma = (direction >= 0) ? 1.0 : -1.0;
    const double kappa = std::tan(steer) / wheelbase;
    if (std::abs(kappa) < 1e-12) {
        return Pose3{p.x + sigma * ds * std::cos(p.th), p.y + sigma * ds * std::sin(p.th), p.th};
    }
    const double dth = sigma * ds * kappa;
    const double th1 = wrap_angle(p.th + dth);
    const double R = 1.0 / kappa;
    return Pose3{
        p.x + sigma * (std::sin(th1) - std::sin(p.th)) * R,
        p.y - sigma * (std::cos(th1) - std::cos(p.th)) * R,
        th1,
    };
}

// Min-heap for open set

struct HeapItem {
    double f;
    int id;
};

struct HeapCmp {
    bool operator()(const HeapItem& a, const HeapItem& b) const { return a.f > b.f; }
};

inline int dir_to_index(int direction) { return (direction >= 0) ? 0 : 1; }

inline std::int64_t pack_key(int ix, int iy, int itheta, int direction, int W, int theta_bins) {
    const int idir = dir_to_index(direction);
    // idx = (((iy*W + ix)*theta_bins + itheta)*2 + idir)
    return (std::int64_t)((((std::int64_t)iy * (std::int64_t)W + (std::int64_t)ix) * (std::int64_t)theta_bins +
                           (std::int64_t)itheta) *
                              2 +
                          (std::int64_t)idir);
}

inline bool goal_reached(const Pose3& p, double gx, double gy, double gth, double pos_tol, double th_tol) {
    const double dx = p.x - gx;
    const double dy = p.y - gy;
    if (dx * dx + dy * dy > pos_tol * pos_tol) return false;
    const double dth = wrap_angle(p.th - gth);
    return std::abs(dth) <= th_tol;
}


/**
 * Expand a batch of primitives from a single pose
 *
 * Inputs:
 *  - pose: (x,y,th)
 *  - prev_dir: +1/-1 (used only for returning direction list; cost still computed in Python)
 *  - steer_angles: 1D array of steer angles (radians)
 *  - allow_reverse: bool
 *  - step_size, n_substeps
 *  - wheelbase
 *  - occ: 2D numpy bool/uint8 grid [H,W], True means obstacle
 *  - origin (ox,oy), res
 *  - theta_bins
 *  - use_fp: whether to use footprint collision checking
 *  - footprint_offsets: Nx2 array of (x,y) offsets in vehicle frame
 *
 * Returns:
 *  dict with arrays for valid successors only:
 *    x,y,th (float64), key_idx (int64), ix,iy,itheta,dir (int32)
 */
py::dict expand_primitives(
    double x0, double y0, double th0,
    int prev_dir,
    py::array_t<double, py::array::c_style | py::array::forcecast> steer_angles,
    bool allow_reverse,
    double step_size, int n_substeps,
    double wheelbase,
    py::array occ_in,
    double ox, double oy, double res,
    int theta_bins,
    bool use_fp,
    py::array_t<double, py::array::c_style | py::array::forcecast> footprint_offsets
) {
    // Accept bool or uint8 occupancy; treat nonzero as occupied
    py::buffer_info occ = occ_in.request();
    if (occ.ndim != 2) throw std::runtime_error("occ must be 2D");
    const int H = (int)occ.shape[0];
    const int W = (int)occ.shape[1];
    const auto* occ_ptr = (const std::uint8_t*)occ.ptr;

    py::buffer_info fp = footprint_offsets.request();
    const double* fp_ptr = (fp.ndim == 2 && fp.shape[1] == 2) ? (const double*)fp.ptr : nullptr;
    const int fpN = (fp_ptr && use_fp) ? (int)fp.shape[0] : 0;

    auto steer = steer_angles.unchecked<1>();
    const int S = (int)steer.shape(0);
    const int n_dirs = allow_reverse ? 2 : 1;

    std::vector<double> out_x; out_x.reserve(n_dirs * S);
    std::vector<double> out_y; out_y.reserve(n_dirs * S);
    std::vector<double> out_th; out_th.reserve(n_dirs * S);
    std::vector<std::int32_t> out_ix; out_ix.reserve(n_dirs * S);
    std::vector<std::int32_t> out_iy; out_iy.reserve(n_dirs * S);
    std::vector<std::int32_t> out_it; out_it.reserve(n_dirs * S);
    std::vector<std::int32_t> out_dir; out_dir.reserve(n_dirs * S);
    std::vector<std::int64_t> out_key; out_key.reserve(n_dirs * S);

    const double ds = step_size / (double)n_substeps;

    auto pose_free = [&](const Pose3& p) -> bool {
        if (fpN == 0) {
            int ix, iy;
            world_to_grid(p.x, p.y, ox, oy, res, ix, iy);
            if (ix < 0 || ix >= W || iy < 0 || iy >= H) return false;
            return occ_ptr[iy * W + ix] == 0;
        }
        const double c = std::cos(p.th);
        const double s = std::sin(p.th);
        for (int i = 0; i < fpN; ++i) {
            const double dx = fp_ptr[2 * i + 0];
            const double dy = fp_ptr[2 * i + 1];
            const double x = p.x + c * dx - s * dy;
            const double y = p.y + s * dx + c * dy;
            int ix, iy;
            world_to_grid(x, y, ox, oy, res, ix, iy);
            if (ix < 0 || ix >= W || iy < 0 || iy >= H) return false;
            if (occ_ptr[iy * W + ix] != 0) return false;
        }
        return true;
    };

    for (int di = 0; di < n_dirs; ++di) {
        const int direction = (di == 0) ? +1 : -1;
        for (int si = 0; si < S; ++si) {
            Pose3 p{x0, y0, th0};
            bool ok = true;
            for (int k = 0; k < n_substeps; ++k) {
                p = propagate_bicycle(p, steer(si), direction, ds, wheelbase);
                if (!pose_free(p)) {
                    ok = false;
                    break;
                }
            }
            if (!ok) continue;

            int ix, iy;
            world_to_grid(p.x, p.y, ox, oy, res, ix, iy);
            if (ix < 0 || ix >= W || iy < 0 || iy >= H) continue;
            const int it = theta_to_bin(p.th, theta_bins);

            out_x.push_back(p.x);
            out_y.push_back(p.y);
            out_th.push_back(p.th);
            out_ix.push_back((std::int32_t)ix);
            out_iy.push_back((std::int32_t)iy);
            out_it.push_back((std::int32_t)it);
            out_dir.push_back((std::int32_t)direction);
            out_key.push_back(pack_key(ix, iy, it, direction, W, theta_bins));
        }
    }

    py::dict d;
    d["x"] = py::array_t<double>(out_x.size(), out_x.data());
    d["y"] = py::array_t<double>(out_y.size(), out_y.data());
    d["th"] = py::array_t<double>(out_th.size(), out_th.data());
    d["ix"] = py::array_t<std::int32_t>(out_ix.size(), out_ix.data());
    d["iy"] = py::array_t<std::int32_t>(out_iy.size(), out_iy.data());
    d["itheta"] = py::array_t<std::int32_t>(out_it.size(), out_it.data());
    d["dir"] = py::array_t<std::int32_t>(out_dir.size(), out_dir.data());
    d["key_idx"] = py::array_t<std::int64_t>(out_key.size(), out_key.data());
    return d;
}

py::dict run_search(
    // start
    double sx,
    double sy,
    double sth,
    // goal
    double gx,
    double gy,
    double gth,
    double pos_tol,
    double th_tol,
    int max_expansions,
    // occupancy
    py::array_t<std::uint8_t, py::array::c_style | py::array::forcecast> occ_in,
    double ox,
    double oy,
    double res,
    int theta_bins,
    // controls
    py::array_t<double, py::array::c_style | py::array::forcecast> steer_angles,
    bool allow_reverse,
    double step_size,
    int n_substeps,
    double wheelbase,
    // penalties
    double reverse_penalty,
    double switch_dir_penalty,
    // heuristics
    py::array_t<double, py::array::c_style | py::array::forcecast> h2d_dist,
    py::array_t<double, py::array::c_style | py::array::forcecast> nh_table,
    double nh_R,
    double nh_res,
    double nh_dth,
    int nh_nxy,
    int nh_theta_bins,
    // optional Voronoi edge shaping
    bool use_rho,
    py::array_t<double, py::array::c_style | py::array::forcecast> rho_in,
    double voronoi_weight,
    // footprint
    bool use_fp,
    py::array_t<double, py::array::c_style | py::array::forcecast> footprint_offsets
) {
    // --- Grab buffers (with GIL held) ---
    py::buffer_info occ = occ_in.request();
    if (occ.ndim != 2) throw std::runtime_error("occ must be 2D");
    const int H = (int)occ.shape[0];
    const int W = (int)occ.shape[1];
    const auto* occ_ptr = (const std::uint8_t*)occ.ptr;

    py::buffer_info h2d = h2d_dist.request();
    if (h2d.ndim != 2) throw std::runtime_error("h2d_dist must be 2D");
    const auto* h2d_ptr = (const double*)h2d.ptr;

    py::buffer_info nh = nh_table.request();
    if (nh.ndim != 3) throw std::runtime_error("nh_table must be 3D");
    const auto* nh_ptr = (const double*)nh.ptr;

    py::buffer_info rho = rho_in.request();
    const double* rho_ptr = nullptr;
    if (use_rho) {
        if (rho.ndim != 2) throw std::runtime_error("rho must be 2D");
        rho_ptr = (const double*)rho.ptr;
    }

    py::buffer_info fp = footprint_offsets.request();
    const double* fp_ptr = nullptr;
    int fpN = 0;
    if (use_fp) {
        if (fp.ndim != 2 || fp.shape[1] != 2) throw std::runtime_error("footprint_offsets must be (N,2)");
        fp_ptr = (const double*)fp.ptr;
        fpN = (int)fp.shape[0];
    }

    auto steer = steer_angles.unchecked<1>();
    const int S = (int)steer.shape(0);

    // --- Precompute goal-frame rotation for NH heuristic ---
    const double cg = std::cos(gth);
    const double sg = std::sin(gth);

    // --- Helpers ---
    auto in_bounds = [&](int ix, int iy) -> bool { return (0 <= ix && ix < W && 0 <= iy && iy < H); };

    auto occ_at = [&](int ix, int iy) -> bool {
        if (!in_bounds(ix, iy)) return true;
        return occ_ptr[iy * W + ix] != 0;
    };

    auto pose_free = [&](const Pose3& p) -> bool {
        if (fpN == 0) {
            int ix, iy;
            world_to_grid(p.x, p.y, ox, oy, res, ix, iy);
            return !occ_at(ix, iy);
        }
        const double c = std::cos(p.th);
        const double s = std::sin(p.th);
        for (int i = 0; i < fpN; ++i) {
            const double dx = fp_ptr[2 * i + 0];
            const double dy = fp_ptr[2 * i + 1];
            const double x = p.x + c * dx - s * dy;
            const double y = p.y + s * dx + c * dy;
            int ix, iy;
            world_to_grid(x, y, ox, oy, res, ix, iy);
            if (occ_at(ix, iy)) return false;
        }
        return true;
    };

    auto h_holonomic = [&](int ix, int iy, double x, double y) -> double {
        if (!in_bounds(ix, iy)) return std::numeric_limits<double>::infinity();
        const double v = h2d_ptr[iy * W + ix];
        if (std::isfinite(v)) return v;
        // goal may be in obstacle => map is inf; fall back
        return std::hypot(x - gx, y - gy);
    };

    auto h_nonhol = [&](double x, double y, double th) -> double {
        // Transform pose into goal frame: R(-gth) * (p - goal)
        const double dx = x - gx;
        const double dy = y - gy;
        const double xl = cg * dx + sg * dy;
        const double yl = -sg * dx + cg * dy;

        if (std::abs(xl) > nh_R || std::abs(yl) > nh_R) {
            return std::hypot(xl, yl);
        }

        const int ix = (int)std::llround(xl / nh_res) + nh_nxy / 2;
        const int iy = (int)std::llround(yl / nh_res) + nh_nxy / 2;
        if (ix < 0 || ix >= nh_nxy || iy < 0 || iy >= nh_nxy) {
            return std::hypot(xl, yl);
        }

        const double thl = wrap_angle(th - gth);
        const int it = (int)std::floor(wrap_to_2pi(thl) / nh_dth) % nh_theta_bins;

        // table layout: [iy, ix, it]
        const std::int64_t idx = ((std::int64_t)iy * (std::int64_t)nh_nxy + (std::int64_t)ix) * (std::int64_t)nh_theta_bins +
                                 (std::int64_t)it;
        const double v = nh_ptr[idx];
        if (std::isfinite(v)) return v;
        return std::hypot(xl, yl);
    };

    auto heuristic = [&](int ix, int iy, const Pose3& p) -> double {
        const double hh = h_holonomic(ix, iy, p.x, p.y);
        const double hn = h_nonhol(p.x, p.y, p.th);
        return (hh > hn) ? hh : hn;
    };

    auto rho_at = [&](int ix, int iy) -> double {
        if (!use_rho || rho_ptr == nullptr) return 0.0;
        if (!in_bounds(ix, iy)) return 0.0;
        double v = rho_ptr[iy * W + ix];
        if (!std::isfinite(v)) return 0.0;
        // rho is expected in [0,1], but clamp defensively.
        if (v < 0.0) v = 0.0;
        if (v > 1.0) v = 1.0;
        return v;
    };

    // --- Allocate per-plan state ---
    const std::int64_t bestN = (std::int64_t)H * (std::int64_t)W * (std::int64_t)theta_bins * 2;
    std::vector<float> best_g((size_t)bestN, std::numeric_limits<float>::infinity());

    std::vector<double> xs;
    std::vector<double> ys;
    std::vector<double> ths;
    std::vector<double> gs;
    std::vector<int> parents;
    std::vector<std::int8_t> dirs;
    std::vector<std::int64_t> keys;

    xs.reserve((size_t)max_expansions + 1);
    ys.reserve((size_t)max_expansions + 1);
    ths.reserve((size_t)max_expansions + 1);
    gs.reserve((size_t)max_expansions + 1);
    parents.reserve((size_t)max_expansions + 1);
    dirs.reserve((size_t)max_expansions + 1);
    keys.reserve((size_t)max_expansions + 1);

    auto push_node = [&](const Pose3& p, double g, int parent, int direction, int ix, int iy, int itheta) -> int {
        const std::int64_t key = pack_key(ix, iy, itheta, direction, W, theta_bins);
        const int id = (int)xs.size();
        xs.push_back(p.x);
        ys.push_back(p.y);
        ths.push_back(p.th);
        gs.push_back(g);
        parents.push_back(parent);
        dirs.push_back((std::int8_t)direction);
        keys.push_back(key);
        return id;
    };

    // Start discretization
    int six, siy;
    world_to_grid(sx, sy, ox, oy, res, six, siy);
    const int sit = theta_to_bin(sth, theta_bins);
    // Start-in-collision => fail quickly.
    if (!pose_free(Pose3{sx, sy, sth})) {
        py::dict out;
        out["path"] = py::array_t<double>({0, 3});
        py::dict st;
        st["expanded"] = 0;
        st["pushed"] = 0;
        st["collision_checks"] = 0;
        st["analytic_attempts"] = 0;
        st["analytic_successes"] = 0;
        out["stats"] = st;
        return out;
    }
    const int start_dir = +1;
    const std::int64_t start_key = pack_key(six, siy, sit, start_dir, W, theta_bins);

    // --- Run search with GIL released ---
    std::priority_queue<HeapItem, std::vector<HeapItem>, HeapCmp> open;

    int expanded = 0;
    int pushed = 0;
    int collision_checks = 0;
    {
        py::gil_scoped_release release;
        Pose3 p0{sx, sy, sth};
        const double h0 = heuristic(six, siy, p0);
        const int n0 = push_node(p0, 0.0, -1, start_dir, six, siy, sit);
        best_g[(size_t)start_key] = 0.0f;
        open.push(HeapItem{h0, n0});
        pushed++;
        const double ds = step_size / (double)n_substeps;
        while (!open.empty() && expanded < max_expansions) {
            const HeapItem top = open.top();
            open.pop();
            const int nid = top.id;
            const std::int64_t key = keys[(size_t)nid];
            const double gcur = gs[(size_t)nid];
            // stale check
            if ((double)best_g[(size_t)key] + 1e-12 < gcur) continue;
            expanded++;
            Pose3 cur{xs[(size_t)nid], ys[(size_t)nid], ths[(size_t)nid]};
            const int prev_dir = (int)dirs[(size_t)nid];
            if (goal_reached(cur, gx, gy, gth, pos_tol, th_tol))
                break;
            const int n_dirs = allow_reverse ? 2 : 1;
            for (int di = 0; di < n_dirs; ++di) {
                const int direction = (di == 0) ? +1 : -1;
                const double base_edge = step_size * ((direction < 0) ? reverse_penalty : 1.0) +
                                         ((direction != prev_dir) ? switch_dir_penalty : 0.0);
                for (int si = 0; si < S; ++si) {
                    collision_checks += 1;
                    Pose3 p = cur;
                    bool ok = true;
                    double rho_accum = 0.0;
                    for (int k = 0; k < n_substeps; ++k) {
                        p = propagate_bicycle(p, steer(si), direction, ds, wheelbase);
                        if (!pose_free(p)) {
                            ok = false;
                            break;
                        }
                        if (use_rho) {
                            int ix, iy;
                            world_to_grid(p.x, p.y, ox, oy, res, ix, iy);
                            rho_accum += rho_at(ix, iy);
                        }
                    }
                    if (!ok) continue;
                    int ix, iy;
                    world_to_grid(p.x, p.y, ox, oy, res, ix, iy);
                    if (!in_bounds(ix, iy)) continue;
                    const int it = theta_to_bin(p.th, theta_bins);
                    const std::int64_t kidx = pack_key(ix, iy, it, direction, W, theta_bins);
                    double edge = base_edge;
                    if (use_rho && voronoi_weight != 0.0) {
                        const double rho_mean = rho_accum / (double)n_substeps;
                        edge += step_size * voronoi_weight * rho_mean;
                    }
                    const double g2 = gcur + edge;
                    if (g2 + 1e-12 >= (double)best_g[(size_t)kidx]) continue;
                    const double h2 = heuristic(ix, iy, p);
                    const double f2 = g2 + h2;
                    // Insert node
                    const int nid2 = push_node(p, g2, nid, direction, ix, iy, it);
                    best_g[(size_t)kidx] = (float)g2;
                    open.push(HeapItem{f2, nid2});
                    pushed++;
                }
            }
        }
    } // GIL reacquired

    // --- Find best goal node (the loop breaks on first reached goal, but might pop stale nodes)
    // We'll scan for the first node that satisfies goal; because we increment expanded after pop,
    // the last processed node is likely close, but scanning keeps it simple.
    int goal_id = -1;
    for (int i = (int)xs.size() - 1; i >= 0; --i) {
        Pose3 p{xs[(size_t)i], ys[(size_t)i], ths[(size_t)i]};
        if (goal_reached(p, gx, gy, gth, pos_tol, th_tol)) {
            goal_id = i;
            break;
        }
    }

    std::vector<double> path;
    if (goal_id >= 0) {
        // reconstruct
        std::vector<int> ids;
        int nid = goal_id;
        while (nid >= 0) {
            ids.push_back(nid);
            nid = parents[(size_t)nid];
        }
        std::reverse(ids.begin(), ids.end());
        path.resize((size_t)ids.size() * 3);
        for (size_t i = 0; i < ids.size(); ++i) {
            const int id = ids[i];
            path[3 * i + 0] = xs[(size_t)id];
            path[3 * i + 1] = ys[(size_t)id];
            path[3 * i + 2] = ths[(size_t)id];
        }
    }

    // Build numpy output
    py::array_t<double> path_arr({(py::ssize_t)(path.size() / 3), (py::ssize_t)3});
    auto pbuf = path_arr.mutable_unchecked<2>();
    for (py::ssize_t i = 0; i < pbuf.shape(0); ++i) {
        pbuf(i, 0) = path[(size_t)(3 * i + 0)];
        pbuf(i, 1) = path[(size_t)(3 * i + 1)];
        pbuf(i, 2) = path[(size_t)(3 * i + 2)];
    }

    py::dict stats;
    stats["expanded"] = expanded;
    stats["pushed"] = pushed;
    stats["collision_checks"] = collision_checks;
    stats["analytic_attempts"] = 0;
    stats["analytic_successes"] = 0;

    py::dict out;
    out["path"] = path_arr;
    out["stats"] = stats;
    return out;
}

PYBIND11_MODULE(hybrid_core, m) {
    m.doc() = "Hybrid A* C++ kernels (full search + debug expansion)";

    m.def(
        "expand_primitives",
        &expand_primitives,
        py::arg("x0"),
        py::arg("y0"),
        py::arg("th0"),
        py::arg("steer_angles"),
        py::arg("allow_reverse"),
        py::arg("step_size"),
        py::arg("n_substeps"),
        py::arg("wheelbase"),
        py::arg("occ_in"),
        py::arg("ox"),
        py::arg("oy"),
        py::arg("res"),
        py::arg("theta_bins"),
        py::arg("use_fp") = false,
        py::arg("footprint_offsets") = py::array_t<double>({0, 2}));

    m.def(
        "run_search",
        &run_search,
        py::arg("sx"),
        py::arg("sy"),
        py::arg("sth"),
        py::arg("gx"),
        py::arg("gy"),
        py::arg("gth"),
        py::arg("pos_tol"),
        py::arg("th_tol"),
        py::arg("max_expansions"),
        py::arg("occ_in"),
        py::arg("ox"),
        py::arg("oy"),
        py::arg("res"),
        py::arg("theta_bins"),
        py::arg("steer_angles"),
        py::arg("allow_reverse"),
        py::arg("step_size"),
        py::arg("n_substeps"),
        py::arg("wheelbase"),
        py::arg("reverse_penalty"),
        py::arg("switch_dir_penalty"),
        py::arg("h2d_dist"),
        py::arg("nh_table"),
        py::arg("nh_R"),
        py::arg("nh_res"),
        py::arg("nh_dth"),
        py::arg("nh_nxy"),
        py::arg("nh_theta_bins"),
        py::arg("use_rho") = false,
        py::arg("rho_in") = py::array_t<double>({1, 1}),
        py::arg("voronoi_weight") = 0.0,
        py::arg("use_fp") = false,
        py::arg("footprint_offsets") = py::array_t<double>({0, 2}));
}
