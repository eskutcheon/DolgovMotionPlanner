
import os, sys
from pathlib import Path
from setuptools import setup, find_packages

BUILD_CPP = os.environ.get("DOLGOV_BUILD_CPP", "0") in ("1", "true", "True")
ext_modules = []
cmdclass = {}

if BUILD_CPP:
    import numpy as np
    try:
        from pybind11.setup_helpers import Pybind11Extension, build_ext
    # except Exception as e:
    #     raise RuntimeError("pybind11 is required to build the C++ extension. Install with: pip install pybind11") from e
    except (ImportError, ModuleNotFoundError) as e:
        raise RuntimeError(
            "To build the C++ extension, install extras and set DOLGOV_BUILD_CPP=1:\n"
            "  pip install -e '.[cpp]'\n"
            "  (PowerShell) $env:DOLGOV_BUILD_CPP='1'\n"
        ) from e
    extra_compile_args = ["/O2", "/EHsc"] if sys.platform.startswith('win') else ['-O3']
    ext_modules = [
        Pybind11Extension(
            "hybrid_core",
            [str(Path(__file__).parent / "cpp" / "hybrid_core.cpp")],
            include_dirs=[np.get_include()],
            cxx_std=17,
            extra_compile_args=extra_compile_args,
        )
    ]
    cmdclass["build_ext": build_ext]

setup(
    name = 'dolgov-cbmp',
    version = '0.1.0',
    description = 'Hybrid A* path planning with optional C++ kernel',
    package_dir = {"": "src"},
    packages = find_packages(where="src", include=["dolgov_cbmp*"]),
    ext_modules = ext_modules,
    cmdclass = cmdclass,
    zip_safe = False,
)



# NOTE: The C++ code avoids global mutable state and releases the GIL inside `run_search`.
# Compiler flags:
#   - GCC/Clang: -O3
#   - MSVC: /O2     # NOTE: initially tested with MINGW (Target from `g++ -v`: x86_64-w64-mingw32)
