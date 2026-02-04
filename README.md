
# Updated Hybrid A* Search for AGV Path Planning with Curvature Constraints

## Overview

DESCRIBE PROJECT, SOURCE, ETC

NOTE: initial version roughly implemented the approach from the 2008 paper without respect to curvature, while updates integrated it properly and made strides to update various aspects for efficiency


## Examples

- ADD VISUALIZATIONS AND STUFF HERE
- LINK TO COMPILED LATEX PDF AS DOCS


## Building & Testing (C++ extension)

### Prerequisites

- Python 3.10+
- (Optional) A C++ compiler with C++17 support
  - Linux: `g++` or `clang++`
  - Linux (Debian/Ubuntu): `sudo apt-get install build-essential python3-dev`
  - macOS: Xcode Command Line Tools
  - Windows: "Build Tools for Visual Studio" (MSVC)

### Python-first environment setup

From the repo root (`DolgovMotionPlanner`):

1. Create a virtual environment using Python's `venv`:

```bash
python -m venv .env
```

2. Activate the virtual environment:

a. For Linux users:

```bash
source .env/bin/activate
```

b. For Windows users:

```bash
.env/Scripts/activate
```

3. Finish environment setup and run initial tests with `pytest`:

a. Python only:

```bash
python -m pip install -U pip
pip install -e .
pytest -v
```

b. Optional C++ backend:

```bash
python -m pip install -U pip
pip install -e ".[cpp]"
$env:DOLGOV_BUILD_CPP="1" # on Linux: export DOLGOV_BUILD_CPP="1"
pip install -e . -v
pytest -m cpp
```

### Troubleshooting

- **`ImportError: No module named hybrid_core`**
  - The extension wasn’t built. Re-run `python -m pip install -e .`.

- **Windows/MSVC flag issues**
  - `setup.py` selects MSVC-friendly optimization flags (`/O2`). If your environment still errors, remove `extra_compile_args` temporarily and try again

- **Forcing a rebuild**
  - Delete `build/` and any `*.so`/`*.pyd` artifacts (or reinstall in a fresh virtualenv), then rerun `python -m pip install -e .`.



## Roadmap

- MENTION SUBPAR C++ INTEGRATION AND PLANS TO UPDATE
- MISC IDEAS FOR IMPROVEMENTS