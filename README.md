# Photon Weave
![Coverage](assets/coverage.svg)
![Build Status](https://github.com/tqsd/photon_weave/actions/workflows/tests.yml/badge.svg)
[![DOI](https://joss.theoj.org/papers/10.21105/joss.07468/status.svg)](https://doi.org/10.21105/joss.07468)

Photon Weave is a quantum optics simulator designed for the modeling and analysis of quantum systems. Focusing on individual temporal modes, it offers comprehensive support for simulating quantum states within Fock spaces along with their polarization degrees of freedom.

## Installation

This package can be installed using pip:
```bash
pip install photon-weave
```
or it can be installed from this repository:
```bash
pip install git+https://github.com/tqsd/photon_weave.git
```

### Installation for developing
In case you want to add a feature, use Poetry to keep the dependency graph consistent (Python 3.12–3.14):
```bash
git clone git@github.com:tqsd/photon_weave.git
cd photon_weave
poetry install --with dev
```

If you prefer a minimal pip-based setup:
```bash
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -e ".[dev]"
```


#### Testing
The tests can simply be run with the `pytest` testing suite. After installing with Poetry, run:
```
JAX_PLATFORMS=cpu poetry run pytest
```
