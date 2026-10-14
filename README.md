# onMEP paper (hbMEP adaptive)

This repository contains code to reproduce the results in the paper [Bayesian adaptive design for neural recruitment curves ](?).

It uses [onMEP v0.5.0](https://github.com/hbmep/onmep/tree/v0.5.0) and [hbMEP v0.9.0](https://github.com/hbmep/hbmep/tree/v0.9.0). See [`pyproject.toml`](https://github.com/hbmep/onmep-manuscript/blob/main/pyproject.toml) for dependencies.

[![Cite](https://img.shields.io/badge/stats.AP-arXiv:2610.15000-B31B1B.svg)](https://doi.org/10.48550/arXiv.2610.15000)
[![Cite](https://img.shields.io/badge/DOI-10.1016%2Fj.brs.2025.09.008-blue.svg)](https://doi.org/10.1016/j.brs.2025.09.008)
[![Cite](https://img.shields.io/badge/stats.AP-arXiv:2407.08709-B31B1B.svg)](https://doi.org/10.48550/arXiv.2407.08709)


## Installation

Begin by creating a virtual environment:

```bash
python3.11 -m venv .venv
```

The command above uses Python 3.11. If you have a different version of Python, you can use [Conda](https://conda.io) to create an environment with the required version:

```bash
conda create -n python-311 python=3.11.9 -y
conda activate python-311
python -m venv .venv
conda deactivate
```

Install the package in editable mode:

```bash
source .venv/bin/activate
pip install --upgrade pip
pip install -e .
```

The Python interpreter will be located at `.venv/bin/python`. You can use it to run the scripts in the [`notebooks`](https://github.com/hbmep/onmep-manuscript/tree/main/notebooks) directory.

## Citation

Please consider citing [Tyagi et al., 2026](https://doi.org/10.48550/arXiv.2610.15000) if you find this code useful in your research:

```bibtex
@misc{tyagi_bayesian_2026,
	title = {Bayesian adaptive design for neural recruitment curves},
	author = {Tyagi, Vishweshwar and Dukkipati, Saihari S. and Twili, Daniel and Mikhaeil, Jonas M. and Gelman, Andrew and Carmel, Jason B. and McIntosh, James R.},
	url = {https://doi.org/10.48550/arXiv.2610.15000},
	publisher = {arXiv},
	year = {2026},
	doi = {10.48550/arXiv.2610.15000},
}
```

as well as [Tyagi et al., 2025](https://doi.org/10.1016/j.brs.2025.09.008)

```bibtex
@article{tyagi_hierarchical_2025,
    title = {Hierarchical {Bayesian} estimation of motor-evoked potential recruitment curves yields accurate and robust estimates},
    author = {Tyagi, Vishweshwar and Murray, Lynda M. and Asan, Ahmet S. and Mandigo, Christopher and Virk, Michael S. and Harel, Noam Y. and Carmel, Jason B. and McIntosh, James R.},
    journal = {Brain Stimulation},
    year = {2025},
    doi = {10.1016/j.brs.2025.09.008},
}
```

Copyright © 2026 The Trustees of Columbia University in the City of New York. All rights reserved.
