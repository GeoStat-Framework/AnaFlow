# Welcome to AnaFlow

[![DOI](https://zenodo.org/badge/116264578.svg)](https://zenodo.org/badge/latestdoi/116264578)
[![PyPI version](https://badge.fury.io/py/anaflow.svg)](https://badge.fury.io/py/anaflow)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://anaflow.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/AnaFlow/master/docs/source/pics/AnaFlow.png" alt="AnaFlow-LOGO" width="251px"/>
</p>

## Purpose

AnaFlow provides several analytical and semi-analytical solutions for the
groundwater-flow equation.


## Installation

You can install the latest version with the following command:

    pip install anaflow

It is highly recomended to install the scipy-scikit `umfpack` to get a solver
for sparse linear systems:

    pip install scikit-umfpack

Have a look at: https://github.com/scikit-umfpack/scikit-umfpack


## Documentation for AnaFlow

You can find the documentation under [geostat-framework.readthedocs.io][doc_link].


## Provided Functions

The following functions are provided directly

```python
anaflow.thiem        # Thiem solution for steady state pumping
anaflow.theis        # Theis solution for transient pumping
anaflow.ext_thiem2D  # extended Thiem solution in 2D
anaflow.ext_theis2D  # extended Theis solution in 2D
anaflow.ext_thiem3D  # extended Thiem solution in 3D
anaflow.ext_theis3D  # extended Theis solution in 3D
```


## Requirements

- [NumPy >= 1.10.0](https://www.numpy.org)
- [SciPy >= 0.19.0](https://www.scipy.org)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[GPL][gpl_link] © 2018-2019

[gpl_link]: https://github.com/GeoStat-Framework/AnaFlow/blob/master/LICENSE
[ogs5_link]: https://www.opengeosys.org/ogs-5/
[doc_link]: https://geostat-framework.readthedocs.io/projects/anaflow/en/latest/
