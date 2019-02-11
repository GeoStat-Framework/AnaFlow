# Welcome to AnaFlow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1135723.svg)](https://doi.org/10.5281/zenodo.1135723)
[![PyPI version](https://badge.fury.io/py/anaflow.svg)](https://badge.fury.io/py/anaflow)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://anaflow.readthedocs.io/en/latest/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/ambv/black)

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/AnaFlow/master/docs/source/pics/Anaflow.png" alt="AnaFlow-LOGO" width="251px"/>
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


### Example

In the following the well known Theis function is called an plotted for three
different time-steps.

```python
import numpy as np
from matplotlib import pyplot as plt
from anaflow import theis


time = [10, 100, 1000]
rad = np.geomspace(0.1, 10)

head = theis(rad=rad, time=time, T=1e-4, S=1e-4, Qw=-1e-4)

for i, step in enumerate(time):
    plt.plot(rad, head[i], label="Theis(t={})".format(step))

plt.legend()
plt.show()
```

<p align="center">
<img src="https://raw.githubusercontent.com/GeoStat-Framework/AnaFlow/master/docs/source/pics/01_call_theis.png" alt="Theis" width="600px"/>
</p>


### Provided Functions

The following functions are provided directly

```python
anaflow.thiem        # Thiem solution for steady state pumping
anaflow.theis        # Theis solution for transient pumping
anaflow.ext_thiem2D  # extended Thiem solution in 2D
anaflow.ext_theis2D  # extended Theis solution in 2D
anaflow.ext_thiem3D  # extended Thiem solution in 3D
anaflow.ext_theis3D  # extended Theis solution in 3D
```


### Laplace Transformation

We provide routines to calculate the laplace-transformation as well as the
inverse laplace-transformation of a given function

```python
anaflow.get_lap      # Get the laplace transformation of a function
anaflow.get_lap_inv  # Get the inverse laplace transformation of a function
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
