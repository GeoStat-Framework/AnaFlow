# Welcome to AnaFlow

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.1135723.svg)](https://doi.org/10.5281/zenodo.1135723)
[![PyPI version](https://badge.fury.io/py/anaflow.svg)](https://badge.fury.io/py/anaflow)
[![Build Status](https://travis-ci.com/GeoStat-Framework/AnaFlow.svg?branch=master)](https://travis-ci.com/GeoStat-Framework/AnaFlow)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=stable)](https://anaflow.readthedocs.io/en/stable/)
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

head = theis(time=time, rad=rad, transmissivity=1e-4, storage=1e-4, rate=-1e-4)

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

* ``thiem`` Thiem solution for steady state pumping
* ``theis`` Theis solution for transient pumping
* ``ext_thiem_2d`` extended Thiem solution in 2D from *Zech 2013*
* ``ext_theis_2d`` extended Theis solution in 2D from *Mueller 2015*
* ``ext_thiem_3d`` extended Thiem solution in 3D from *Zech 2013*
* ``ext_theis_3d`` extended Theis solution in 3D from *Mueller 2015*
* ``neuman2004`` transient solution from *Neuman 2004*
* ``neuman2004_steady`` steady solution from *Neuman 2004*
* ``grf`` "General Radial Flow" Model from *Barker 1988*
* ``ext_grf`` the transient extended GRF model
* ``ext_grf_steady`` the steady extended GRF model
* ``ext_thiem_tpl`` extended Thiem solution for truncated power laws
* ``ext_theis_tpl`` extended Theis solution for truncated power laws
* ``ext_thiem_tpl_3d`` extended Thiem solution in 3D for truncated power laws
* ``ext_theis_tpl_3d`` extended Theis solution in 3D for truncated power laws


### Laplace Transformation

We provide routines to calculate the laplace-transformation as well as the
inverse laplace-transformation of a given function

* ``get_lap`` Get the laplace transformation of a function
* ``get_lap_inv`` Get the inverse laplace transformation of a function


## Requirements

- [NumPy >= 1.14.5](https://www.numpy.org)
- [SciPy >= 1.1.0](https://www.scipy.org)
- [pentapy >= 1.1.0](https://github.com/GeoStat-Framework/pentapy)


## Contact

You can contact us via <info@geostat-framework.org>.


## License

[MIT][mit_link] © 2019 - 2020

[mit_link]: https://github.com/GeoStat-Framework/AnaFlow/blob/master/LICENSE
[doc_link]: https://anaflow.readthedocs.io
