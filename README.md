AnaFlow: A python-package containing analytical solutions for groundwater flow
==============================================================================
[![DOI](https://zenodo.org/badge/116264578.svg)](https://zenodo.org/badge/latestdoi/116264578)
<<<<<<< HEAD

<p align="center">
<img src="/docs/source/Anaflow.png" alt="AnaFlow-LOGO" width="251px"/>
</p>

You can find the documentation under: http://anaflow.readthedocs.io
=======
[![PyPI version](https://badge.fury.io/py/anaflow.svg)](https://badge.fury.io/py/anaflow)
[![Documentation Status](https://readthedocs.org/projects/docs/badge/?version=latest)](https://anaflow.readthedocs.io/en/latest/)
>>>>>>> 6d35523d596e8942c004c35a101a86f83584f18d

Contents
--------
Anaflow provides several analytical and semi-analytical solutions for the
groundwater-flow-equation.

Functions
---------
The following functions are provided directly

 - `thiem      ` -- Thiem solution for steady state pumping
 - `theis      ` -- Theis solution for transient pumping
 - `ext_thiem2D` -- extended Thiem solution in 2D
 - `ext_theis2D` -- extended Theis solution in 2D
 - `ext_thiem3D` -- extended Thiem solution in 3D
 - `ext_theis3D` -- extended Theis solution in 3D
 - `diskmodel  ` -- Solution for a diskmodel
 - `stehfest   ` -- Stehfest algorithm for laplace inversion

Subpackages
-----------
Using any of these subpackages requires an explicit import.
For example: ``import anaflow.helper``

 - `gwsolutions` -- Solutions for the groundwater flow equation
 - `laplace    ` -- Functions concerning the laplace-transform
 - `helper     ` -- Several helper-functions

Installation
------------
AnaFlow is on [PyPI](https://pypi.org/project/anaflow/). You just need to run
the following command:

    pip install -U anaflow

If you want the latest version, just download the
[code](https://github.com/MuellerSeb/AnaFlow/archive/master.zip)
and run the following command from the source code directory:

    pip install -U .

It is highly recomended to install the scipy-scikit `umfpack` to get a solver
for sparse linear systems:

    pip install -U scikit-umfpack

Under Ubuntu you can install the required SuiteSparse library with:

    sudo apt-get install libsuitesparse-dev

For further information have a look at:
 - http://pypi.python.org/pypi/scikit-umfpack
 - http://faculty.cse.tamu.edu/davis/suitesparse.html

Dependencies
------------
 - [NumPy](http://www.numpy.org): 1.10.0 or higher
 - [SciPy](http://www.scipy.org): 0.19.0 or higher
 - [scikit-umfpack](http://pypi.python.org/pypi/scikit-umfpack): recomended

<<<<<<< HEAD
Created December 2017, Copyright Sebastian Mueller 2017-2018
=======
[![ForTheBadge built-with-science](http://ForTheBadge.com/images/badges/built-with-science.svg)](https://GitHub.com/Naereen/)

Created December 2017, Copyright Sebastian Mueller 2017
>>>>>>> 6d35523d596e8942c004c35a101a86f83584f18d
