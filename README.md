
AnaFlow: A python-package containing analytical solutions for the groundwater flow equation
===========================================================================================
[![DOI](https://zenodo.org/badge/116264578.svg)](https://zenodo.org/badge/latestdoi/116264578)

You can find the documentation under: http://anaflow.readthedocs.io

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
Just download the code an run the following command from the
source code directory:

    pip install -U .

It is highly recomended to install the scipy-scikit `umfpack` to get a solver
for sparse linear systems:

    pip install -U scikit-umfpack

Have a look at: https://pypi.python.org/pypi/scikit-umfpack

Created December 2017, Copyright Sebastian Mueller 2017
