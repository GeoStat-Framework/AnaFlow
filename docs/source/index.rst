==================
AnaFlow Quickstart
==================

.. image:: pics/Anaflow.png
   :width: 150px
   :align: center

AnaFlow provides several analytical and semi-analytical solutions for the
groundwater-flow equation.


Installation
============

The package can be installed via `pip <https://pypi.org/project/gstools/>`_.
On Windows you can install `WinPython <https://winpython.github.io/>`_ to get
Python and pip running.

.. code-block:: none

    pip install anaflow


It is highly recomended to install the scipy-scikit `umfpack` to get a solver
for sparse linear systems:

.. code-block:: none

    pip install scikit-umfpack

Have a look at: https://github.com/scikit-umfpack/scikit-umfpack


Provided Functions
==================

The following functions are provided directly

.. code-block:: python

    anaflow.thiem        # Thiem solution for steady state pumping
    anaflow.theis        # Theis solution for transient pumping
    anaflow.ext_thiem2D  # extended Thiem solution in 2D
    anaflow.ext_theis2D  # extended Theis solution in 2D
    anaflow.ext_thiem3D  # extended Thiem solution in 3D
    anaflow.ext_theis3D  # extended Theis solution in 3D


Laplace Transformation
======================

We provide routines to calculate the laplace-transformation as well as the
inverse laplace-transformation of a given function

.. code-block:: python

    anaflow.get_lap      # Get the laplace transformation of a function
    anaflow.get_lap_inv  # Get the inverse laplace transformation of a function


Requirements
============

- `NumPy >= 1.10.0 <https://www.numpy.org>`_
- `SciPy >= 0.19.0 <https://www.scipy.org/>`_


License
=======

`GPL <https://github.com/GeoStat-Framework/AnaFlow/blob/master/LICENSE>`_ © 2018
