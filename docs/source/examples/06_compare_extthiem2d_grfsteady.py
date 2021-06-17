r"""
extended Thiem 2D vs. steady solution for coarse graining transmissivity
========================================================================

The extended Thiem 2D solutions is the analytical solution of the groundwater
flow equation for the coarse graining transmissivity for pumping tests.
Therefore the results should coincide.

References:

- `Schneider & Attinger 2008 <https://doi.org/10.1029/2007WR005898>`__
- `Zech & Attinger 2016 <https://doi.org/10.5194/hess-20-1655-2016>`__
"""
import numpy as np
from matplotlib import pyplot as plt
from anaflow import ext_thiem_2d, ext_grf_steady
from anaflow.tools.coarse_graining import T_CG


rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 2.0  # reference radius

var = 0.5  # variance of the log-transmissivity
len_scale = 10.0  # correlation length of the log-transmissivity
TG = 1e-4  # the geometric mean of the transmissivity

rate = -1e-4  # pumping rate

head1 = ext_thiem_2d(rad, r_ref, TG, var, len_scale, rate)
head2 = ext_grf_steady(
    rad, r_ref, T_CG, rate=rate, trans_gmean=TG, var=var, len_scale=len_scale
)

plt.plot(rad, head1, label="Ext Thiem 2D")
plt.plot(rad, head2, label="grf(T_CG)", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
