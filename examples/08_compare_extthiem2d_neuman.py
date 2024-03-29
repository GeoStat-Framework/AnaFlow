r"""
extended Thiem 2D vs. steady solution for apparent transmissivity from Neuman
=============================================================================

Both, the extended Thiem and the Neuman solution, represent an effective steady
drawdown in a heterogeneous aquifer.
In both cases the heterogeneity is represented by two point statistics,
characterized by mean, variance and length scale of the log transmissivity field.
Therefore these approaches should lead to similar results.

References:

- `Neuman 2004 <https://doi.org/10.1029/2003WR002405>`__
- `Zech & Attinger 2016 <https://doi.org/10.5194/hess-20-1655-2016>`__
"""
import numpy as np
from matplotlib import pyplot as plt

from anaflow import ext_thiem_2d, neuman2004_steady

rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 30.0  # reference radius

var = 0.5  # variance of the log-transmissivity
len_scale = 10.0  # correlation length of the log-transmissivity
TG = 1e-4  # the geometric mean of the transmissivity

rate = -1e-4  # pumping rate

head1 = ext_thiem_2d(rad, r_ref, TG, var, len_scale, rate)
head2 = neuman2004_steady(rad, r_ref, TG, var, len_scale, rate)

plt.plot(rad, head1, label="extended Thiem 2D")
plt.plot(rad, head2, label="Steady Neuman 2004", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
