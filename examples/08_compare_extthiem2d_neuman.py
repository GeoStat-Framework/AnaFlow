# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import ext_thiem_2d, neuman2004_steady


rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 30.0                 # reference radius
var = 0.5                    # variance of the log-transmissivity
len_scale = 10.0             # correlation length of the log-transmissivity
TG = 1e-4                    # the geometric mean of the transmissivity
rate = -1e-4                 # pumping rate

head1 = ext_thiem_2d(rad, r_ref, TG, var, len_scale, rate)
head2 = neuman2004_steady(rad, r_ref, TG, var, len_scale, rate)

plt.plot(rad, head1, label="extended Thiem 2D")
plt.plot(rad, head2, label="Steady Neuman 2004", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
