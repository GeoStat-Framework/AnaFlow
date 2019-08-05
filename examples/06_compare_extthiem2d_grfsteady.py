# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import ext_thiem_2d, ext_grf_steady
from anaflow.tools.coarse_graining import T_CG


rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 2.0                  # reference radius
var = 0.5                    # variance of the log-transmissivity
len_scale = 10.0             # correlation length of the log-transmissivity
TG = 1e-4                    # the geometric mean of the transmissivity
rate = -1e-4                 # pumping rate

head1 = ext_thiem_2d(rad, r_ref, TG, var, len_scale, rate)
head2 = ext_grf_steady(rad, r_ref, T_CG, rate=rate, trans_gmean=TG, var=var, len_scale=len_scale)

plt.plot(rad, head1, label="Ext Thiem 2D")
plt.plot(rad, head2, label="grf(T_CG)", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
