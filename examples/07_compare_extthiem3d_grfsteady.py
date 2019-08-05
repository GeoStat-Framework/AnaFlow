# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import ext_thiem_3d, ext_grf_steady
from anaflow.tools.coarse_graining import K_CG


rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 2.0                  # reference radius
var = 0.5                    # variance of the log-transmissivity
len_scale = 10.0             # correlation length of the log-transmissivity
KG = 1e-4                    # the geometric mean of the transmissivity
anis = 0.7                   # aniso ratio
rate = -1e-4                 # pumping rate

head1 = ext_thiem_3d(rad, r_ref, KG, var, len_scale, anis, 1, rate)
head2 = ext_grf_steady(rad, r_ref, K_CG, rate=rate, cond_gmean=KG, var=var, len_scale=len_scale, anis=anis)

plt.plot(rad, head1, label="Ext Thiem 3D")
plt.plot(rad, head2, label="grf(K_CG)", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
