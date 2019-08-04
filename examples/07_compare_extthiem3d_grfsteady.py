# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import ext_thiem3D, grf_steady
from anaflow.tools.coarse_graining import K_CG


rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 2.0                  # reference radius
var = 0.5                    # variance of the log-transmissivity
len_scale = 10.0             # correlation length of the log-transmissivity
KG = 1e-4                    # the geometric mean of the transmissivity
e = 0.7                      # aniso ratio
rate = -1e-4                 # pumping rate

head1 = ext_thiem3D(rad, r_ref, KG, var, len_scale, e, rate, L=1)
head2 = grf_steady(rad, r_ref, K_CG, rate=rate, KG=KG, sig2=var, corr=len_scale, e=e)

plt.plot(rad, head1, label="Ext Thiem 3D")
plt.plot(rad, head2, label="grf(K_CG)", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
