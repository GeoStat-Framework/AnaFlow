# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import thiem, grf_steady


rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
r_ref = 20.0                 # reference radius
T = 1e-4                     # the transmissivity
rate = -1e-4                 # pumping rate

head1 = thiem(rad, r_ref, T, rate)
head2 = grf_steady(rad, r_ref, T, rate=rate)

plt.plot(rad, head1, label="Thiem")
plt.plot(rad, head2, label="grf(T)", linestyle="--")

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
plt.tight_layout()
plt.show()
