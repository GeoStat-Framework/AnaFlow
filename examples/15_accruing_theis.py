# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import theis


time = np.geomspace(1, 200)
rad = [1, 5]

# Q(t) = Q * erf(t / T)
lap_kwargs = {"cond": 4, "cond_kw": {"a": 100}}
head = theis(
    time=time,
    rad=rad,
    storage=1e-4,
    transmissivity=1e-4,
    rate=-1e-4,
    lap_kwargs=lap_kwargs,
)

for i, step in enumerate(rad):
    plt.plot(time, head[:, i], label="Theis(r={})".format(step))

plt.legend()
plt.tight_layout()
plt.show()
