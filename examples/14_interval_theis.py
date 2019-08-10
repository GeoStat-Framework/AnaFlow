# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import theis


time = np.linspace(10, 200)
rad = [1, 5]

# Q(t) = Q * characteristic([0, T])
lap_kwargs = {"cond": 3, "cond_kw": {"a": 100}}
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

plt.title("The Stehfest algorithm is not suitable for this!")
plt.legend()
plt.tight_layout()
plt.show()
