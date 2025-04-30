"""
Impact of roughness on effective drawdown on anti-persistent fields
===================================================================

When the field roughness approaches zero, the drawdown approaches
the homogeneous solution for the geometric mean transmissivity.
"""

import numpy as np
from matplotlib import pyplot as plt

import anaflow as af

min_s = 60
hour_s = min_s * 60
day_s = hour_s * 24


def dashes(i=1, max_n=12, width=1):
    return i * [width, width] + [max_n * 2 * width - 2 * i * width, width]


time_labels = ["10 s", "30 min", "steady\nshape"]
time = [10, 30 * min_s, 10 * day_s]  # 10s, 30min, 10d
rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]

TG = 1e-4  # the geometric mean of the transmissivity
var = 2.25  # variance of the log-transmissivity
len_scale = 10  # correlation length of the log-transmissivity

# different roughness values from very rough to very smooth
rough = [0.01, 0.1, 0.25, 0.5, 1.0]
S = 1e-4  # storativity
rate = -1e-4  # pumping rate
heads = []
hom_theis = af.theis(time, rad, S, TG, rate=rate)
ext_theis = af.ext_theis_2d(
    time, rad, S, TG, var, len_scale / np.sqrt(np.pi) * 2, rate=rate
)

for roughness in rough:
    # rescale to constant integral scale
    ln = len_scale / (roughness * np.sqrt(np.pi) / (2 * roughness + 2.0))
    # ln = len_scale
    heads.append(
        af.ext_theis_int(time, rad, S, TG, var, ln, roughness, rate=rate, parts=50)
    )

time_ticks = []
for i, step in enumerate(time):
    for j, roughness in enumerate(rough):
        label = (
            r"Theis$^{Int}_{CG}(\nu=" + f"{roughness:.4})$"
            if i == len(time) - 1
            else None
        )
        plt.plot(
            rad,
            heads[j][i],
            label=label,
            color=f"C0",
            dashes=dashes(j),
            alpha=(1 + i) / len(time),
        )
    time_ticks.append(hom_theis[i][-1])
    label = "Theis($T_G$)" if i == len(time) - 1 else None
    plt.plot(rad, hom_theis[i], label=label, color="k")  # , dashes=dashes(1))


plt.title(
    f"$T_G=1$ cm²/s, $\\sigma^2={var}$, $\\ell={len_scale}$ m, $S={S}$, $Q={-rate*1000}$ L/s"
)
plt.xlabel("r / m")
plt.ylabel("h / m")
plt.grid()
plt.legend()
plt.gca().locator_params(tight=True, nbins=5)
plt.gca().set_ylim([-3.2, 0.1])
plt.gca().set_xlim([0, rad[-1]])
ylim = plt.gca().get_ylim()
ax2 = plt.gca().twinx()
ax2.set_yticks(time_ticks)
ax2.set_yticklabels(time_labels)
ax2.set_ylim(ylim)
plt.tight_layout()
plt.show()
