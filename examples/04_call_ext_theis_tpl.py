# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import theis, ext_theis_tpl
from anaflow.tools.special import aniso


time_labels = ["10 s", "10 min", "10 h"]
time = [10, 600, 36000]      # 10s, 10min, 10h
rad = np.geomspace(0.05, 4)  # radial distance from the pumping well in [0, 4]
S = 1e-4                     # storage
KG = 1e-4                    # the geometric mean of the conductivity
corr = 20.0                  # upper bound for the length scale
hurst=0.5                    # hurst coefficient
var = 0.5                    # variance of the log-conductivity
Qw = -1e-4                   # pumping rate
KH = KG*np.exp(-var/2.0)     # the harmonic mean of the conductivity

head_KG = theis(time=time, rad=rad, T=KG, S=S, Qw=Qw)
head_KH = theis(time=time, rad=rad, T=KH, S=S, Qw=Qw)
head_ef = ext_theis_tpl(
    time=time,
    rad=rad,
    S=S,
    KG=KG,
    corr=corr,
    hurst=hurst,
    sig2=var,
    Qw=Qw,
)
time_ticks=[]
for i, step in enumerate(time):
    if i == 0:
        label_TG = "Theis($K_G$)"
        label_TH = "Theis($K_H$)"
        label_ef = "extended Theis TPL 2D"
    else:
        label_TG = label_TH = label_ef = None
    plt.plot(rad, head_KG[i], label=label_TG, color="C"+str(i), linestyle="--")
    plt.plot(rad, head_KH[i], label=label_TH, color="C"+str(i), linestyle=":")
    plt.plot(rad, head_ef[i], label=label_ef, color="C"+str(i))
    time_ticks.append(head_KG[i][-1])

plt.xlabel("r in [m]")
plt.ylabel("h in [m]")
plt.legend()
ylim = plt.gca().get_ylim()
plt.gca().set_xlim([0, rad[-1]])
ax2 = plt.gca().twinx()
ax2.set_yticks(time_ticks)
ax2.set_yticklabels(time_labels)
ax2.set_ylim(ylim)
plt.show()