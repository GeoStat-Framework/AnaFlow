# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import theis, ext_theis2D


time_labels = ["10 s", "10 min", "10 h"]
time = [10, 600, 36000]      # 10s, 10min, 10h
rad = np.geomspace(0.05, 4)  # radius from the pumping well in [0, 4]
var = 0.5                    # variance of the log-transmissivity
corr = 10.0                  # correlation length of the log-transmissivity
TG = 1e-4                    # the geometric mean of the transmissivity
TH = TG*np.exp(-var/2.0)     # the harmonic mean of the transmissivity
S = 1e-4                     # storativity
Qw = -1e-4                   # pumping rate

head_TG = theis(time=time, rad=rad, T=TG, S=S, Qw=Qw)
head_TH = theis(time=time, rad=rad, T=TH, S=S, Qw=Qw)
head_ef = ext_theis2D(time=time, rad=rad, TG=TG, sig2=var, corr=corr, S=S, Qw=Qw)
time_ticks=[]
for i, step in enumerate(time):
    if i == 0:
        label_TG = "Theis($T_G$)"
        label_TH = "Theis($T_H$)"
        label_ef = "extended Theis"
    else:
        label_TG = label_TH = label_ef = None
    plt.plot(rad, head_TG[i], label=label_TG, color="C"+str(i), linestyle="--")
    plt.plot(rad, head_TH[i], label=label_TH, color="C"+str(i), linestyle=":")
    plt.plot(rad, head_ef[i], label=label_ef, color="C"+str(i))
    time_ticks.append(head_TG[i][-1])

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
