# -*- coding: utf-8 -*-
import numpy as np
from matplotlib import pyplot as plt
from anaflow import theis, ext_theis3D
from anaflow.tools.special import aniso


time_labels = ["10 s", "10 min", "10 h"]
time = [10, 600, 36000]                 # 10s, 10min, 10h
rad = np.geomspace(0.05, 4)             # radial distance from the pumping well in [0, 4]
var = 0.5                               # variance of the log-conductivity
corr = 10.0                             # correlation length of the log-conductivity
e = 0.75                                # anisotropy ratio of the log-conductivity
KG = 1e-4                               # the geometric mean of the conductivity
Kefu = KG*np.exp(var*(0.5 - aniso(e)))  # the effective conductivity for uniform flow
KH = KG*np.exp(-var/2.0)                # the harmonic mean of the conductivity
S = 1e-4                                # storage
Qw = -1e-4                              # pumping rate
L = 1.0                                 # vertical extend of the aquifer

head_Kefu = theis(time=time, rad=rad, T=Kefu*L, S=S, Qw=Qw)
head_KH = theis(time=time, rad=rad, T=KH*L, S=S, Qw=Qw)
head_ef = ext_theis3D(time=time, rad=rad, KG=KG, sig2=var, corr=corr, e=e, S=S, Qw=Qw, L=1)
time_ticks=[]
for i, step in enumerate(time):
    label_TG = "Theis($K_{efu}$)" if i == 0 else None
    label_TH = "Theis($K_H$)" if i == 0 else None
    label_ef = "extended Theis 3D" if i == 0 else None
    plt.plot(rad, head_Kefu[i], label=label_TG, color="C"+str(i), linestyle="--")
    plt.plot(rad, head_KH[i], label=label_TH, color="C"+str(i), linestyle=":")
    plt.plot(rad, head_ef[i], label=label_ef, color="C"+str(i))
    time_ticks.append(head_ef[i][-1])

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
