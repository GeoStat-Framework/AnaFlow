r"""
Convergence of the extended Theis solutions for the Integral model
==================================================================

Here we set an outer boundary to the transient solution, so this condition
coincides with the references head of the steady solution.
"""

import numpy as np
from matplotlib import pyplot as plt

from anaflow import ext_theis_int, ext_thiem_int

time = 86400  # time point for steady state (1 day)
rad = np.geomspace(0.05, 5) / 5  # radius from the pumping well in [0, 1]
r_ref = 2  # reference radius

KG = 1e-4  # the geometric mean of the transmissivity
len_scale = 1  # correlation length of the log-transmissivity
roughness = 1  # roughness parameter
var = 1  # variance of the log-transmissivity
S = 1e-4  # storativity
rate = -1e-4  # pumping rate
dim = 2

head1 = ext_thiem_int(rad, r_ref, KG, var, len_scale, roughness, dim=dim, rate=rate)
head2 = ext_theis_int(
    time,
    rad,
    S,
    KG,
    var,
    len_scale,
    roughness,
    dim=dim,
    rate=rate,
    r_bound=r_ref,
)

plt.plot(rad, head1, label=r"Thiem$^{Int}_{CG}$", linewidth=3, color="k")
plt.plot(
    rad,
    head2,
    label=r"Theis$^{Int}_{CG}$" + f"(t=1 day)",
    linestyle="--",
    linewidth=3,
    color="C1",
)
plt.title(
    f"$T_G=1$ cm²/s, $\\sigma^2={var}$, $\\nu={roughness}$, $S={S}$, $Q=0.1$ L/s, "
    + r"$r_{ref}=2\ell$"
)

plt.xlabel(r"r / $\ell$")
plt.ylabel("h / m")
plt.grid()
plt.gca().set_ylim([-1.25, 0])
plt.gca().set_xlim([0, 1])
plt.gca().locator_params(tight=True, nbins=5)
plt.tight_layout()
plt.legend(loc="lower right")
plt.show()
