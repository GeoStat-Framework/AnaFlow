r"""
The integral variogram model
============================
"""

import numpy as np
from matplotlib import pyplot as plt

import anaflow as af


def dashes(i=1, max_n=12, width=1):
    return i * [width, width] + [max_n * 2 * width - 2 * i * width, width]


rad = np.linspace(0, 3, 1000)
rough = [0.01, 0.1, 0.25, 0.5, 1.0, 1.5, 2.0, 10.0]
# rescale for integral scale of 1
length = [1 / (nu * np.sqrt(np.pi) / (2 * nu + 2.0)) for nu in rough]
ln_gau = 2 / np.sqrt(np.pi)
var = 1
TG = 1e-4
TH = TG * np.exp(-var / 2)

for i, (ln, nu) in enumerate(zip(length, rough)):
    t = 1e4 * af.tools.Int_CG(rad, TG, var, ln, nu)
    plt.plot(
        rad,
        t,
        label=r"$T^{Int}_{CG}$($\nu$" + f"={nu:.4})",
        dashes=dashes(i),
        color="C0" if nu < 1 else "C2",
    )
plt.plot(
    rad,
    af.tools.T_CG(rad, TG, var, ln_gau) / TG,
    label=r"$T_{CG}$ - Gaussian",
    color="C1",
)
plt.title(f"$T_G=1$ cm²/s, $\\sigma^2={var}$")
plt.xlabel(r"r / $\ell$")
plt.ylabel(r"T / cm²s⁻¹")
plt.grid()
ax = plt.gca()
ax.legend()
ax.ticklabel_format(axis="y", style="scientific", scilimits=[-3, 0], useMathText=True)
ax.locator_params(tight=True, nbins=5)

ylim = ax.get_ylim()
ax2 = ax.twinx()
ax2.set_ylim(ylim)
ax2.set_yticks([TH * 1e4, TG * 1e4])
ax2.set_yticklabels([r"$T_H$", r"$T_G$"])

plt.tight_layout()
plt.show()
