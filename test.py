#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Sep  5 15:35:27 2018

@author: muellese
"""

#from scipy.special import gammainc, gammaincc, exp1, gamma
from matplotlib import pyplot as plt

import numpy as np

from scipy.special import gamma, gammaincc, exp1, expn
def inc_gamma(a, x):
    return exp1(x) if a == 0 else gamma(a)*gammaincc(a, x)

x = np.linspace(.1, 10.)
s = 0.00001

plt.plot(x, inc_gamma(0.1, x), label="inc_g(x)")
plt.plot(x, exp1(x), label="exp1(x)")

plt.legend()

plt.show()
print(inc_gamma(0, 0.1))