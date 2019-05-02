# Copyright (c) 2015, Disney Research
# All rights reserved.
#
# Author(s): Sehoon Ha <sehoon.ha@disneyresearch.com>
# Disney Research Robotics Group


import numpy as np


def grad(fun, x, h):
    n = len(x)
    g = np.zeros(n)
    for i in range(n):
        dx = np.zeros(n)
        dx[i] = h
        f1 = fun(x - dx)
        f2 = fun(x + dx)
        g[i] = (0.5 * f2 - 0.5 * f1) / h
    return g
