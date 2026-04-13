"""
ODE system solvers
version 03.03.2026
@author: serpinsky
"""

import numpy as np

def Euler(ode_system, init_conditions, stop_conditions, report, dx, x_0 = 0, max_steps = 100_000):

    x = np.zeros(max_steps)
    m = len(init_conditions) + 1
    n = len(report(x_0, init_conditions))
    res = np.zeros((max_steps, m + n))

    i = 0
    y = init_conditions
    x[i] = x_0
    res[i, 0] = x[i]
    res[i, 1: m] = y
    res[i, m: m + n] = report(x[i], y)

    while stop_conditions(x[i], y) > 0 and i + 1 < max_steps:
        k = ode_system(x[i], y)
        i += 1
        y += k * dx
        x[i] = x[i - 1] + dx
        res[i, 0] = x[i]
        res[i, 1: m] = y
        res[i, m: m + n] = report(x[i], y)
    return res[0: i + 1, :]

def RungeKutta4(ode_system, init_conditions, stop_conditions, report, dx, x_0 = 0, max_steps = 100_000):

    x = np.zeros(max_steps)
    m = len(init_conditions) + 1
    n = len(report(x_0, init_conditions))
    res = np.zeros((max_steps, m + n))

    i = 0
    y = init_conditions
    x[i] = x_0
    res[i, 0] = x[i]
    res[i, 1: m] = y
    res[i, m: m + n] = report(x[i], y)

    while stop_conditions(x[i], y) > 0 and i + 1 < max_steps:
        k_1 = ode_system(x[i], y)
        k_2 = ode_system(x[i] + 0.5 * dx, y + k_1 * 0.5 * dx)
        k_3 = ode_system(x[i] + 0.5 * dx, y + k_2 * 0.5 * dx)
        k_4 = ode_system(x[i] + dx, y + k_3 * dx)
        i += 1
        y += (k_1 + 2 * k_2 + 2 * k_3 + k_4) * dx / 6
        x[i] = x[i - 1] + dx
        res[i, 0] = x[i]
        res[i, 1: m] = y
        res[i, m: m + n] = report(x[i], y)
    return res[0: i + 1, :]