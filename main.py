from math import *
from ODE_solvers import *
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
from pyballistics import ozvb_lagrange

ds = pd.read_csv('СМ6-69_Пальников_СД_data.csv', skiprows=[10]).set_index('var').to_dict()

[d, q, n_s, K, p_ign, p_0, Delta, omega_q, l_m_d, I_en, f_n, k, T_v, delta,
 b, z_e, kappa_1, lambda_1, kappa_2, lambda_2, k_I, k_f, T_ref, Pr, lambda_g,
 mu_g0, T_0s, T_cs, rho_b, c_b, lambda_b, p_a0, k_a, c_a0, T_01, T_02,
 T_03, time_step] = list(ds['value'].values())

T_0_min = T_01
T_0_norm = T_02
T_0_max = T_03
v_pm = 995
p_max = 355
omega = q * omega_q
l_m = l_m_d * d
W_0 = omega / Delta

psi_s = kappa_1 * (1 + lambda_1)

I_e = I_en
f_ign = f_n
b_ign = b

S = n_s * pi * d ** 2 / 4
l_0 = W_0 / S

zeta = (1 / Delta - 1 / delta) / (f_ign / p_ign + b)
phi = K + 1 / 3 * omega * (1 + zeta) / q

R_g = f_n / T_v
S_w0 = 4 * W_0 / d


def f(T_0):
    return f_n * (1 + k_f * (T_0 - T_ref))


def I_e(T_0):
    return I_en * (1 - k_I * (T_0 - T_ref))


def p_a(x):
    return p_a0 * (1 + k_a * (k_a + 1) / 4 * (x[1] / c_a0) ** 2 + k_a * x[1] / c_a0 * (
            1 + ((k_a + 1) / 4 * (x[1] / c_a0)) ** 2) ** (1 / 2))


def psi(z):
    return kappa_1 * z * (1 + lambda_1 * z) * np.heaviside(1 - z, 0.5) + (
            psi_s + kappa_2 * (z - 1) * (1 + lambda_2 * (z - 1))) * np.heaviside(z - 1, 0.5)


def p_m(x, T_0):
    return (f(T_0) * omega * psi(x[5]) + f_ign * omega * zeta - (k - 1) * (phi * q * x[1] ** 2 / 2 + x[2] + x[3])) / (
            S * x[0] - omega / delta * (1 - psi(x[5])) - b * omega * (psi(x[5]) + zeta))


def system(t, x, T_0):
    T = p_m(x, T_0) / R_g * (1 / (psi(x[5]) + zeta) * (x[0] / (l_0 * Delta) - 1 / delta + psi(x[5]) / delta) - b)
    mu_g = mu_g0 * (T_0s + T_cs) / (T + T_cs) * (T / T_0s) ** 1.5
    Re = (omega * (1 + zeta) * x[1] * d) / (2 * S * x[0] * mu_g)
    Nu = 0.023 * Re ** 0.8 * Pr ** 0.4
    q_w = Nu * lambda_g / d * (T - T_0 - (x[4]) ** (1 / 2))
    l_p = x[0] - l_0
    S_w = S_w0 + pi * d * l_p
    dx_dt = np.zeros(6)
    dx_dt[0] = x[1]
    dx_dt[1] = (p_m(x, T_0) - p_a(x)) * S / (phi * q) * np.heaviside(p_m(x, T_0) - p_a(x) - p_0, 0.5)
    dx_dt[2] = p_a(x) * S * x[1]
    dx_dt[3] = S_w * q_w
    dx_dt[4] = 2 * q_w ** 2 / (c_b * rho_b * lambda_b) - 2 * x[1] / x[0] * x[4]
    dx_dt[5] = p_m(x, T_0) / I_e(T_0) * np.heaviside(z_e - x[5], 0.5)
    return dx_dt


init = [l_0, 0, 0, 0, 0, 0]

dt = time_step
max_steps = 10000


def stop(t, x):
    return l_m - (x[0] - l_0)


def sys_min(t, x):
    return system(t, x, T_0_min)


def sys_norm(t, x):
    return system(t, x, T_0_norm)


def sys_max(t, x):
    return system(t, x, T_0_max)


def report_min(t, x):
    return [psi(x[5]), p_m(x, T_0_min) / 1e6, x[0] - l_0]


def report_norm(t, x):
    return [psi(x[5]), p_m(x, T_0_norm) / 1e6, x[0] - l_0]


def report_max(t, x):
    return [psi(x[5]), p_m(x, T_0_max) / 1e6, x[0] - l_0]


def create_opts_for_lagrange(csv_path: str, powder_name: str = '180/57',
                             T_0: float = 288.15) -> dict:
    ds = pd.read_csv(csv_path, skiprows=[10]).set_index('var').to_dict()
    vals = ds['value']

    omega = vals['q'] * vals['omega/q']
    W_0 = omega / vals['Delta']
    l_m = vals['l_m/d'] * vals['d']

    opts = {
        'powders': [{'omega': omega, 'dbname': powder_name}],
        'init_conditions': {
            'q': vals['q'],
            'd': vals['d'],
            'W_0': W_0,
            'T_0': T_0,
            'phi_1': vals['K'],
            'p_0': vals['p_0']
        },
        'igniter': {
            'p_ign_0': vals['p_ign'],
            'k_ign': 1.25,
            'T_ign': 2427,
            'f_ign': 260000.0,
            'b_ign': 0.0006
        },
        'meta_lagrange': {
            'CFL': 0.9,
            'n_cells': 300
        },
        'stop_conditions': {
            'v_p': 995.0,
            'x_p': l_m
        }
    }

    return opts


def calculate_mean_pressure(p_array, x_array):
    x_p = x_array[-1] - x_array[0]
    if x_p <= 0:
        return np.mean(p_array)

    integral = np.trapezoid(p_array, x_array[:-1])
    return integral / x_p


def find_reference_points_lagrange(result, x_initial=0.0):
    layers = result['layers']

    t_vals = []
    p_b_vals = []
    p_p_vals = []
    p_m_vals = []
    v_p_vals = []
    l_p_vals = []
    z_vals = []
    psi_vals = []

    for layer in layers:
        x = layer['x']
        u = layer['u']
        p = layer['p']

        t_vals.append(layer['t'] * 1000)
        p_b_vals.append(p[0] / 1e6)
        p_p_vals.append(p[-1] / 1e6)
        p_m_vals.append(calculate_mean_pressure(p, x) / 1e6)
        v_p_vals.append(u[-1])
        l_p_vals.append(x[-1] - x_initial)
        z_vals.append(layer.get('z_1', [0])[-1] if 'z_1' in layer else 0)
        psi_vals.append(layer.get('psi_1', [0])[-1] if 'psi_1' in layer else 0)

    t_vals = np.array(t_vals)
    p_b_vals = np.array(p_b_vals)
    p_p_vals = np.array(p_p_vals)
    p_m_vals = np.array(p_m_vals)
    v_p_vals = np.array(v_p_vals)
    l_p_vals = np.array(l_p_vals)
    z_vals = np.array(z_vals)
    psi_vals = np.array(psi_vals)

    points = {}

    idx_0 = np.where(v_p_vals > 0.1)[0]
    if len(idx_0) > 0:
        i = idx_0[0]
        points['0'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                       'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

    idx = np.argmax(p_b_vals)
    points['p_b_max'] = {'t': t_vals[idx], 'p_b': p_b_vals[idx], 'p_p': p_p_vals[idx],
                         'p_m': p_m_vals[idx], 'v_p': v_p_vals[idx], 'l_p': l_p_vals[idx]}

    idx = np.argmax(p_p_vals)
    points['p_p_max'] = {'t': t_vals[idx], 'p_b': p_b_vals[idx], 'p_p': p_p_vals[idx],
                         'p_m': p_m_vals[idx], 'v_p': v_p_vals[idx], 'l_p': l_p_vals[idx]}

    idx = np.argmax(p_m_vals)
    points['p_m_max'] = {'t': t_vals[idx], 'p_b': p_b_vals[idx], 'p_p': p_p_vals[idx],
                         'p_m': p_m_vals[idx], 'v_p': v_p_vals[idx], 'l_p': l_p_vals[idx]}

    idx_s = np.where(z_vals >= 1.0)[0]
    if len(idx_s) > 0:
        i = idx_s[0]
        points['s'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                       'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

    idx_e = np.where(psi_vals >= 0.99)[0]
    if len(idx_e) > 0:
        i = idx_e[0]
        points['e'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                       'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

    i = len(t_vals) - 1
    points['m'] = {'t': t_vals[i], 'p_b': p_b_vals[i], 'p_p': p_p_vals[i],
                   'p_m': p_m_vals[i], 'v_p': v_p_vals[i], 'l_p': l_p_vals[i]}

    return points, (t_vals, p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals)


def plot_lagrange_results(t_vals, p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals,
                          points, T_label="+15°C"):
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(t_vals, p_b_vals, '-', color='blue', linewidth=2, label='$p_b$', alpha=0.7)
    ax1.plot(t_vals, p_p_vals, '-', color='red', linewidth=2, label='$p_p$', alpha=0.7)
    ax1.plot(t_vals, p_m_vals, '-', color='limegreen', linewidth=2.5, label='$p_m$')

    ax1.set_xlabel('t, мс', fontsize=14)
    ax1.set_ylabel('Давление, МПа', fontsize=14)
    ax1.set_ylim(0, 500)
    ax1.yaxis.set_major_locator(MultipleLocator(75))
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.grid(True, alpha=0.7)

    ax2 = ax1.twinx()
    ax2.plot(t_vals, v_p_vals, '--', color='limegreen', linewidth=2.5, label='$V_p$')
    ax2.set_ylabel('$V_p$, м/с', fontsize=14)
    ax2.set_ylim(0, 1200)
    ax2.yaxis.set_major_locator(MultipleLocator(200))

    ax3 = ax1.twinx()
    ax3.spines['right'].set_position(('axes', 1.15))
    ax3.plot(t_vals, l_p_vals, ':', color='limegreen', linewidth=3, label='$l_p$')
    ax3.set_ylabel('$l_p$, м', fontsize=14)
    ax3.set_ylim(0, 6)
    ax3.yaxis.set_major_locator(MultipleLocator(2))

    for name, p in points.items():
        if name in ['0', 'p_m_max', 'm']:
            ax1.scatter([p['t']], [p['p_m']], color='black', s=50, zorder=5)
            ax2.scatter([p['t']], [p['v_p']], color='black', s=50, zorder=5)
            ax3.scatter([p['t']], [p['l_p']], color='black', s=50, zorder=5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    lines3, labels3 = ax3.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left')

    plt.tight_layout()
    return fig


def plot_lagrange_vs_path(p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals,
                          points, T_label="+15°C"):
    fig, ax1 = plt.subplots(figsize=(20, 10))

    ax1.plot(l_p_vals, p_b_vals, '-', color='blue', linewidth=2, label='$p_b$', alpha=0.7)
    ax1.plot(l_p_vals, p_p_vals, '-', color='red', linewidth=2, label='$p_p$', alpha=0.7)
    ax1.plot(l_p_vals, p_m_vals, '-', color='limegreen', linewidth=2.5, label='$p_m$')

    ax1.set_xlabel('$l_p$, м', fontsize=14)
    ax1.set_ylabel('Давление, МПа', fontsize=14)
    ax1.set_ylim(0, 500)
    ax1.set_xlim(0, max(l_p_vals) * 1.05)
    ax1.yaxis.set_major_locator(MultipleLocator(75))
    ax1.xaxis.set_major_locator(MultipleLocator(0.5))
    ax1.grid(True, alpha=0.7)

    for name, p in points.items():
        if name in ['0', 'p_m_max', 'm']:
            ax1.scatter([p['l_p']], [p['p_m']], color='black', s=60, zorder=5)

    ax2 = ax1.twinx()
    ax2.plot(l_p_vals, v_p_vals, '--', color='limegreen', linewidth=2.5, label='$V_p$')
    ax2.set_ylabel('$V_p$, м/с', fontsize=14)
    ax2.set_ylim(0, 1200)
    ax2.yaxis.set_major_locator(MultipleLocator(200))

    for name, p in points.items():
        if name in ['0', 'p_m_max', 'm']:
            ax2.scatter([p['l_p']], [p['v_p']], color='black', s=60, zorder=5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    plt.title(f'Зависимость от пройденного пути (T = {T_label})', fontsize=14)
    plt.tight_layout()

    return fig


def plot_unified_graphs(result_termo, result_lagrange, T_label="+15°C"):
    layers = result_lagrange['layers']
    t_lag = np.array([layer['t'] * 1000 for layer in layers])
    p_b_lag = np.array([layer['p'][0] / 1e6 for layer in layers])
    p_p_lag = np.array([layer['p'][-1] / 1e6 for layer in layers])
    p_m_lag = np.array([np.mean(layer['p']) / 1e6 for layer in layers])
    v_p_lag = np.array([layer['u'][-1] for layer in layers])
    l_p_lag = np.array([layer['x'][-1] for layer in layers])

    t_termo = result_termo['t'] * 1000
    p_m_termo = result_termo['p'] / 1e6
    v_p_termo = result_termo['v']
    l_p_termo = result_termo['x']

    points, _ = find_reference_points_lagrange(result_lagrange)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 10))

    ax1.plot(t_lag, p_b_lag, '-', color='blue', linewidth=2, label='$p_b$', alpha=0.7)
    ax1.plot(t_lag, p_p_lag, '-', color='red', linewidth=2, label='$p_p$', alpha=0.7)
    ax1.plot(t_lag, p_m_lag, '-', color='limegreen', linewidth=2.5, label='$p_m$ (газодинам.)')
    ax1.plot(t_termo, p_m_termo, '--', color='darkgreen', linewidth=2.5, label='$p_m$ (нульмерная)')

    ax1.set_xlabel('t, мс', fontsize=14)
    ax1.set_ylabel('Давление, МПа', fontsize=14)
    ax1.set_ylim(0, 500)
    ax1.yaxis.set_major_locator(MultipleLocator(75))
    ax1.xaxis.set_major_locator(MultipleLocator(2))
    ax1.grid(True, alpha=0.7)

    ax1_twin = ax1.twinx()
    ax1_twin.plot(t_lag, v_p_lag, '--', color='limegreen', linewidth=2.5, label='$V_p$ (газодинам.)')
    ax1_twin.plot(t_termo, v_p_termo, '--', color='darkgreen', linewidth=2.5, label='$V_p$ (нульмерная)')
    ax1_twin.set_ylabel('$V_p$, м/с', fontsize=14)
    ax1_twin.set_ylim(0, 1200)
    ax1_twin.yaxis.set_major_locator(MultipleLocator(200))

    ax1_twin2 = ax1.twinx()
    ax1_twin2.spines['right'].set_position(('axes', 1.12))
    ax1_twin2.plot(t_lag, l_p_lag, ':', color='limegreen', linewidth=3, label='$l_p$ (газодинам.)')
    ax1_twin2.plot(t_termo, l_p_termo, ':', color='darkgreen', linewidth=3, label='$l_p$ (нульмерная)')
    ax1_twin2.set_ylabel('$l_p$, м', fontsize=14)
    ax1_twin2.set_ylim(0, 6)
    ax1_twin2.yaxis.set_major_locator(MultipleLocator(2))

    for name, p in points.items():
        if name in ['0', 'p_m_max', 'm']:
            ax1.scatter([p['t']], [p['p_m']], color='black', s=50, zorder=5)
            ax1_twin.scatter([p['t']], [p['v_p']], color='black', s=50, zorder=5)
            ax1_twin2.scatter([p['t']], [p['l_p']], color='black', s=50, zorder=5)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_twin.get_legend_handles_labels()
    lines3, labels3 = ax1_twin2.get_legend_handles_labels()
    ax1.legend(lines1 + lines2 + lines3, labels1 + labels2 + labels3, loc='upper left', fontsize=10)

    ax1.set_title(f'Зависимость от времени (T = {T_label})', fontsize=14)

    ax2.plot(l_p_lag, p_b_lag, '-', color='blue', linewidth=2, label='$p_b$', alpha=0.7)
    ax2.plot(l_p_lag, p_p_lag, '-', color='red', linewidth=2, label='$p_p$', alpha=0.7)
    ax2.plot(l_p_lag, p_m_lag, '-', color='limegreen', linewidth=2.5, label='$p_m$ (газодинам.)')
    ax2.plot(l_p_termo, p_m_termo, '--', color='darkgreen', linewidth=2.5, label='$p_m$ (нульмерная)')

    ax2.set_xlabel('$l_p$, м', fontsize=14)
    ax2.set_ylabel('Давление, МПа', fontsize=14)
    ax2.set_ylim(0, 500)
    ax2.set_xlim(0, max(l_p_lag) * 1.05)
    ax2.yaxis.set_major_locator(MultipleLocator(75))
    ax2.xaxis.set_major_locator(MultipleLocator(0.5))
    ax2.grid(True, alpha=0.7)

    ax2_twin = ax2.twinx()
    ax2_twin.plot(l_p_lag, v_p_lag, '--', color='limegreen', linewidth=2.5, label='$V_p$ (газодинам.)')
    ax2_twin.plot(l_p_termo, v_p_termo, '--', color='darkgreen', linewidth=2.5, label='$V_p$ (нульмерная)')
    ax2_twin.set_ylabel('$V_p$, м/с', fontsize=14)
    ax2_twin.set_ylim(0, 1200)
    ax2_twin.yaxis.set_major_locator(MultipleLocator(200))

    for name, p in points.items():
        if name in ['0', 'p_m_max', 'm']:
            ax2.scatter([p['l_p']], [p['p_m']], color='black', s=60, zorder=5)
            ax2_twin.scatter([p['l_p']], [p['v_p']], color='black', s=60, zorder=5)

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2_twin.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=11)

    ax2.set_title(f'Зависимость от пройденного пути (T = {T_label})', fontsize=14)

    plt.tight_layout()
    return fig


res_min = RungeKutta4(sys_min, init, stop, report_min, dt, 0, max_steps)
res_norm = RungeKutta4(sys_norm, init, stop, report_norm, dt, 0, max_steps)
res_max = RungeKutta4(sys_max, init, stop, report_max, dt, 0, max_steps)

df_1 = pd.DataFrame(res_min, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
pd.set_option('display.precision', 5)
t_1 = df_1.t
p_m_T_01 = df_1.p_m
V_p_T_01 = df_1.V_p
l_p_T_01 = df_1.l_p

df_1_0 = df_1.query('l_p>=0' and 'V_p>0')
index_0_1 = df_1_0[:1].index[0]
t_01 = df_1.t.iloc[index_0_1] * 1e3
t_0_1 = f'{t_01:.2f}'
p_m_0_1 = f'{df_1.p_m.iloc[index_0_1]:.4f}'
V_p_0_1 = f'{df_1.V_p.iloc[index_0_1]:.4f}'
l_p_0_1 = f'{df_1.l_p.iloc[index_0_1]:.4f}'

max_index_1 = df_1['p_m'].idxmax()
t_max1 = df_1.t.iloc[max_index_1] * 1e3
t_max_1 = f'{t_max1:.2f}'
p_m_max_1 = f'{df_1.p_m.iloc[max_index_1]:.4f}'
V_p_max_1 = f'{df_1.V_p.iloc[max_index_1]:.4f}'
l_p_max_1 = f'{df_1.l_p.iloc[max_index_1]:.4f}'

df_e_1 = df_1[df_1['z'] <= z_e]
e_index_1 = df_e_1.index[-1]

t_e1 = df_1.t.iloc[e_index_1] * 1e3
t_e_1 = f'{t_e1:.2f}'
p_m_e_1 = f'{df_1.p_m.iloc[e_index_1]:.4f}'
V_p_e_1 = f'{df_1.V_p.iloc[e_index_1]:.4f}'
l_p_e_1 = f'{df_1.l_p.iloc[e_index_1]:.4f}'

index_m_1 = df_1[df_1['x_p'] >= (l_m + l_0)].index[0]
t_m1 = df_1.t.iloc[index_m_1] * 1e3
t_m_1 = f'{t_m1:.2f}'
p_m_m_1 = f'{df_1.p_m.iloc[index_m_1]:.4f}'
V_p_m_1 = f'{df_1.V_p.iloc[index_m_1]:.4f}'
l_p_m_1 = f'{df_1.l_p.iloc[index_m_1]:.4f}'

res_T_01 = {
    'Точка': ['0', 'p_max', 'e', 'm'],
    't,мс': [t_0_1, t_max_1, '-', t_m_1],
    'p_m,МПа': [p_m_0_1, p_m_max_1, '-', p_m_m_1],
    'V_p,м/с': [V_p_0_1, V_p_max_1, '-', V_p_m_1],
    'l_p,м': [l_p_0_1, l_p_max_1, '-', l_p_m_1],
}
d_res_T_01 = pd.DataFrame(res_T_01)
print('___ Результаты при температура T = -50 градусов ___')
print(d_res_T_01)

res_T_0_1 = {
    't': [t_0_1, t_max_1, t_e_1, t_m_1],
    'p_m': [p_m_0_1, p_m_max_1, p_m_e_1, p_m_m_1],
    'v_p': [V_p_0_1, V_p_max_1, V_p_e_1, V_p_m_1],
    'l_p': [l_p_0_1, l_p_max_1, l_p_e_1, l_p_m_1],
}

index_labels = ['0', 'p_max', 'e', 'm']
d_res_T_0_1 = pd.DataFrame(res_T_0_1, index=index_labels)

df_2 = pd.DataFrame(res_norm, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
pd.set_option('display.precision', 5)
t_2 = df_2.t
p_m_T_02 = df_2.p_m
V_p_T_02 = df_2.V_p
l_p_T_02 = df_2.l_p

df_2_0 = df_2.query('l_p>=0' and 'V_p>0')
index_0_2 = df_2_0[:1].index[0]
t_02 = df_2.t.iloc[index_0_2] * 1e3
t_0_2 = f'{t_02:.2f}'
p_m_0_2 = f'{df_2.p_m.iloc[index_0_2]:.4f}'
V_p_0_2 = f'{df_2.V_p.iloc[index_0_2]:.4f}'
l_p_0_2 = f'{df_2.l_p.iloc[index_0_2]:.4f}'

max_index_2 = df_2['p_m'].idxmax()
t_max2 = df_2.t.iloc[max_index_2] * 1e3
t_max_2 = f'{t_max2:.2f}'
p_m_max_2 = f'{df_2.p_m.iloc[max_index_2]:.4f}'
V_p_max_2 = f'{df_2.V_p.iloc[max_index_2]:.4f}'
l_p_max_2 = f'{df_2.l_p.iloc[max_index_2]:.4f}'

df_e_2 = df_2[df_2['z'] >= z_e]
e_index_2 = df_e_2.index[0]
t_e2 = df_2.t.iloc[e_index_2] * 1e3
t_e_2 = f'{t_e2:.2f}'
p_m_e_2 = f'{df_2.p_m.iloc[e_index_2]:.4f}'
V_p_e_2 = f'{df_2.V_p.iloc[e_index_2]:.4f}'
l_p_e_2 = f'{df_2.l_p.iloc[e_index_2]:.4f}'

index_m_2 = df_2[df_2['x_p'] >= (l_m + l_0)].index[0]
t_m2 = df_2.t.iloc[index_m_2] * 1e3
t_m_2 = f'{t_m2:.2f}'
p_m_m_2 = f'{df_2.p_m.iloc[index_m_2]:.4f}'
V_p_m_2 = f'{df_2.V_p.iloc[index_m_2]:.4f}'
l_p_m_2 = f'{df_2.l_p.iloc[index_m_2]:.4f}'

res_T_02 = {
    'Точка': ['0', 'p_max', 'e', 'm'],
    't,мс': [t_0_2, t_max_2, t_e_2, t_m_2],
    'p_m,МПа': [p_m_0_2, p_m_max_2, p_m_e_2, p_m_m_2],
    'V_p,м/с': [V_p_0_2, V_p_max_2, V_p_e_2, V_p_m_2],
    'l_p,м': [l_p_0_2, l_p_max_2, l_p_e_2, l_p_m_2],
}
d_res_T_02 = pd.DataFrame(res_T_02)
print('___ Результаты при температура T = +15 градусов ___')
print(d_res_T_02)

res_T_0_2 = {
    't': [t_0_2, t_max_2, t_e_2, t_m_2],
    'p_m': [p_m_0_2, p_m_max_2, p_m_e_2, p_m_m_2],
    'v_p': [V_p_0_2, V_p_max_2, V_p_e_2, V_p_m_2],
    'l_p': [l_p_0_2, l_p_max_2, l_p_e_2, l_p_m_2],
}

index_labels = ['0', 'p_max', 'e', 'm']
d_res_T_0_2 = pd.DataFrame(res_T_0_2, index=index_labels)

df_3 = pd.DataFrame(res_max, columns=['t', 'x_p', 'V_p', 'A_pa', 'Q_w', 'eta_t', 'z', 'psi', 'p_m', 'l_p'])
pd.set_option('display.precision', 5)
new_df_3 = df_3.rename(columns={'t': 't,c', 'x_p': 'x_p,м', 'V_p': 'V_p, м/с', 'p_m': 'p_m, МПа', 'l_p': 'l_p,м'})
df_3_last = new_df_3.tail(1)
t_3 = df_3.t
p_m_T_03 = df_3.p_m
V_p_T_03 = df_3.V_p
l_p_T_03 = df_3.l_p

df_0_3 = df_3.query('l_p>=0' and 'V_p>0')
index_0_3 = df_0_3[:1].index[0]
t_03 = df_3.t.iloc[index_0_3] * 1e3
t_0_3 = f'{t_03:.2f}'
p_m_0_3 = f'{df_3.p_m.iloc[index_0_3]:.4f}'
V_p_0_3 = f'{df_3.V_p.iloc[index_0_3]:.4f}'
l_p_0_3 = f'{df_3.l_p.iloc[index_0_3]:.4f}'

max_index_3 = df_3['p_m'].idxmax()
t_max3 = df_3.t.iloc[max_index_3] * 1e3
t_max_3 = f'{t_max3:.2f}'
p_m_max_3 = f'{df_3.p_m.iloc[max_index_3]:.4f}'
V_p_max_3 = f'{df_3.V_p.iloc[max_index_3]:.4f}'
l_p_max_3 = f'{df_3.l_p.iloc[max_index_3]:.4f}'

df_e_3 = df_3[df_3['z'] >= z_e]
e_index_3 = df_e_3.index[0]
t_e3 = df_3.t.iloc[e_index_3] * 1e3
t_e_3 = f'{t_e3:.2f}'
p_m_e_3 = f'{df_3.p_m.iloc[e_index_3]:.4f}'
V_p_e_3 = f'{df_3.V_p.iloc[e_index_3]:.4f}'
l_p_e_3 = f'{df_3.l_p.iloc[e_index_3]:.4f}'

index_m_3 = df_3[df_3['x_p'] >= (l_m + l_0)].index[0]
t_m3 = df_3.t.iloc[index_m_3] * 1e3
t_m_3 = f'{t_m3:.2f}'
p_m_m_3 = f'{df_3.p_m.iloc[index_m_3]:.4f}'
V_p_m_3 = f'{df_3.V_p.iloc[index_m_3]:.4f}'
l_p_m_3 = f'{df_3.l_p.iloc[index_m_3]:.4f}'

res_T_03 = {
    'Точка': ['0', 'p_max', 'e', 'm'],
    't,мс': [t_0_3, t_max_3, t_e_3, t_m_3],
    'p_m,МПа': [p_m_0_3, p_m_max_3, p_m_e_3, p_m_m_3],
    'V_p,м/с': [V_p_0_3, V_p_max_3, V_p_e_3, V_p_m_3],
    'l_p,м': [l_p_0_3, l_p_max_3, l_p_e_3, l_p_m_3],
}

d_res_T_03 = pd.DataFrame(res_T_03)
print('___ Результаты при температура T = +50 градусов ___')
print(d_res_T_03)

res_T_0_3 = {
    't': [t_0_3, t_max_3, t_e_3, t_m_3],
    'p_m': [p_m_0_3, p_m_max_3, p_m_e_3, p_m_m_3],
    'v_p': [V_p_0_3, V_p_max_3, V_p_e_3, V_p_m_3],
    'l_p': [l_p_0_3, l_p_max_3, l_p_e_3, l_p_m_3],
}

index_labels = ['0', 'p_max', 'e', 'm']
d_res_T_0_3 = pd.DataFrame(res_T_0_3, index=index_labels)

set_decimal = '.'
d_res_T_0_1.to_csv('СМ6-69_Пальников_СД_output.csv', sep='\t', mode='w', float_format='%.4f', index=True,
                   decimal=set_decimal)
d_res_T_0_2.to_csv('СМ6-69_Пальников_СД_output.csv', sep='\t', mode='a', float_format='%.4f', index=True,
                   decimal=set_decimal)
d_res_T_0_3.to_csv('СМ6-69_Пальников_СД_output.csv', sep='\t', mode='a', float_format='%.4f', index=True,
                   decimal=set_decimal)

print('Расчетное максимальное давление = ', f'{df_2.p_m.iloc[max_index_2]:.4f}', 'МПа')
print('Эталонное максимальное давление = ', p_max, 'МПа')
epsilon_p = (df_2.p_m.iloc[max_index_2] - p_max) / p_max * 100
print('Погрешность по давлению = ', f'{epsilon_p:.4f}', '%')
print('Расчетное дульная скорость = ', f'{df_2.V_p.iloc[index_m_2]:.4f}', 'м/с')
print('Эталонное дульная скорость = ', v_pm, 'м/с')
epsilon_v = (df_2.V_p.iloc[index_m_2] - v_pm) / v_pm * 100
print('Погрешность по скорости = ', f'{epsilon_v:.4f}', '%')

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot()
plt.grid(linewidth=2)
line_p_m_1 = ax.plot(t_1 * 1e3, p_m_T_01, '-', label='$p_m$ ($T= -50℃$)', color="blue", linewidth=3)
ax.set_xlabel('t,мс', fontsize=30)
ax.xaxis.set_major_locator(MultipleLocator(2.5))
ax.yaxis.set_major_locator(MultipleLocator(75))
ax.set_ylabel('$p_{m}$,МПа', color='black', fontsize=30)
ax.tick_params('y', colors='black', labelsize=25)
ax.tick_params('x', colors='black', labelsize=25)
ax.xaxis.set_major_locator(MultipleLocator(1))

ax.scatter(
    [df_1['t'].iloc[index_0_1] * 1e3, df_1['t'].iloc[max_index_1] * 1e3,
     df_1['t'].iloc[index_m_1] * 1e3],
    [df_1['p_m'].iloc[index_0_1], df_1['p_m'].iloc[max_index_1],
     df_1['p_m'].iloc[index_m_1]], marker="o", color='purple', s=50)
ax.set_ylim(0, 500)

line_p_m_2 = ax.plot(t_2 * 1e3, p_m_T_02, '-', label='$p_m$ ($T=+15℃$)', color="limegreen", linewidth=3)
ax.scatter(
    [df_2['t'].iloc[index_0_2] * 1e3, df_2['t'].iloc[max_index_2] * 1e3,
     df_2['t'].iloc[index_m_2] * 1e3],
    [df_2['p_m'].iloc[index_0_2], df_2['p_m'].iloc[max_index_2],
     df_2['p_m'].iloc[index_m_2]], marker="o", color='red', s=50)

line_p_m_3 = ax.plot(t_3 * 1e3, p_m_T_03, '-', alpha=1, label='$p_m$ ($T= +50℃$)', color="crimson", linewidth=3)
ax.scatter(
    [df_3['t'].iloc[index_0_3] * 1e3, df_3['t'].iloc[max_index_3] * 1e3,
     df_3['t'].iloc[e_index_3] * 1e3, df_3['t'].iloc[index_m_3] * 1e3],
    [df_3['p_m'].iloc[index_0_3], df_3['p_m'].iloc[max_index_3],
     df_3['p_m'].iloc[e_index_3], df_3['p_m'].iloc[index_m_3]], marker="o", color='black', s=50)

ax2 = ax.twinx()

line_V_p_1 = ax2.plot(t_1 * 1e3, V_p_T_01, '--', label='$V_p$ (T= -50℃)', color="blue", linewidth=3)
ax2.set_ylabel('$V_p$,м/с', color='black', fontsize=30)
ax2.scatter(
    [df_1['t'].iloc[index_0_1] * 1e3, df_1['t'].iloc[max_index_1] * 1e3,
     df_1['t'].iloc[index_m_1] * 1e3],
    [df_1['V_p'].iloc[index_0_1], df_1['V_p'].iloc[max_index_1],
     df_1['V_p'].iloc[index_m_1]], marker="o", color='purple', s=50)
ax2.set_ylim(0, 1200)
ax2.yaxis.set_major_locator(MultipleLocator(200))

line_V_p_2 = ax2.plot(t_2 * 1e3, V_p_T_02, '--', label='$V_p$ (T= +15℃)', color="limegreen", linewidth=3)
ax2.scatter(
    [df_2['t'].iloc[index_0_2] * 1e3, df_2['t'].iloc[max_index_2] * 1e3,
     df_2['t'].iloc[index_m_2] * 1e3],
    [df_2['V_p'].iloc[index_0_2], df_2['V_p'].iloc[max_index_2],
     df_2['V_p'].iloc[index_m_2]], marker="o", color='red', s=50)

line_V_p_3 = ax2.plot(t_3 * 1e3, V_p_T_03, '--', alpha=1, label='$V_p$ (T= +50℃)', color="crimson", linewidth=3)
ax2.scatter(
    [df_3['t'].iloc[index_0_3] * 1e3, df_3['t'].iloc[max_index_3] * 1e3,
     df_3['t'].iloc[e_index_3] * 1e3, df_3['t'].iloc[index_m_3] * 1e3],
    [df_3['V_p'].iloc[index_0_3], df_3['V_p'].iloc[max_index_3],
     df_3['V_p'].iloc[e_index_3], df_3['V_p'].iloc[index_m_3]], marker="o", color='black', s=50)
ax2.tick_params('y', colors='black', labelsize=25)

ax3 = ax.twinx()

ax3.spines['right'].set_position(('axes', 1.12))
ax3.set_ylabel('$l_p$,м', color='black', fontsize=30)
line_l_p_1 = ax3.plot(t_1 * 1e3, l_p_T_01, ':', label='$l_p$ (T= -50℃)', color="blue", linewidth=4)

ax3.scatter(
    [df_1['t'].iloc[index_0_1] * 1e3, df_1['t'].iloc[max_index_1] * 1e3,
     df_1['t'].iloc[index_m_1] * 1e3],
    [df_1['l_p'].iloc[index_0_1], df_1['l_p'].iloc[max_index_1],
     df_1['l_p'].iloc[index_m_1]], marker="o", color='purple', s=50)
ax3.set_ylim(0, 6)
ax3.yaxis.set_major_locator(MultipleLocator(2))
line_l_p_2 = ax3.plot(t_2 * 1e3, l_p_T_02, ':', label='$l_p$ (T= +15℃)', color="limegreen", linewidth=4)
ax3.scatter(
    [df_2['t'].iloc[index_0_2] * 1e3, df_2['t'].iloc[max_index_2] * 1e3,
     df_2['t'].iloc[index_m_2] * 1e3],
    [df_2['l_p'].iloc[index_0_2], df_2['l_p'].iloc[max_index_2],
     df_2['l_p'].iloc[index_m_2]], marker="o", color='red', s=50)

line_l_p_3 = ax3.plot(t_3 * 1e3, l_p_T_03, ':', label='$l_p$ (T= +50℃)', alpha=1, color="crimson", linewidth=4)
ax3.scatter(
    [df_3['t'].iloc[index_0_3] * 1e3, df_3['t'].iloc[max_index_3] * 1e3,
     df_3['t'].iloc[e_index_3] * 1e3, df_3['t'].iloc[index_m_3] * 1e3],
    [df_3['l_p'].iloc[index_0_3], df_3['l_p'].iloc[max_index_3],
     df_3['l_p'].iloc[e_index_3], df_3['l_p'].iloc[index_m_3]], marker="o", color='black', s=50)
ax3.tick_params('y', colors='black', labelsize=25)

fig.legend(bbox_to_anchor=(0.3, 0.95), fontsize=20)

plt.tight_layout()
plt.savefig('СМ6-69_Пальников_СД_график_1.png', dpi=300)

fig.show()

fig = plt.figure(figsize=(20, 10))
ax = fig.add_subplot()
plt.grid(linewidth=2)

line_p_m_1 = ax.plot(df_1['l_p'], p_m_T_01, '-', label='$p_m$ ($T= -50℃$)', color="blue", linewidth=3)
ax.set_xlabel('$l_p$,м', fontsize=20)
ax.tick_params('x', colors='black', labelsize=20)

ax.xaxis.set_major_locator(MultipleLocator(0.5))
ax.yaxis.set_major_locator(MultipleLocator(100))
ax.set_ylabel('$p_m$,МПа', color='black', fontsize=20)
ax.tick_params('y', colors='black', labelsize=20)
ax.scatter(
    [df_1['l_p'].iloc[index_0_1], df_1['l_p'].iloc[max_index_1],
     df_1['l_p'].iloc[index_m_1]],
    [df_1['p_m'].iloc[index_0_1], df_1['p_m'].iloc[max_index_1],
     df_1['p_m'].iloc[index_m_1]], marker="o", color='purple', s=50)
ax.set_ylim(0, 500)
ax.set_xlim(0, 5)

line_p_m_2 = ax.plot(df_2['l_p'], p_m_T_02, '-', label='$p_m$ ($T=+15℃$)', color="limegreen", linewidth=3)
ax.scatter(
    [df_2['l_p'].iloc[index_0_2], df_2['l_p'].iloc[max_index_2],
     df_2['l_p'].iloc[index_m_2]],
    [df_2['p_m'].iloc[index_0_2], df_2['p_m'].iloc[max_index_2],
     df_2['p_m'].iloc[index_m_2]], marker="o", color='red', s=50)

line_p_m_3 = ax.plot(df_3['l_p'], p_m_T_03, '-', label='$p_m$ ($T= +50℃$)', alpha=1, color="crimson", linewidth=3)
ax.scatter(
    [df_3['l_p'].iloc[index_0_3], df_3['l_p'].iloc[max_index_3],
     df_3['l_p'].iloc[e_index_3], df_3['l_p'].iloc[index_m_3]],
    [df_3['p_m'].iloc[index_0_3], df_3['p_m'].iloc[max_index_3],
     df_3['p_m'].iloc[e_index_3], df_3['p_m'].iloc[index_m_3]], marker="o", color='black', s=50)
yticks = ax.xaxis.get_major_ticks()
yticks[1].label1.set_visible(False)

ax2 = ax.twinx()
line_V_p_1 = ax2.plot(df_1['l_p'], V_p_T_01, '--', label='$V_p$ (T= -50℃)', color="blue", linewidth=3)
ax2.tick_params('y', colors='black', labelsize=20)
ax2.set_ylabel('$V_p$,м/с', color='black', fontsize=20)
ax2.scatter(
    [df_1['l_p'].iloc[index_0_1], df_1['l_p'].iloc[max_index_1],
     df_1['l_p'].iloc[index_m_1]],
    [df_1['V_p'].iloc[index_0_1], df_1['V_p'].iloc[max_index_1],
     df_1['V_p'].iloc[index_m_1]], marker="o", color='purple', s=50)
ax2.set_ylim(0, 1200)
ax2.yaxis.set_major_locator(MultipleLocator(200))

ax2.plot(df_2['l_p'], V_p_T_02, '--', label='$V_p$ (T= +15℃)', color="limegreen", linewidth=3)
ax2.scatter(
    [df_2['l_p'].iloc[index_0_2], df_2['l_p'].iloc[max_index_2],
     df_2['l_p'].iloc[index_m_2]],
    [df_2['V_p'].iloc[index_0_2], df_2['V_p'].iloc[max_index_2],
     df_2['V_p'].iloc[index_m_2]], marker="o", color='red', s=50)

ax2.plot(df_3['l_p'], V_p_T_03, '--', label='$V_p$ (T= +50℃)', alpha=1, color="crimson", linewidth=3)
ax2.scatter(
    [df_3['l_p'].iloc[index_0_3], df_3['l_p'].iloc[max_index_3],
     df_3['l_p'].iloc[e_index_3], df_3['l_p'].iloc[index_m_3]],
    [df_3['V_p'].iloc[index_0_3], df_3['V_p'].iloc[max_index_3],
     df_3['V_p'].iloc[e_index_3], df_3['V_p'].iloc[index_m_3]], marker="o", color='black', s=50)

fig.legend(ncol=2, bbox_to_anchor=(0.65, 0.3), fontsize=20)
plt.tight_layout()
plt.savefig('СМ6-69_Пальников_СД_график_2.png', dpi=300)
fig.show()

opts = create_opts_for_lagrange('СМ6-69_Пальников_СД_data.csv', '180/57', T_0=288.15)
result = ozvb_lagrange(opts)

points, data = find_reference_points_lagrange(result)
t_vals, p_b_vals, p_p_vals, p_m_vals, v_p_vals, l_p_vals = data

df = pd.DataFrame([
    {'Точка': f'«{k}»', 't, мс': f"{v['t']:.2f}",
     'p_b, МПа': f"{v['p_b']:.2f}", 'p_p, МПа': f"{v['p_p']:.2f}",
     'p_m, МПа': f"{v['p_m']:.2f}", 'v_p, м/с': f"{v['v_p']:.2f}",
     'l_p, м': f"{v['l_p']:.4f}"}
    for k, v in points.items()
])
print(df.to_string(index=False))

fig = plot_lagrange_results(t_vals, p_b_vals, p_p_vals, p_m_vals,
                            v_p_vals, l_p_vals, points)
fig.savefig('СМ6-69_Пальников_СД_график_3.png', dpi=300)

fig_path = plot_lagrange_vs_path(p_b_vals, p_p_vals, p_m_vals, v_p_vals,
                                 l_p_vals, points)
fig_path.savefig('СМ6-69_Пальников_СД_график_4.png', dpi=300)

result_termo = {'t': df_2['t'].values, 'p': df_2['p_m'].values * 1e6,
                'v': df_2['V_p'].values, 'x': df_2['l_p'].values}

fig_unified = plot_unified_graphs(result_termo, result, T_label="+15°C")
fig_unified.savefig('СМ6-69_Пальников_СД_график_5.png', dpi=300)