import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict
import pandas as pd

from ODE_solvers import RungeKutta4


@dataclass
class GunParameters:
    d: float
    q: float
    v_pm: float
    p_max_target: float

    omega: float
    W_0: float
    l_m: float

    I_e: float
    f: float
    k: float
    delta_powder: float

    kappa_1: float = 0.72
    lambda_1: float = 0.18
    kappa_2: float = 0.54
    lambda_2: float = -0.90
    z_e: float = 1.532

    k_f: float = 0.0003
    k_I: float = 0.0016
    K: float = 1.05

    p_0: float = 30e6
    p_ign: float = 5e6
    n_S: float = 1.04

    f_ign: float = 260000
    b_ign: float = 0.0006
    k_ign: float = 1.25
    T_ign: float = 2427

    Pr: float = 0.74
    lambda_g: float = 0.2218
    mu_0: float = 1.75e-5
    T_0s: float = 273
    T_cs: float = 628

    c_b: float = 500
    rho_b: float = 7900
    lambda_b: float = 40

    p_a0: float = 101325
    k_air: float = 1.4
    c_0a: float = 340

    @property
    def S(self) -> float:
        return np.pi * (self.d ** 2) / 4 * self.n_S

    @property
    def l_0(self) -> float:
        return self.W_0 / self.S

    @property
    def Delta(self) -> float:
        return self.omega / self.W_0

    def get_phi(self) -> float:
        b = 0.001
        zeta = (1 / self.Delta - 1 / self.delta_powder) / (self.f_ign / self.p_ign + b)
        phi = self.K + (1 / 3) * (self.omega / self.q) * (1 + zeta)
        return phi


class BallisticsCalculator:
    def __init__(self, gun: GunParameters):
        self.gun = gun
        self.T_ref = 288.15

    def get_powder_params(self, T0: float) -> Dict:
        T0_K = T0 + 273.15
        f_T = self.gun.f * (1 + self.gun.k_f * (T0_K - self.T_ref))
        I_e_T = self.gun.I_e * (1 - self.gun.k_I * (T0_K - self.T_ref))
        return {'f': f_T, 'I_e': I_e_T, 'T_0': T0_K}

    def calc_psi(self, z: float) -> float:
        if z <= 1.0:
            psi = self.gun.kappa_1 * z * (1 + self.gun.lambda_1 * z)
        else:
            psi_s = self.gun.kappa_1 * 1.0 * (1 + self.gun.lambda_1 * 1.0)
            dz = z - 1.0
            psi = psi_s + self.gun.kappa_2 * dz * (1 + self.gun.lambda_2 * dz)
        return max(0.0, min(1.0, psi))

    def ozvb_system(self, t: float, y: np.ndarray, params: Dict) -> np.ndarray:
        x_p, v_p, A_pa, Q_w, eta_T, z = y

        f = params['f']
        I_e = params['I_e']
        T_0 = params['T_0']
        phi = self.gun.get_phi()

        psi = self.calc_psi(z)

        Delta = self.gun.Delta
        delta = self.gun.delta_powder
        b = 0.001
        zeta = (1 / Delta - 1 / delta) / (self.gun.f_ign / self.gun.p_ign + b)

        # Уравнение Резаля
        free_volume_solid = self.gun.omega / delta * (1 - psi)
        covolume_gas = self.gun.omega * (psi + zeta) * b
        W_free = self.gun.S * x_p - free_volume_solid - covolume_gas

        if W_free <= 1e-12:
            W_free = 1e-12

        energy_gas = f * self.gun.omega * psi
        energy_igniter = self.gun.f_ign * self.gun.omega * zeta
        energy_kinetic = phi * self.gun.q * v_p ** 2 / 2

        numerator = energy_gas + energy_igniter - (self.gun.k - 1) * (energy_kinetic + A_pa + Q_w)
        p_m = numerator / W_free
        p_m = max(p_m, 0.0)

        # Давление воздуха
        mach = v_p / self.gun.c_0a
        if mach < 1.0:
            p_a = self.gun.p_a0 * (1 + self.gun.k_air * (self.gun.k_air + 1) / 4 * mach ** 2)
        else:
            p_a = self.gun.p_a0 * (2 * self.gun.k_air * mach ** 2 - (self.gun.k_air - 1)) / (self.gun.k_air + 1)

        pressure_diff = p_m - p_a

        # Уравнение движения
        if v_p > 1e-9:
            dv_p_dt = pressure_diff * self.gun.S / (phi * self.gun.q)
        else:
            if pressure_diff > self.gun.p_0:
                dv_p_dt = pressure_diff * self.gun.S / (phi * self.gun.q)
            else:
                dv_p_dt = 0.0

        # Теплопередача
        l_p = max(0.0, x_p - self.gun.l_0)
        S_w0 = 4 * self.gun.W_0 / self.gun.d
        S_w = S_w0 + np.pi * self.gun.d * l_p

        R_g = f / 2850
        mass_gas = self.gun.omega * (psi + zeta)
        rho_g = mass_gas / W_free if W_free > 0 else 0.0
        T_gas = p_m / (R_g * rho_g) if rho_g > 0 else T_0
        T_gas = max(T_0, min(T_gas, 5000))

        mu_g = self.gun.mu_0 * (self.gun.T_0s + self.gun.T_cs) / (T_gas + self.gun.T_cs)
        mu_g *= (T_gas / self.gun.T_0s) ** 1.5

        if x_p > 1e-6 and mu_g > 0:
            Re = self.gun.omega * (1 + zeta) * abs(v_p) * self.gun.d / (2 * self.gun.S * x_p * mu_g)
            Re = max(Re, 1.0)
            Nu = 0.023 * (Re ** 0.8) * (self.gun.Pr ** 0.4)
            T_wall = T_0 + np.sqrt(max(0.0, eta_T))
            q_w = Nu * self.gun.lambda_g / self.gun.d * (T_gas - T_wall)
        else:
            q_w = 0.0

        dQ_w_dt = S_w * q_w

        if self.gun.c_b * self.gun.rho_b * self.gun.lambda_b > 0:
            deta_T_dt = 2 * q_w ** 2 / (self.gun.c_b * self.gun.rho_b * self.gun.lambda_b)
        else:
            deta_T_dt = 0.0

        if x_p > 1e-6:
            deta_T_dt -= 2 * v_p / x_p * eta_T

        if z < self.gun.z_e and I_e > 0:
            dz_dt = p_m / I_e
        else:
            dz_dt = 0.0

        dx_p_dt = v_p
        dA_pa_dt = p_a * self.gun.S * v_p

        return np.array([dx_p_dt, dv_p_dt, dA_pa_dt, dQ_w_dt, deta_T_dt, dz_dt])

    def calculate_termo_manual(self, T0: float, dt: float = 1e-5, verbose: bool = True) -> Dict:
        """Шаг 10 мкс как в методичке"""
        params = self.get_powder_params(T0)

        l_0 = self.gun.l_0
        y0 = np.array([l_0, 0.0, 0.0, 0.0, 0.0, 0.0])

        ode_system = lambda t, y: self.ozvb_system(t, y, params)
        stop_condition = lambda t, y: (self.gun.l_0 + self.gun.l_m) - y[0]

        def report(t, y):
            x_p, v_p, A_pa, Q_w, eta_T, z = y
            psi = self.calc_psi(z)

            Delta = self.gun.Delta
            delta = self.gun.delta_powder
            b = 0.001
            zeta = (1 / Delta - 1 / delta) / (self.gun.f_ign / self.gun.p_ign + b)
            phi = self.gun.get_phi()
            f = params['f']

            free_volume_solid = self.gun.omega / delta * (1 - psi)
            covolume_gas = self.gun.omega * (psi + zeta) * b
            W_free = self.gun.S * x_p - free_volume_solid - covolume_gas
            if W_free <= 0: W_free = 1e-10

            numerator = f * self.gun.omega * psi + self.gun.f_ign * self.gun.omega * zeta
            numerator -= (self.gun.k - 1) * (phi * self.gun.q * v_p ** 2 / 2 + A_pa + Q_w)
            p_m = numerator / W_free
            p_m = max(0.0, p_m)

            mach = v_p / self.gun.c_0a
            if mach < 1.0:
                p_a = self.gun.p_a0 * (1 + self.gun.k_air * (self.gun.k_air + 1) / 4 * mach ** 2)
            else:
                p_a = self.gun.p_a0 * (2 * self.gun.k_air * mach ** 2 - (self.gun.k_air - 1)) / (self.gun.k_air + 1)

            return np.array([p_m, p_a, psi, z])

        result_array = RungeKutta4(ode_system, y0, stop_condition, report, dt, x_0=0.0)

        return {
            't': result_array[:, 0],
            'x_p': result_array[:, 1],
            'v_p': result_array[:, 2],
            'p_m': result_array[:, 7],
            'p_a': result_array[:, 8],
            'psi_1': result_array[:, 9],
            'z_1': result_array[:, 10],
        }

    def find_reference_points(self, result: Dict) -> pd.DataFrame:
        """Точное определение с интерполяцией"""
        t = result['t']
        p_m = result['p_m']
        v_p = result['v_p']
        x_p = result['x_p']
        z = result['z_1']
        psi = result['psi_1']
        p_a = result['p_a']

        l_0 = self.gun.l_0
        l_p = x_p - l_0

        points = []

        # Точка «0»: v_p = 0.1252 м/с (интерполяция)
        target_v0 = 0.1252
        for i in range(1, len(v_p)):
            if v_p[i - 1] < target_v0 <= v_p[i]:
                alpha = (target_v0 - v_p[i - 1]) / (v_p[i] - v_p[i - 1])
                t_0 = t[i - 1] + alpha * (t[i] - t[i - 1])
                p_0 = p_m[i - 1] + alpha * (p_m[i] - p_m[i - 1])
                l_0_val = l_p[i - 1] + alpha * (l_p[i] - l_p[i - 1])

                points.append({
                    'Точка': '«0»',
                    't, мс': t_0 * 1000,
                    'p_m, МПа': p_0 / 1e6,
                    'v_p, м/с': target_v0,
                    'l_p, м': l_0_val
                })
                break

        # Точка «max»
        idx_max = np.argmax(p_m)
        points.append({
            'Точка': '«max»',
            't, мс': t[idx_max] * 1000,
            'p_m, МПа': p_m[idx_max] / 1e6,
            'v_p, м/с': v_p[idx_max],
            'l_p, м': l_p[idx_max]
        })

        # Точка «s»: z = 1
        for i in range(1, len(z)):
            if z[i - 1] < 1.0 <= z[i]:
                alpha = (1.0 - z[i - 1]) / (z[i] - z[i - 1])
                t_s = t[i - 1] + alpha * (t[i] - t[i - 1])
                p_s = p_m[i - 1] + alpha * (p_m[i] - p_m[i - 1])
                v_s = v_p[i - 1] + alpha * (v_p[i] - v_p[i - 1])
                l_s = l_p[i - 1] + alpha * (l_p[i] - l_p[i - 1])

                points.append({
                    'Точка': '«s»',
                    't, мс': t_s * 1000,
                    'p_m, МПа': p_s / 1e6,
                    'v_p, м/с': v_s,
                    'l_p, м': l_s
                })
                break

        # Точка «e»: psi = 0.99
        for i in range(1, len(psi)):
            if psi[i - 1] < 0.99 <= psi[i]:
                alpha = (0.99 - psi[i - 1]) / (psi[i] - psi[i - 1])
                t_e = t[i - 1] + alpha * (t[i] - t[i - 1])
                p_e = p_m[i - 1] + alpha * (p_m[i] - p_m[i - 1])
                v_e = v_p[i - 1] + alpha * (v_p[i] - v_p[i - 1])
                l_e = l_p[i - 1] + alpha * (l_p[i] - l_p[i - 1])

                points.append({
                    'Точка': '«e»',
                    't, мс': t_e * 1000,
                    'p_m, МПа': p_e / 1e6,
                    'v_p, м/с': v_e,
                    'l_p, м': l_e
                })
                break

        # Точка «m»
        points.append({
            'Точка': '«m»',
            't, мс': t[-1] * 1000,
            'p_m, МПа': p_m[-1] / 1e6,
            'v_p, м/с': v_p[-1],
            'l_p, м': l_p[-1]
        })

        return pd.DataFrame(points)


def run_test_verification():
    print("=" * 80)
    print("ПРОВЕРКА НА ТЕСТОВЫХ ДАННЫХ (ШАГ 10 мкс)")
    print("=" * 80)

    test_gun = GunParameters(
        d=0.122, q=21.76, v_pm=681.0, p_max_target=235.4e6,
        omega=4.352, W_0=0.0068, l_m=3.05,
        I_e=0.96e6, f=1.0e6, k=1.23, delta_powder=1600,
        kappa_1=0.72, lambda_1=0.18, kappa_2=0.54, lambda_2=-0.90, z_e=1.532,
        k_f=0.0003, k_I=0.0016, K=1.05,
        p_0=30e6, p_ign=5e6, n_S=1.04
    )

    calc = BallisticsCalculator(test_gun)

    # Шаг 10 мкс как в методичке
    result = calc.calculate_termo_manual(15, dt=1e-5, verbose=False)

    df_calc = calc.find_reference_points(result)
    print("\nОпорные точки T=15°C:")
    print(df_calc.to_string(index=False))

    expected = {
        '«0»': {'t': 2.21, 'p_m': 30.2358, 'v_p': 0.1252, 'l_p': 5.0035e-7},
        '«max»': {'t': 6.4, 'p_m': 235.4334, 'v_p': 289.0371, 'l_p': 0.4286},
        '«s»': {'t': 7.97, 'p_m': 204.0069, 'v_p': 464.2892, 'l_p': 1.0233},
        '«m»': {'t': 11.4, 'p_m': 75.0216, 'v_p': 681.0003, 'l_p': 3.0502}
    }

    print(f"\n{'=' * 90}")
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА")
    print(f"{'=' * 90}")
    print(f"{'Точка':<8} {'Параметр':<12} {'Методичка':<15} {'Расчёт':<15} {'Отклонение,%':<15}")
    print("-" * 90)

    for point in ['«0»', '«max»', '«s»', '«m»']:
        if point in df_calc['Точка'].values:
            row_calc = df_calc[df_calc['Точка'] == point].iloc[0]
            row_exp = expected[point]

            for param in ['t, мс', 'p_m, МПа', 'v_p, м/с', 'l_p, м']:
                calc_val = row_calc[param]
                exp_key = param.replace(', мс', '').replace(', МПа', '').replace(', м/с', '').replace(', м', '')
                exp_val = row_exp[exp_key]

                if exp_val is None or exp_val == 0:
                    continue

                dev = abs(calc_val - exp_val) / abs(exp_val) * 100
                status = "✓" if dev < 0.5 else "✗"
                print(f"{point:<8} {param:<12} {exp_val:<15.4f} {calc_val:<15.4f} {dev:<15.3f} {status}")

    return calc, result, df_calc


if __name__ == '__main__':
    run_test_verification()