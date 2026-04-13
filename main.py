import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass
from typing import Dict, List, Tuple
import pandas as pd

from ODE_solvers import RungeKutta4
from pyballistics import get_db_powder, get_powder_names


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


class PowderSelector:
    """Выбор пороха из БД pyballistics по методичке стр. 3"""

    def __init__(self):
        self.all_powders = get_powder_names()

    def filter_by_type(self, powder_type: str) -> List[str]:
        if powder_type == 'tube':
            return [p for p in self.all_powders if 'тр' in p.lower()]
        elif powder_type == 'seven':
            return [p for p in self.all_powders if '/' in p and 'тр' not in p.lower()]
        return self.all_powders

    def check_force_valid(self, f: float) -> bool:
        f_mj = f / 1e6
        return 0.85 <= f_mj <= 1.05

    def select_powders(self, I_e_target: float, powder_type: str, n_candidates: int = 5) -> Tuple[
        pd.DataFrame, List[Dict]]:
        candidates = []
        type_names = self.filter_by_type(powder_type)

        print(f"\n{'=' * 60}")
        print(f"ПОДБОР ПОРОХА ТИПА: {'ТРУБЧАТЫЙ' if powder_type == 'tube' else 'СЕМИКАНАЛЬНЫЙ'}")
        print(f"Целевой импульс I_e = {I_e_target / 1e6:.2f} МПа·с")
        print(f"{'=' * 60}")

        for name in type_names:
            try:
                params = get_db_powder(name)
                I_e = params['I_e']
                f = params['f']

                if not self.check_force_valid(f):
                    continue

                rel_diff = abs(I_e - I_e_target) / I_e_target * 100

                candidates.append({
                    'name': name,
                    'I_e_MPa_s': I_e / 1e6,
                    'f_MJ_kg': f / 1e6,
                    'k': params['k'],
                    'delta': params['delta'],
                    'rel_diff_pct': rel_diff,
                    'params': params
                })
            except:
                continue

        candidates.sort(key=lambda x: x['rel_diff_pct'])
        selected = candidates[:n_candidates]

        table_data = []
        for i, cand in enumerate(selected, 1):
            table_data.append({
                '№ п/п': i,
                'Марка пороха': cand['name'],
                'I_e, МПа·с': f"{cand['I_e_MPa_s']:.3f}",
                'f, МДж/кг': f"{cand['f_MJ_kg']:.3f}",
                'Отклонение, %': f"{cand['rel_diff_pct']:.1f}"
            })

        df = pd.DataFrame(table_data)
        print("\nТаблица выбранных марок пороха:")
        print(df.to_string(index=False))

        return df, selected

    def justify_selection(self, selected: List[Dict], I_e_target: float) -> Dict:
        if not selected:
            raise ValueError("Нет подходящих марок пороха")

        best = min(selected, key=lambda x: (x['rel_diff_pct'], abs(x['f_MJ_kg'] - 1.0)))

        print(f"\n{'=' * 60}")
        print("ОБОСНОВАНИЕ ВЫБОРА")
        print(f"{'=' * 60}")
        print(f"Выбрана марка: {best['name']}")
        print(f"  Импульс I_e = {best['I_e_MPa_s']:.3f} МПа·с (отклонение {best['rel_diff_pct']:.1f}%)")
        print(f"  Сила f = {best['f_MJ_kg']:.3f} МДж/кг")
        print(f"  k = {best['k']:.3f}, δ = {best['delta']:.0f} кг/м³")

        return best


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

        mach = v_p / self.gun.c_0a
        if mach < 1.0:
            p_a = self.gun.p_a0 * (1 + self.gun.k_air * (self.gun.k_air + 1) / 4 * mach ** 2)
        else:
            p_a = self.gun.p_a0 * (2 * self.gun.k_air * mach ** 2 - (self.gun.k_air - 1)) / (self.gun.k_air + 1)

        pressure_diff = p_m - p_a

        if v_p > 1e-9:
            dv_p_dt = pressure_diff * self.gun.S / (phi * self.gun.q)
        else:
            if pressure_diff > self.gun.p_0:
                dv_p_dt = pressure_diff * self.gun.S / (phi * self.gun.q)
            else:
                dv_p_dt = 0.0

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

    def calculate_termo_manual(self, T0: float, dt: float = 1e-5) -> Dict:
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
        t = result['t']
        p_m = result['p_m']
        v_p = result['v_p']
        x_p = result['x_p']
        z = result['z_1']
        psi = result['psi_1']
        l_0 = self.gun.l_0
        l_p = x_p - l_0
        points = []

        # Точка «0»: v_p = 0.1252 м/с
        target_v0 = 0.1252
        for i in range(1, len(v_p)):
            if v_p[i - 1] < target_v0 <= v_p[i]:
                alpha = (target_v0 - v_p[i - 1]) / (v_p[i] - v_p[i - 1])
                t_0 = t[i - 1] + alpha * (t[i] - t[i - 1])
                p_0 = p_m[i - 1] + alpha * (p_m[i] - p_m[i - 1])
                l_0_val = l_p[i - 1] + alpha * (l_p[i] - l_p[i - 1])
                points.append({
                    'Точка': '«0»', 't, мс': t_0 * 1000, 'p_m, МПа': p_0 / 1e6,
                    'v_p, м/с': target_v0, 'l_p, м': l_0_val
                })
                break

        # Точка «max»
        idx_max = np.argmax(p_m)
        points.append({
            'Точка': '«max»', 't, мс': t[idx_max] * 1000, 'p_m, МПа': p_m[idx_max] / 1e6,
            'v_p, м/с': v_p[idx_max], 'l_p, м': l_p[idx_max]
        })

        # Точка «s»: z = 1
        for i in range(1, len(z)):
            if z[i - 1] < 1.0 <= z[i]:
                alpha = (1.0 - z[i - 1]) / (z[i] - z[i - 1])
                points.append({
                    'Точка': '«s»', 't, мс': (t[i - 1] + alpha * (t[i] - t[i - 1])) * 1000,
                    'p_m, МПа': (p_m[i - 1] + alpha * (p_m[i] - p_m[i - 1])) / 1e6,
                    'v_p, м/с': v_p[i - 1] + alpha * (v_p[i] - v_p[i - 1]),
                    'l_p, м': l_p[i - 1] + alpha * (l_p[i] - l_p[i - 1])
                })
                break

        # Точка «e»: psi = 0.99
        for i in range(1, len(psi)):
            if psi[i - 1] < 0.99 <= psi[i]:
                alpha = (0.99 - psi[i - 1]) / (psi[i] - psi[i - 1])
                points.append({
                    'Точка': '«e»', 't, мс': (t[i - 1] + alpha * (t[i] - t[i - 1])) * 1000,
                    'p_m, МПа': (p_m[i - 1] + alpha * (p_m[i] - p_m[i - 1])) / 1e6,
                    'v_p, м/с': v_p[i - 1] + alpha * (v_p[i] - v_p[i - 1]),
                    'l_p, м': l_p[i - 1] + alpha * (l_p[i] - l_p[i - 1])
                })
                break

        # Точка «m»
        points.append({
            'Точка': '«m»', 't, мс': t[-1] * 1000, 'p_m, МПа': p_m[-1] / 1e6,
            'v_p, м/с': v_p[-1], 'l_p, м': l_p[-1]
        })

        return pd.DataFrame(points)


# =============================================================================
# ТЕСТОВАЯ ЗАДАЧА (валидация на данных методички)
# =============================================================================
def run_test_verification():
    """Проверка на тестовых данных методички (стр. 1 тестовых данных)"""
    print("=" * 80)
    print("ПРОВЕРКА НА ТЕСТОВЫХ ДАННЫХ (НУЛЬМЕРНАЯ ПОСТАНОВКА)")
    print("=" * 80)

    # Точные параметры из методички
    test_gun = GunParameters(
        d=0.122,  # 122 мм
        q=21.76,  # кг
        v_pm=681.0,  # м/с
        p_max_target=235.4e6,  # Па
        omega=4.352,  # кг
        W_0=0.0068,  # м³
        l_m=3.05,  # м
        I_e=0.96e6,  # Па·с (0,96·10⁶)
        f=1.0e6,  # Дж/кг (10⁶)
        k=1.23,
        delta_powder=1600,  # кг/м³
        kappa_1=0.72,
        lambda_1=0.18,
        kappa_2=0.54,
        lambda_2=-0.90,
        z_e=1.532,
        k_f=0.0003,
        k_I=0.0016,
        K=1.05,  # для v_pm = 681 м/с (600 < v <= 700)
        p_0=30e6,
        p_ign=5e6,
        n_S=1.04
    )

    print(f"\nПараметры орудия:")
    print(f"  Калибр d = {test_gun.d * 1000:.0f} мм")
    print(f"  Масса снаряда q = {test_gun.q:.2f} кг")
    print(f"  Масса заряда ω = {test_gun.omega:.3f} кг")
    print(f"  Объём каморы W₀ = {test_gun.W_0:.4f} м³")
    print(f"  Длина ствола l_m = {test_gun.l_m:.2f} м")
    print(f"  Площадь S = {test_gun.S:.6f} м²")
    print(f"  Начальная длина l₀ = {test_gun.l_0:.4f} м")
    print(f"  Плотность заряжания Δ = {test_gun.Delta:.0f} кг/м³")
    print(f"  Коэффициент φ = {test_gun.get_phi():.4f}")

    calc = BallisticsCalculator(test_gun)

    # Расчёт для T = 15°C
    result = calc.calculate_termo_manual(15, dt=1e-5)
    df_calc = calc.find_reference_points(result)

    print(f"\nОпорные точки T=15°C (расчёт):")
    print(df_calc.to_string(index=False))

    # Ожидаемые значения из таблицы 1 методички
    expected = {
        '«0»': {'t': 2.21, 'p_m': 30.2358, 'v_p': 0.1252, 'l_p': 5.0035e-7},
        '«max»': {'t': 6.4, 'p_m': 235.4334, 'v_p': 289.0371, 'l_p': 0.4286},
        '«s»': {'t': 7.97, 'p_m': 204.0069, 'v_p': 464.2892, 'l_p': 1.0233},
        '«m»': {'t': 11.4, 'p_m': 75.0216, 'v_p': 681.0003, 'l_p': 3.0502}
    }

    print(f"\n{'=' * 90}")
    print("СРАВНИТЕЛЬНАЯ ТАБЛИЦА (Т₀ = +15°C)")
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
                status = "✓" if dev < 2.0 else "✗"
                print(f"{point:<8} {param:<12} {exp_val:<15.4f} {calc_val:<15.4f} {dev:<15.3f} {status}")

    return calc, result, df_calc


# =============================================================================
# ЗАДАЧА С ВЫБОРОМ ПОРОХА (для варианта)
# =============================================================================
def gunpowder_issue():
    """Решение прямой задачи с выбором пороха из БД (последняя строка таблицы)"""
    print("\n\n" + "=" * 80)
    print("РЕШЕНИЕ ПРЯМОЙ ЗАДАЧИ С ВЫБОРОМ ПОРОХА (ВАРИАНТ)")
    print("=" * 80)

    # ===== ЭТАП 1: Выбор пороха =====
    selector = PowderSelector()

    # Данные из обратной задачи для последней строки таблицы
    # Трубчатый порох: I_e = 1.72 МПа·с
    I_e_tube_target = 1.72e6
    # Семиканальный порох: I_e = 1.36 МПа·с
    I_e_seven_target = 1.36e6

    # Выбор трубчатого пороха
    table_tube, candidates_tube = selector.select_powders(
        I_e_tube_target, 'tube', n_candidates=5
    )
    best_tube = selector.justify_selection(candidates_tube, I_e_tube_target)

    # Выбор семиканального пороха
    print(f"\n{'=' * 60}")
    table_seven, candidates_seven = selector.select_powders(
        I_e_seven_target, 'seven', n_candidates=5
    )
    best_seven = selector.justify_selection(candidates_seven, I_e_seven_target)

    # ===== ЭТАП 2: Формирование исходных данных (трубчатый порох) =====
    print(f"\n{'=' * 80}")
    print("ФОРМИРОВАНИЕ ИСХОДНЫХ ДАННЫХ (ТРУБЧАТЫЙ ПОРОХ)")
    print(f"{'=' * 80}")

    q = 21.76  # кг
    d = 0.122  # м
    v_pm = 995  # м/с (последняя строка таблицы)

    # Параметры из обратной задачи для трубчатого пороха
    omega_tube = 9.183  # q * 0.422
    W_0_tube = 0.012083  # omega / 760
    l_m_tube = 4.526  # 37.1 * d

    gun_tube = GunParameters(
        d=d, q=q, v_pm=v_pm, p_max_target=355e6,
        omega=omega_tube, W_0=W_0_tube, l_m=l_m_tube,
        I_e=best_tube['params']['I_e'],
        f=best_tube['params']['f'],
        k=best_tube['params']['k'],
        delta_powder=best_tube['params']['delta'],
        kappa_1=best_tube['params']['kappa_1'],
        lambda_1=best_tube['params']['lambda_1'],
        kappa_2=best_tube['params']['kappa_2'],
        lambda_2=best_tube['params']['lambda_2'],
        z_e=best_tube['params']['z_e'],
        K=1.03  # для v_pm > 800 м/с
    )

    print(f"\nПараметры орудия:")
    print(f"  Калибр d = {gun_tube.d * 1000:.0f} мм")
    print(f"  Масса снаряда q = {gun_tube.q:.2f} кг")
    print(f"  Дульная скорость v_pm = {gun_tube.v_pm:.0f} м/с")
    print(f"  Масса заряда ω = {gun_tube.omega:.3f} кг")
    print(f"  Плотность заряжания Δ = {gun_tube.Delta:.0f} кг/м³")
    print(f"  Длина ствола l_m = {gun_tube.l_m:.3f} м")
    print(f"  Выбран порох: {best_tube['name']}")

    # ===== ЭТАП 3: Расчёт для трёх температур =====
    calc_tube = BallisticsCalculator(gun_tube)
    temperatures = [15, 50, -50]

    for T in temperatures:
        print(f"\n{'=' * 60}")
        print(f"РАСЧЁТ ТРУБЧАТОГО ПОРОХА, T₀ = {T}°C")
        print(f"{'=' * 60}")

        result = calc_tube.calculate_termo_manual(T, dt=1e-5)
        df_points = calc_tube.find_reference_points(result)

        print("\nОпорные точки:")
        print(df_points.to_string(index=False))
        df_points.to_csv(f'tube_termo_T{T}.csv', index=False)

    # ===== ЭТАП 4: Расчёт для семиканального пороха (опционально) =====
    print(f"\n{'=' * 80}")
    print("ФОРМИРОВАНИЕ ИСХОДНЫХ ДАННЫХ (СЕМИКАНАЛЬНЫЙ ПОРОХ)")
    print(f"{'=' * 80}")

    omega_seven = 8.791  # q * 0.404
    W_0_seven = 0.011270  # omega / 780
    l_m_seven = 4.697  # 38.5 * d

    gun_seven = GunParameters(
        d=d, q=q, v_pm=v_pm, p_max_target=355e6,
        omega=omega_seven, W_0=W_0_seven, l_m=l_m_seven,
        I_e=best_seven['params']['I_e'],
        f=best_seven['params']['f'],
        k=best_seven['params']['k'],
        delta_powder=best_seven['params']['delta'],
        kappa_1=best_seven['params']['kappa_1'],
        lambda_1=best_seven['params']['lambda_1'],
        kappa_2=best_seven['params']['kappa_2'],
        lambda_2=best_seven['params']['lambda_2'],
        z_e=best_seven['params']['z_e'],
        K=1.03
    )

    print(f"\nПараметры орудия:")
    print(f"  Масса заряда ω = {gun_seven.omega:.3f} кг")
    print(f"  Плотность заряжания Δ = {gun_seven.Delta:.0f} кг/м³")
    print(f"  Длина ствола l_m = {gun_seven.l_m:.3f} м")
    print(f"  Выбран порох: {best_seven['name']}")

    calc_seven = BallisticsCalculator(gun_seven)

    for T in temperatures:
        print(f"\n{'=' * 60}")
        print(f"РАСЧЁТ СЕМИКАНАЛЬНОГО ПОРОХА, T₀ = {T}°C")
        print(f"{'=' * 60}")

        result = calc_seven.calculate_termo_manual(T, dt=1e-5)
        df_points = calc_seven.find_reference_points(result)

        print("\nОпорные точки:")
        print(df_points.to_string(index=False))
        df_points.to_csv(f'seven_termo_T{T}.csv', index=False)

    return best_tube, best_seven


# =============================================================================
# ТОЧКА ВХОДА
# =============================================================================
if __name__ == '__main__':
    # Запуск тестовой задачи (валидация)
    print("ЗАПУСК ТЕСТОВОЙ ЗАДАЧИ...")
    run_test_verification()

    # Запуск основной задачи с выбором пороха
    print("\n\nЗАПУСК ОСНОВНОЙ ЗАДАЧИ...")
    gunpowder_issue()