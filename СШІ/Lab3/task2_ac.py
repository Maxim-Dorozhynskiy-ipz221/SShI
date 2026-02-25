import numpy as np
import matplotlib.pyplot as plt

def tri_mf(x, params):
    a, b, c = params
    y = np.zeros_like(x, dtype=float)
    left = (a < x) & (x < b)
    right = (b < x) & (x < c)
    if b != a:
        y[left] = (x[left] - a) / (b - a)
    y[x == b] = 1.0
    if c != b:
        y[right] = (c - x[right]) / (c - b)
    return np.clip(y, 0, 1)

def trap_mf(x, params):
    a, b, c, d = params
    y = np.zeros_like(x, dtype=float)
    y[(b <= x) & (x <= c)] = 1.0
    rise = (a < x) & (x < b)
    fall = (c < x) & (x < d)
    if b != a:
        y[rise] = (x[rise] - a) / (b - a)
    if d != c:
        y[fall] = (d - x[fall]) / (d - c)
    return np.clip(y, 0, 1)

def defuzz_centroid(x, mu):
    area = np.trapezoid(mu, x)
    if area == 0:
        return 0.5 * (x.min() + x.max())
    return np.trapezoid(mu * x, x) / area

def implication(alpha, mf):
    return np.minimum(alpha, mf)

temperature = np.linspace(-10, 40, 1001)
delta_t = np.linspace(-2, 2, 801)
knob_angle = np.linspace(-90, 90, 1001)

T_vcold = tri_mf(temperature, [-10, -10, 5])
T_cold  = tri_mf(temperature, [0, 8, 16])
T_norm  = tri_mf(temperature, [18, 21, 24])
T_warm  = tri_mf(temperature, [22, 26, 30])
T_vwarm = tri_mf(temperature, [28, 40, 40])

dT_neg  = tri_mf(delta_t, [-2, -2, 0])
dT_zero = tri_mf(delta_t, [-0.2, 0, 0.2])
dT_pos  = tri_mf(delta_t, [0, 2, 2])

K_left_big   = trap_mf(knob_angle, [-90, -90, -75, -60])
K_left_small = tri_mf(knob_angle, [-40, -20, 0])
K_center     = tri_mf(knob_angle, [-5, 0, 5])
K_right_small= tri_mf(knob_angle, [0, 20, 40])
K_right_big  = trap_mf(knob_angle, [60, 75, 90, 90])

def fuzzy_ac(temp_val, dtemp_val):
    muT = {
        'vwarm': np.interp(temp_val, temperature, T_vwarm),
        'warm':  np.interp(temp_val, temperature, T_warm),
        'norm':  np.interp(temp_val, temperature, T_norm),
        'cold':  np.interp(temp_val, temperature, T_cold),
        'vcold': np.interp(temp_val, temperature, T_vcold),
    }
    muD = {
        'neg':  np.interp(dtemp_val, delta_t, dT_neg),
        'zero': np.interp(dtemp_val, delta_t, dT_zero),
        'pos':  np.interp(dtemp_val, delta_t, dT_pos),
    }

    K = np.zeros_like(knob_angle)

    def add_rule(alpha, out_set):
        nonlocal K
        K = np.maximum(K, implication(alpha, out_set))

    add_rule(min(muT['vwarm'], muD['pos']),  K_left_big)
    add_rule(min(muT['vwarm'], muD['neg']),  K_left_small)
    add_rule(min(muT['warm'],  muD['pos']),  K_left_big)
    add_rule(min(muT['warm'],  muD['neg']),  K_center)
    add_rule(min(muT['vcold'], muD['neg']),  K_right_big)
    add_rule(min(muT['vcold'], muD['pos']),  K_right_small)
    add_rule(min(muT['cold'],  muD['neg']),  K_right_big)
    add_rule(min(muT['cold'],  muD['pos']),  K_center)
    add_rule(min(muT['vwarm'], muD['zero']), K_left_big)
    add_rule(min(muT['warm'],  muD['zero']), K_left_small)
    add_rule(min(muT['vcold'], muD['zero']), K_right_big)
    add_rule(min(muT['cold'],  muD['zero']), K_right_small)
    add_rule(min(muT['norm'],  muD['pos']),  K_left_small)
    add_rule(min(muT['norm'],  muD['neg']),  K_right_small)
    add_rule(min(muT['norm'],  muD['zero']), K_center)

    angle_out = defuzz_centroid(knob_angle, K)
    return angle_out, K

def draw_surface():
    Tg, dTg = np.meshgrid(np.linspace(-5,35,41), np.linspace(-1.5,1.5,41))
    infer_vec = np.vectorize(lambda t, d: fuzzy_ac(t, d)[0])
    Ksurf = infer_vec(Tg, dTg)

    fig = plt.figure(figsize=(6,4.5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(Tg, dTg, Ksurf, cmap='viridis', edgecolor='none', alpha=0.95)
    ax.set_xlabel('Temperature (°C)')
    ax.set_ylabel('ΔT (°C/unit)')
    ax.set_zlabel('Knob angle (°)')
    ax.set_title('AC controller surface')
    plt.tight_layout()
    plt.savefig('ac_surface.png', dpi=160)
    plt.show()

if __name__ == '__main__':
    for (tv, dv) in [(30, 0.0), (30, 0.8), (20, 0.0), (12, -0.6), (28, -0.4)]:
        k, _ = fuzzy_ac(tv, dv)
        print(f'T={tv}°C, ΔT={dv:+.2f} → knob={k:.1f}°')

    draw_surface()