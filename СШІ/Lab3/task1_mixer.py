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

def centroid_defuzz(x, mu):
    area = np.trapezoid(mu, x)
    if area == 0:
        return 0.5 * (x.min() + x.max())
    return np.trapezoid(mu * x, x) / area

def implication(alpha, mf):
    return np.minimum(alpha, mf)

temperature = np.linspace(0, 100, 1001)
pressure = np.linspace(0, 100, 1001)
theta = np.linspace(-90, 90, 1001)

Tcold  = tri_mf(temperature, [0, 0, 20])
Tcool  = tri_mf(temperature, [10, 25, 40])
Twarm  = tri_mf(temperature, [30, 45, 60])
Tnvhot = tri_mf(temperature, [50, 65, 80])
Thot   = tri_mf(temperature, [70, 100, 100])

Pweak   = tri_mf(pressure, [0, 0, 35])
Pnvs    = tri_mf(pressure, [25, 50, 75])
Pstrong = tri_mf(pressure, [60, 100, 100])

ALleft  = trap_mf(theta, [-90, -90, -75, -60])
AMleft  = tri_mf(theta, [-70, -40, -10])
ASleft  = tri_mf(theta, [-25, -10, -2])
Azero   = tri_mf(theta, [-5, 0, 5])
ASright = tri_mf(theta, [2, 10, 25])
AMright = tri_mf(theta, [10, 40, 70])
ALright = trap_mf(theta, [60, 75, 90, 90])

def fuzzy_mixer(t_val, p_val):
    t_mu = {
        'hot':   np.interp(t_val, temperature, Thot),
        'nvhot': np.interp(t_val, temperature, Tnvhot),
        'warm':  np.interp(t_val, temperature, Twarm),
        'cool':  np.interp(t_val, temperature, Tcool),
        'cold':  np.interp(t_val, temperature, Tcold)
    }
    p_mu = {
        'strong': np.interp(p_val, pressure, Pstrong),
        'nvs':    np.interp(p_val, pressure, Pnvs),
        'weak':   np.interp(p_val, pressure, Pweak)
    }

    hot_out = np.zeros_like(theta)
    cold_out = np.zeros_like(theta)

    sets = {
        'LL': ALleft, 'ML': AMleft, 'SL': ASleft,
        'ZR': Azero,
        'SR': ASright, 'MR': AMright, 'LR': ALright
    }

    def add_rule(alpha, hot_set, cold_set):
        nonlocal hot_out, cold_out
        hot_out = np.maximum(hot_out, implication(alpha, hot_set))
        cold_out = np.maximum(cold_out, implication(alpha, cold_set))

    add_rule(min(t_mu['hot'],   p_mu['strong']), sets['ML'], sets['MR'])
    add_rule(min(t_mu['hot'],   p_mu['nvs']),    sets['ZR'], sets['MR'])
    add_rule(min(t_mu['nvhot'], p_mu['strong']), sets['SL'], sets['ZR'])
    add_rule(min(t_mu['nvhot'], p_mu['weak']),   sets['SR'], sets['SR'])
    add_rule(min(t_mu['warm'],  p_mu['nvs']),    sets['ZR'], sets['ZR'])
    add_rule(min(t_mu['cool'],  p_mu['strong']), sets['MR'], sets['ML'])
    add_rule(min(t_mu['cool'],  p_mu['nvs']),    sets['MR'], sets['SL'])
    add_rule(min(t_mu['cold'],  p_mu['weak']),   sets['LR'], sets['ZR'])
    add_rule(min(t_mu['cold'],  p_mu['strong']), sets['ML'], sets['MR'])
    add_rule(min(t_mu['warm'],  p_mu['strong']), sets['SL'], sets['SL'])
    add_rule(min(t_mu['warm'],  p_mu['weak']),   sets['SR'], sets['SR'])

    hot_angle = centroid_defuzz(theta, hot_out)
    cold_angle = centroid_defuzz(theta, cold_out)
    return hot_angle, cold_angle

def draw_surfaces():
    Tg, Pg = np.meshgrid(np.linspace(0,100,41), np.linspace(0,100,41))
    infer_vec = np.vectorize(lambda t, p: fuzzy_mixer(t, p))
    Hsurf, Csurf = infer_vec(Tg, Pg)

    fig = plt.figure(figsize=(11,4.5))
    ax1 = fig.add_subplot(121, projection='3d')
    ax1.plot_surface(Tg, Pg, Hsurf, cmap='coolwarm', edgecolor='none', alpha=0.95)
    ax1.set_title('Hot tap angle (°)')
    ax1.set_xlabel('Temperature'); ax1.set_ylabel('Pressure'); ax1.set_zlabel('Angle')

    ax2 = fig.add_subplot(122, projection='3d')
    ax2.plot_surface(Tg, Pg, Csurf, cmap='coolwarm', edgecolor='none', alpha=0.95)
    ax2.set_title('Cold tap angle (°)')
    ax2.set_xlabel('Temperature'); ax2.set_ylabel('Pressure'); ax2.set_zlabel('Angle')

    plt.tight_layout()
    plt.savefig('mixer_surfaces.png', dpi=160)
    plt.show()

if __name__ == '__main__':
    ha, ca = fuzzy_mixer(80, 70)
    print(f'T=80, P=70 → hot={ha:.1f}°, cold={ca:.1f}°')
    draw_surfaces()