"""Nearshore directional wave forecast, calibrated to the supplied 24 hours.

Usage: python predict.py conditions.json out_dir

The stationary model transports directional spectral action over an inferred
bar/channel bathymetry. It includes refraction, shoaling, bottom dissipation,
depth-induced breaking, reflected waves and their interference, radiation-stress
setup, a channel-current closure, and a finite-depth nonlinear crest correction.
Parameters are fitted offline; prediction needs only NumPy, SciPy and data/grid.npz.
All paths are resolved relative to this file, except the two command-line paths.
"""
from pathlib import Path
import argparse
import json
import numpy as np
from scipy.special import gammaln, lambertw

G = 9.81
DF = 0.02
FIELDS = ('Hs', 'Tm', 'Dm', 'Qb', 'mwl', 'Cs', 'u')

# Effective bathymetry and source-term calibration. Distances are metres.
P = np.array([
    8.829740180454452, .6064979174366546, .14651129405134453,
    1.7353342963005038, 384.5719398883566, 33.63556451505326,
    .6051174296527252, .963440940777976, .025489999362021065,
    65.26089895710263, .968445226572906, 1.0057320452850274,
    .36529938660581773, .2346755441146462,
    .4534467318442965, .05194092021779539,
])
GAP = np.array([1.2741061658945123, 383.59204384857196,
                33.41398164000698, 379.6846264224274,
                47.0549485357204, 1.020257076229906, .8766437347662657])
REFLECT = np.array([.2956784427431953, 1.831905744894497,
                    1.017557644085664, -.041752730611973946])
CURRENT = np.array([.7240673413584451, 1.5315076086950643])

# Nontrivial solution of (1-Q)/log(Q) = -r. The root Q=1 applies for r>=1.
_BJ_R = np.linspace(.0001, .99999, 5000)
_BJ_Q = np.maximum(0., (-_BJ_R * lambertw(
    -np.exp(-1 / _BJ_R) / _BJ_R)).real)


def breaking_fraction(r):
    return np.interp(r, _BJ_R, _BJ_Q, left=0., right=1.)


def dispersion(f, h):
    """Linear finite-depth dispersion; returns k, c, cg and cg/c."""
    omega = 2 * np.pi * f
    kh0 = np.maximum(omega * omega * h / G, 1e-12)
    kh = np.sqrt(kh0 * (1 + kh0))
    for _ in range(7):
        t = np.tanh(kh)
        kh -= (kh * t - kh0) / (t + kh * (1 - t * t))
    k = kh / h
    c = omega / k
    n = .5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 700)))
    return k, c, c * n, n


def interpolate_y(a, coordinate):
    """Interpolate each frequency/direction beam with open lateral boundaries."""
    ny = a.shape[0]
    lower = np.floor(coordinate).astype(np.int32)
    weight = coordinate - lower
    lower = np.clip(lower, 0, ny - 1)
    upper = np.clip(lower + 1, 0, ny - 1)
    weight = np.where(coordinate < 0, 0, weight)
    return (np.take_along_axis(a, lower, axis=0) * (1 - weight)
            + np.take_along_axis(a, upper, axis=0) * weight)


def bathymetry(x, y):
    X, Y = np.meshgrid(x, y, indexing='ij')
    base = (P[0] - P[1] * (X / 100) - P[2] * (X / 100)**2
            - P[3] * np.exp(-((X - P[4]) / P[5])**2))
    channel = GAP[0] * np.exp(-((X - GAP[1]) / GAP[2])**2
                              - ((Y - GAP[3]) / GAP[4])**2)
    depth = np.minimum(base, P[14] + P[15] * (560 - X)) + channel
    # The shallow-water measurement cutoff differs from physical drying.
    # Extend predictions through the shallow cells absent from the observations.
    drying_depth = np.minimum(base, .052 * (560 - X)) + channel
    return depth, drying_depth


def geometry(tide, f, x, y):
    depth, drying_depth = bathymetry(x, y)
    h = np.maximum(depth + tide, P[12])
    k, c, cg, n = dispersion(f[None, None, :], h[:, :, None])
    theta0 = np.deg2rad(np.linspace(-P[9], P[9], 25))
    shape = (len(x), len(y), len(f), len(theta0))
    theta = np.empty(shape)
    theta[0] = theta0
    coordinates = np.zeros(shape)
    focusing = np.ones(shape)
    yi = np.arange(len(y))[:, None, None]
    dy = y[1] - y[0]
    for i in range(1, len(x)):
        dx = x[i] - x[i - 1]
        upstream = yi - dx / dy * np.tan(theta[i - 1])
        previous_angle = interpolate_y(theta[i - 1], upstream)
        angle = np.arcsin(np.clip(np.sin(previous_angle)
                         * (c[i] / c[i - 1])[:, :, None], -.985, .985))
        angle -= dx * GAP[5] * np.gradient(np.log(c[i]), dy, axis=0)[:, :, None]
        theta[i] = np.clip(angle, -1.35, 1.35)
        slope = .5 * (np.tan(theta[i - 1]) + np.tan(theta[i]))
        coordinates[i] = yi - dx / dy * slope
        divergence = np.gradient(np.tan(.5 * (theta[i - 1] + theta[i])), dy, axis=0)
        focusing[i] = np.exp(np.clip(-dx * divergence * GAP[6], -.6, .6))
    velocity_x = cg[:, :, :, None] * np.cos(theta)
    return dict(depth=depth, drying_depth=drying_depth, h=h, k=k,
                cg=cg, n=n, theta=theta, theta0=theta0,
                velocity_x=velocity_x, coordinates=coordinates, focusing=focusing)


def crest_height(Hs, Tm, h):
    k, _, _, _ = dispersion(1 / np.maximum(Tm, 1.), h)
    t = np.tanh(k * h)
    second_order = k * Hs / 16 * (3 - t * t) / (t**3)
    return Hs * (.5 + np.minimum(.3, second_order))


def channel_current_and_action(Hs, Tinc, Dinc, setup, h, x, y):
    """Wave-driven channel jet and local opposing-current action correction."""
    X, Y = np.meshgrid(x, y, indexing='ij')
    ix = (x >= 410) & (x <= 510)
    left = (y >= 130) & (y <= 230)
    right = (y >= 530) & (y <= 630)
    middle = (y >= 350) & (y <= 410)
    head = (.5 * (setup[np.ix_(ix, left)].mean()
                  + setup[np.ix_(ix, right)].mean())
            - setup[np.ix_(ix, middle)].mean())
    speed = CURRENT[0] * np.sqrt(G * max(0., head))
    u = -speed * np.exp(-((X - 440) / 150)**2 - ((Y - 380) / 50)**2)
    if speed < 1e-9:
        return u, Hs

    # Solve omega = intrinsic_sigma(k,h) + u*kx on the incoming branch.
    period = np.maximum(Tinc, 1.)
    omega = 2 * np.pi / period
    angle = np.deg2rad(Dinc)
    k0, _, cg0, _ = dispersion(1 / period, h)
    ky, kx = k0 * np.sin(angle), k0 * np.cos(angle)
    flow = np.maximum(u, -.65 * cg0)
    for _ in range(12):
        k = np.sqrt(kx*kx + ky*ky)
        kh = k * h
        sigma = np.sqrt(G * k * np.tanh(kh))
        n = .5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 700)))
        cg = sigma / k * n
        vx = cg * kx / k + flow
        step = (sigma + flow * kx - omega) / np.maximum(vx, .1)
        kx -= np.clip(step, -.15 * kx, .15 * kx)
    k = np.sqrt(kx*kx + ky*ky)
    kh = k * h
    sigma = np.sqrt(G * k * np.tanh(kh))
    n = .5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 700)))
    vx = sigma / k * n * kx / k + flow
    action_ratio = ((sigma / omega) * cg0 * np.cos(angle)
                    / np.maximum(vx, .25))
    action_ratio = np.clip(action_ratio, .25, 6.)
    return u, Hs * action_ratio**(.5 * CURRENT[1])


def forecast(condition, x, y, f, wall_x=520., wall_y=860.):
    ef = np.asarray(condition['Ef'], dtype=float)
    directions = np.broadcast_to(np.asarray(condition['dirf'], dtype=float), f.shape)
    spreading = np.broadcast_to(np.asarray(condition['sf'], dtype=float), f.shape)
    tide = float(condition['tide'])
    if ef.shape != f.shape:
        raise ValueError(f'Ef must contain {len(f)} spectral densities')
    if (not np.isfinite(ef).all() or np.any(ef < 0)
            or not np.isfinite(directions).all() or not np.isfinite(spreading).all()
            or np.any(spreading < 0) or not np.isfinite(tide)):
        raise ValueError('Conditions must be finite; Ef and sf must be nonnegative')
    directions = (directions + 180) % 360 - 180
    ge = geometry(tide, f, x, y)
    h, k = ge['h'], ge['k']
    theta, vx, n = ge['theta'], ge['velocity_x'], ge['n']
    theta0 = ge['theta0']
    X, Y = np.meshgrid(x, y, indexing='ij')
    wet = ((ge['drying_depth'] + tide > 0)
           & ~((X >= wall_x) & (Y >= wall_y)))
    if ef.sum() <= 1e-14:
        output = {key: np.zeros(h.shape) for key in FIELDS}
        output['mwl'][:] = tide
    else:
        s = spreading * P[10]
        normalizer = np.exp(gammaln(s + 1) - gammaln(s + .5)) / (2 * np.sqrt(np.pi))
        distribution = (np.maximum(np.cos((theta0[None, :] - np.deg2rad(directions)[:, None]) / 2), 0)
                        ** (2 * s[:, None]) * normalizer[:, None]
                        * (theta0[1] - theta0[0]))
        boundary = ef[:, None] * distribution * DF * P[11]**2
        energy = np.empty_like(theta)
        energy[0] = boundary
        Qb = np.empty(h.shape)
        attenuation = np.ones_like(energy)
        for i in range(len(x)):
            dx = x[i] - x[i - 1] if i else 0.
            e = (interpolate_y(energy[i - 1] * vx[i - 1], ge['coordinates'][i])
                 * ge['focusing'][i] / vx[i]) if i else energy[0].copy()
            bottom_loss = (P[8] * (2 * np.pi * f[None, :, None])**2 / G**2
                           / np.sinh(k[i, :, :, None] * h[i, :, None, None])**2)
            decay = np.exp(-bottom_loss * dx / vx[i])
            e *= decay
            m0 = np.maximum(e.sum((1, 2)), 1e-30)
            fm = (e * f[None, :, None]).sum((1, 2)) / m0
            maximum_height = P[6] * h[i]
            q_before = breaking_fraction(8 * m0 / maximum_height**2)
            breaking_loss = .25 * P[7] * q_before * fm * maximum_height**2 / m0
            breaking_decay = np.exp(-np.minimum(30., breaking_loss[:, None, None] * dx / vx[i]))
            e *= breaking_decay
            attenuation[i] = decay * breaking_decay
            energy[i] = e
            q_after = breaking_fraction(8 * e.sum((1, 2)) / maximum_height**2)
            Qb[i] = (1 - P[13]) * q_before + P[13] * q_after

        # Incoherent reflected energy supplies spectral moments and direction;
        # incident/reflected phase interference supplies the local wave height.
        source = int(np.clip(np.searchsorted(x, wall_x) - 1, 0, len(x) - 2))
        reflected = np.zeros_like(energy)
        reflected[source] = REFLECT[0] * energy[source] * (y[:, None, None] >= wall_y)
        yi = np.arange(len(y))[:, None, None]
        dy = y[1] - y[0]
        for i in range(source - 1, -1, -1):
            dx = x[i + 1] - x[i]
            upstream = yi - dx / dy * np.tan(theta[i])
            reflected[i] = (interpolate_y(reflected[i + 1] * vx[i + 1], upstream) / vx[i]
                            * attenuation[i + 1]**(REFLECT[1] - 1) * ge['focusing'][i + 1])
        kx = k[:, :, :, None] * np.cos(theta)
        phase = np.zeros_like(energy)
        phase[source] = ((kx[source] + kx[source + 1]) * (wall_x - x[source])
                         * REFLECT[2] + REFLECT[3])
        for i in range(source - 1, -1, -1):
            phase[i] = phase[i + 1] + (kx[i] + kx[i + 1]) * (x[i + 1] - x[i]) * REFLECT[2]
        interference = (2 * np.sqrt(energy * reflected) * np.cos(phase)
                        * np.exp(-np.maximum(wall_x - x, 0)[:, None, None, None] / 1e6))
        combined = energy + reflected
        m0 = np.maximum(combined.sum((2, 3)), 1e-30)
        m1 = np.maximum((combined * f[None, None, :, None]).sum((2, 3)), 1e-30)
        Hs = 4 * np.sqrt(np.maximum((combined + interference).sum((2, 3)), 0.))
        Tm = m0 / m1
        Dm = np.rad2deg(np.angle((energy * np.exp(1j * theta)
                                   - reflected * np.exp(-1j * theta)).sum((2, 3))))
        mi = np.maximum(energy.sum((2, 3)), 1e-30)
        Tinc = mi / np.maximum((energy * f[None, None, :, None]).sum((2, 3)), 1e-30)
        Dinc = np.rad2deg(np.angle((energy * np.exp(1j * theta)).sum((2, 3))))
        stress_xx = (energy * (n[:, :, :, None] * (1 + np.cos(theta)**2) - .5)).sum((2, 3))
        setup = np.zeros_like(h)
        setup[1:] = np.cumsum(-(stress_xx[1:] - stress_xx[:-1]) / (.5 * (h[1:] + h[:-1])), axis=0)
        u, Hs = channel_current_and_action(Hs, Tinc, Dinc, setup, h, x, y)
        output = dict(Hs=Hs, Tm=Tm, Dm=Dm, Qb=np.clip(Qb, 0, 1),
                      mwl=tide + setup, Cs=crest_height(Hs, Tm, h), u=u)

    for key in FIELDS:
        if not np.isfinite(output[key][wet]).all():
            raise FloatingPointError(f'Nonfinite {key} on a wet cell')
        output[key] = np.where(wet, output[key], np.nan).astype(np.float32)
    return output


def main(conditions_file, out_dir):
    data_dir = Path(__file__).resolve().parent / 'data'
    with np.load(data_dir / 'grid.npz') as grid:
        x, y, f = grid['x'], grid['y'], grid['freq']
    wall_x, wall_y = 520., 860.
    chart_path = data_dir / 'chart.npz'
    if chart_path.exists():
        with np.load(chart_path) as chart:
            structure = np.argwhere(chart['structure'])
            if len(structure):
                wall_x = float(chart['x'][structure[:, 0].min()])
                wall_y = float(chart['y'][structure[:, 1].min()])
    with open(conditions_file, encoding='utf-8') as source:
        conditions = json.load(source)
    if isinstance(conditions, dict):
        conditions = conditions.get('cases', [conditions])
    if not isinstance(conditions, list):
        raise ValueError('conditions.json must contain a list of cases')
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    for index, case in enumerate(conditions):
        case_id = str(case.get('case', case.get('hour', index)))
        if not case_id or '/' in case_id or '\\' in case_id or case_id in ('.', '..'):
            raise ValueError('case identifiers must be valid filename components')
        result = forecast(case, x, y, f, wall_x, wall_y)
        np.savez_compressed(destination / f'case_{case_id}.npz', **result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('conditions_file', type=Path)
    parser.add_argument('out_dir', type=Path)
    arguments = parser.parse_args()
    main(arguments.conditions_file, arguments.out_dir)
