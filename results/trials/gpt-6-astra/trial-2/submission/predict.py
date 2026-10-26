"""Nearshore wave forecast, calibrated to the supplied 24-hour observing record.

Usage: python predict.py conditions.json out_dir

The solver propagates a frequency/direction spectrum with refraction, diffraction,
wind input, bottom friction, depth-limited breaking, and partial structure
reflection. Radiation-stress gradients drive setup and the channel outflow.
The fitted bathymetry follows a Dean beach profile with a displaced bar and a gap.
All coefficients are frozen: forecasting does not fit to or read the observations.
Only NumPy and SciPy are required. Site data are located relative to this file.
"""

import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.fft import fft, fftfreq, ifft
from scipy.integrate import cumulative_trapezoid
from scipy.special import expit, gammaln, ndtr

# Incoming cone and bearing offset; reflection amplitude, ray displacement,
# aperture width and attenuation; coherence and length; breaker gamma and rate;
# bottom friction, source-depth quadrature, and wind growth. Lengths are metres.
MODEL = np.array(
    [
        65.97222365698202,
        -0.351622931901472,
        0.5602591914649069,
        1.166760880732892,
        20.315155647952086,
        0.00011513825141287963,
        1.2999999997347618,
        219.01251102022485,
        0.6258725916660574,
        1.3298039294222956,
        0.04267846812914969,
        0.8178179742322208,
        0.00038435011238913347,
    ],
    dtype=float,
)
# Channel depth, cross-shore center/width, alongshore center/width, diffraction.
CHANNEL = np.array(
    [
        1.3652156287727715,
        384.90733934710994,
        23.536917513574167,
        381.19188122031494,
        33.02306008466822,
        0.92560874582726,
    ],
    dtype=float,
)
SETUP = np.array([0.8943537671359032, 0.9148876117130075])
CURRENT = np.array([1.9888363683556256, 1.0450683547142243])
JET = np.array(
    [443.68648400784156, 105.70247410259414, 380.5954663801026, 35.86627654881683]
)

QDIAG = np.array(
    [
        -0.019685292106048097,
        0.1572560265225842,
        -0.43305862625884156,
        0.17693159619363125,
    ]
)
EDGE = np.array(
    [
        31.6412222694775,
        526.4436452613509,
        6.852793431535695,
        4.165732697004537,
        15.392275834470999,
        0.3475315239509191,
    ]
)
EDGE_HEIGHT = np.array([3.113784709023962, 2.0000000000000004])

G = 9.81
_Q = np.r_[0.0, np.geomspace(1e-14, 0.999999, 20000), 1.0]
_R = np.r_[0.0, -(1 - _Q[1:-1]) / np.log(_Q[1:-1]), 1.0]


def wavenumber(f, h):
    """Solve the finite-depth linear dispersion relation with Newton iteration."""
    w = np.maximum(2 * np.pi * f, 1e-9)
    k = np.maximum(w * w / G, w / np.sqrt(G * h))
    for _ in range(9):
        t = np.tanh(k * h)
        k -= (G * k * t - w * w) / (G * (t + k * h * (1 - t * t)))
    return k


def _solve_waves(
    E,
    theta,
    depth,
    f,
    pars,
    diffraction=1.0,
    ns=1,
    forcing=None,
    wall=None,
    return_detail=False,
):
    """March spectral action flux with a split-step mild-slope approximation.

    Each frequency/direction component carries a complex flux amplitude. The
    Fourier step resolves lateral diffraction; the phase screen refracts waves
    over the local bottom. Breaking couples the components through total energy.
    Output columns are Hs, Tm, Dm, Qb, Sxx/(rho*g), Cs, Sxy/(rho*g), and setup.
    """
    gamma, alpha, cf, hfloor, mix = pars[:5]
    nx, ny = depth.shape
    windcoef = pars[5] if len(pars) > 5 else 0.0
    U = 0.0 if forcing is None else float(forcing[0])
    wd = 0.0 if forcing is None else float(forcing[1])
    pad = 50
    hh = np.pad(depth, ((0, 0), (pad, pad)), mode="edge")
    nyy = hh.shape[1]
    h = np.maximum(hh, hfloor)
    K = wavenumber(f[None, :, None], h[:, None, :])
    w = 2 * np.pi * f[:, None]
    N = 0.5 * (
        1 + 2 * K * h[:, None, :] / np.sinh(np.minimum(2 * K * h[:, None, :], 500))
    )
    CG = w[None, :, :] / K * N
    ky0 = K[0, :, 0, None] * np.sin(theta)[None, :]
    A = (
        np.broadcast_to(
            np.sqrt(E * CG[0, :, 0, None] * np.cos(theta)[None, :])[:, :, None],
            (len(f), len(theta), nyy),
        )
        .astype(complex)
        .copy()
    )
    bg = np.empty((nx, len(f), len(theta)))
    fbg = np.empty_like(bg)
    dbg = np.empty_like(bg)
    mxx = np.empty((nx, ny))
    myy = np.empty((nx, ny))
    ratios = np.empty((nx, ny))
    wallmask = (
        np.zeros_like(hh, dtype=bool)
        if wall is None
        else np.pad(wall, ((0, 0), (pad, pad)), mode="edge")
    )
    qr = 2 * np.pi * fftfreq(nyy, 10)
    out = np.zeros((nx, nyy, 8))
    efs = np.zeros((nx, nyy, len(f)))
    dirs = np.zeros_like(efs)
    for ix in range(nx):
        if ix:
            for sub in range(ns):
                v = (sub + 0.5) / ns
                km = (1 - v) * K[ix - 1] + v * K[ix]
                kref = km[:, 0, None]
                kxr = np.sqrt(np.maximum(kref * kref - ky0**2, 1e-10))
                local = np.sqrt(
                    (km[:, None, :] ** 2 - ky0[:, :, None] ** 2).astype(complex)
                )
                potential = np.exp(1j * (10 / ns / 2) * (local - kxr[:, :, None]))
                # Wide-angle free propagation of the slowly varying envelope.
                kq = np.sqrt(
                    (
                        kref[:, :, None] ** 2
                        - (ky0[:, :, None] + qr[None, None, :]) ** 2
                    ).astype(complex)
                )
                propagator = np.exp(1j * (10 / ns) * (kq - kxr[:, :, None]))
                if diffraction != 1:
                    # Independent strength of diffraction with the same ray advection.
                    adv = np.exp(
                        -1j * (10 / ns) * (ky0 / kxr)[:, :, None] * qr[None, None, :]
                    )
                    propagator = adv * np.exp(
                        1j
                        * (10 / ns)
                        * diffraction
                        * (
                            kq
                            - kxr[:, :, None]
                            + (ky0 / kxr)[:, :, None] * qr[None, None, :]
                        )
                    )
                A *= potential
                A = ifft(fft(A, axis=-1) * propagator, axis=-1)
                A *= potential
        # Local phase gradients determine the propagation directions.
        ky = np.empty_like(A.real)
        ky[:, :, 1:-1] = np.angle(A[:, :, 2:] * A[:, :, :-2].conj()) / 20
        ky[:, :, 0] = ky[:, :, 1]
        ky[:, :, -1] = ky[:, :, -2]
        ky += ky0[:, :, None]
        si = np.clip(ky / K[ix, :, None, :], -0.98, 0.98)
        co = np.sqrt(1 - si * si)
        speed = CG[ix, :, None, :] * co
        if ix:
            hm = np.maximum(mix * hh[ix - 1] + (1 - mix) * hh[ix], hfloor)
            km = wavenumber(f[:, None], hm[None, :])
            nm = 0.5 * (
                1
                + 2 * km * hm[None, :] / np.sinh(np.minimum(2 * km * hm[None, :], 500))
            )
            cgm = w / km * nm
            com = np.sqrt(1 - np.clip(ky / km[:, None, :], -0.98, 0.98) ** 2)
            gm = cgm[:, None, :] * com
            ei = abs(A) ** 2 / np.maximum(gm, 0.05)
            m0 = ei.sum((0, 1))
            m1 = (ei * f[:, None, None]).sum((0, 1))
            r = 8 * m0 / (gamma * hm) ** 2
            q = np.interp(r, _R, _Q)
            rate = 2 * alpha * m1 / np.maximum(m0, 1e-12) * q / np.maximum(r, 1e-12)
            fric = (
                cf * w * w / (G * G * np.sinh(np.minimum(km * hm[None, :], 100)) ** 2)
            )
            kmbar = wavenumber(m1 / np.maximum(m0, 1e-12), hm)
            wrate = (
                2.36e-5
                * (2 * np.pi * m1 / np.maximum(m0, 1e-12))
                * (m0 * kmbar**2 / 0.00302) ** 2
            )
            sm = np.clip(ky / km[:, None, :], -0.98, 0.98)
            grow = (
                windcoef
                * w[:, None, :]
                * np.maximum(
                    U * (com * np.cos(wd) + sm * np.sin(wd)) / (w / km)[:, None, :] - 1,
                    0,
                )
            )
            A *= np.exp(
                -0.5
                * np.clip(
                    (
                        rate[None, None, :]
                        + fric[:, None, :]
                        + wrate[None, None, :]
                        - grow
                    )
                    * 10
                    / np.maximum(gm, 0.1),
                    -10,
                    50,
                )
            )
        else:
            q = np.zeros(nyy)
        e = abs(A) ** 2 / np.maximum(speed, 0.05)
        m0 = e.sum((0, 1))
        m1 = (e * f[:, None, None]).sum((0, 1))
        mx = (e * co).sum((0, 1))
        my = (e * si).sum((0, 1))
        Hs = 4 * np.sqrt(np.maximum(m0, 1e-12))
        Tm = m0 / np.maximum(m1, 1e-12)
        Dm = np.rad2deg(np.arctan2(my, mx))
        stress = (e * (N[ix, :, None, :] * (1 + co * co) - 0.5)).sum((0, 1))
        sxy = (e * N[ix, :, None, :] * si * co).sum((0, 1))
        km = wavenumber(1 / np.maximum(Tm, 1), h[ix])
        sig = np.tanh(km * h[ix])
        crest = 0.5 * Hs + np.minimum(
            0.3 * Hs, km * Hs**2 * (3 - sig**2) / (16 * sig**3)
        )
        if ix == 0:
            r = 8 * m0 / (gamma * h[ix]) ** 2
            q = np.interp(r, _R, _Q)
        ratios[ix] = r[pad:-pad]
        out[ix, :, 0] = Hs
        out[ix, :, 1] = Tm
        out[ix, :, 2] = Dm
        out[ix, :, 3] = q
        out[ix, :, 4] = stress
        out[ix, :, 5] = crest
        out[ix, :, 6] = sxy
        if ix:
            out[ix, :, 7] = out[ix - 1, :, 7] - (
                stress - out[ix - 1, :, 4] + 10 * np.gradient(sxy, 10)
            ) / np.maximum(0.5 * (hh[ix] + hh[ix - 1]), 0.25)
        efs[ix] = e.sum(1).T
        dirs[ix] = np.arctan2((e * si).sum(1), (e * co).sum(1)).T
        mxx[ix] = mx[pad:-pad]
        myy[ix] = my[pad:-pad]
        bg[ix] = e[:, :, pad + 95]
        fbg[ix] = abs(A[:, :, pad + 95]) ** 2
        dbg[ix] = np.arcsin(si[:, :, pad + 95])
        A[:, :, (hh[ix] <= 0.08) | wallmask[ix]] = 0
    if return_detail:
        return (
            out[:, pad:-pad],
            efs[:, pad:-pad],
            dirs[:, pad:-pad],
            {
                "energy": bg,
                "flux": fbg,
                "direction": dbg,
                "mx": mxx,
                "my": myy,
                "ratio": ratios,
            },
        )
    return out[:, pad:-pad], efs[:, pad:-pad], dirs[:, pad:-pad]


def bathymetry(x, y):
    """Survey-constrained beach and wave-inferred bar/channel correction (m)."""
    X, Y = np.meshgrid(x, y, indexing="ij")
    z = 9.0 * np.maximum(1.0 - X / 560.0, 0.0) ** (2.0 / 3.0)
    z -= 0.05 * np.maximum(X - 560.0, 0.0)
    z -= 1.6 * np.exp(-0.5 * ((X - 385.0) / 22.0) ** 2)
    a, xc, wx, yc, wy, _ = CHANNEL
    z += a * np.exp(-0.5 * ((X - xc) / wx) ** 2 - 0.5 * ((Y - yc) / wy) ** 2)
    # Centimetric toe adjustment constrained by the observed shoreline crossings.
    z += 0.03 * np.exp(-0.5 * ((X - 560) / 5) ** 2 - 0.5 * ((Y - 850) / 5) ** 2)
    return z


def incident_spectrum(case, freq):
    n = len(freq)
    ef = np.asarray(case["Ef"], dtype=float)
    directions = np.broadcast_to(np.asarray(case["dirf"], dtype=float), (n,))
    spread = np.broadcast_to(np.asarray(case["sf"], dtype=float), (n,))
    if ef.shape != (n,) or not np.all(np.isfinite(ef)) or np.any(ef < 0):
        raise ValueError(f"Ef must contain {n} finite, nonnegative spectral densities")
    if (
        not np.all(np.isfinite(directions))
        or not np.all(np.isfinite(spread))
        or np.any(spread < 0)
    ):
        raise ValueError(
            "dirf and sf must be finite, with nonnegative spreading parameters"
        )
    cut, direction_bias = MODEL[:2]
    theta = np.deg2rad(np.linspace(-cut, cut, 35))
    mu = np.deg2rad(directions + direction_bias)
    difference = (theta[None, :] - mu[:, None] + np.pi) % (2 * np.pi) - np.pi
    shape = np.maximum(0.0, np.cos(difference / 2)) ** (2 * spread[:, None])
    norm = 2 * np.sqrt(np.pi) * np.exp(gammaln(spread + 0.5) - gammaln(spread + 1))
    # Full-circle normalization retains the loss of offshore-travelling energy.
    energy = ef[:, None] * 0.02 * shape / norm[:, None] * (theta[1] - theta[0])
    return energy, theta


def add_reflection(fields, details, h, freq, theta, x, y):
    """Specular return rays, aperture diffraction, and coherent standing waves."""
    R, trans, width, decay, coherence, length = MODEL[2:8]
    iw = int(np.argmin(abs(x - 520.0)))
    if h[iw, 95] <= 0.08:
        return fields
    hb = np.maximum(h[:, 95], 0.12)
    k = wavenumber(freq[None, :, None], hb[:, None, None])
    si = np.clip(k[0:1] * np.sin(theta)[None, None, :] / k, -0.98, 0.98)
    co = np.sqrt(1 - si * si)
    nn = 0.5 * (
        1
        + 2
        * k
        * hb[:, None, None]
        / np.sinh(np.minimum(2 * k * hb[:, None, None], 500))
    )
    cg = 2 * np.pi * freq[None, :, None] / k * nn
    distance = np.maximum(x[iw] - x, 0.0)
    er = R**2 * details["flux"][iw : iw + 1] / np.maximum(cg * co, 0.05)
    er *= np.exp(-2 * decay * distance)[:, None, None]
    er[x >= x[iw]] = 0.0
    travel = -cumulative_trapezoid(si / co, x, axis=0, initial=0)
    travel -= travel[iw : iw + 1]
    phase = -cumulative_trapezoid(k * co, x, axis=0, initial=0)
    phase -= phase[iw : iw + 1]
    coherent = np.minimum(1.0, coherence * np.exp(-distance / length))[
        :, None, None
    ] * np.cos(2 * phase)
    ebg = details["energy"]
    hbg = 4 * np.sqrt(np.maximum(ebg.sum((1, 2)), 1e-20))
    # The incoming field is two dimensional; reflected rays use the locally
    # straight part of the structure, away from the channel.
    for iy, yy in enumerate(y):
        aperture = ndtr((yy - 860.0 - trans * travel) / width)
        reflected = er * aperture
        mr = reflected.sum((1, 2))
        m1r = (reflected * freq[None, :, None]).sum((1, 2))
        cross = (2 * np.sqrt(np.maximum(ebg * reflected, 0)) * coherent).sum((1, 2))
        hi = fields[:, iy, 0].copy()
        m0 = hi**2 / 16
        m1 = m0 / np.maximum(fields[:, iy, 1], 1e-6)
        total = np.maximum(m0 + mr + cross * hi / np.maximum(hbg, 1e-10), 0.0)
        fields[:, iy, 0] = 4 * np.sqrt(total)
        fields[:, iy, 1] = np.divide(
            m0 + mr, m1 + m1r, out=fields[:, iy, 1].copy(), where=(m1 + m1r) > 1e-15
        )
        mx = details["mx"][:, iy] - (reflected * co).sum((1, 2))
        my = details["my"][:, iy] + (reflected * si).sum((1, 2))
        fields[:, iy, 2] = np.rad2deg(np.arctan2(my, mx))
    return fields


def predict(case, x, y, freq, bed, wall):
    for key in ("tide", "U10", "wdir"):
        if key not in case or not np.isfinite(float(case[key])):
            raise ValueError(f"{key} must be a finite number")
    if float(case["U10"]) < 0:
        raise ValueError("U10 must be nonnegative")
    tide = float(case["tide"])
    h = bed + tide
    wet = (h > 0.08) & ~wall
    energy, theta = incident_spectrum(case, freq)
    if energy.sum() < 1e-18:
        result = {
            k: np.zeros(h.shape) for k in ("Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u")
        }
        result["mwl"][:] = tide
        result["Tm"][:] = 8.0
    else:
        params = np.array([MODEL[8], MODEL[9], MODEL[10], 0.12, MODEL[11], MODEL[12]])
        fields, _, _, details = _solve_waves(
            energy,
            theta,
            h,
            freq,
            params,
            diffraction=CHANNEL[5],
            forcing=[float(case["U10"]), np.deg2rad(float(case["wdir"]))],
            wall=wall,
            return_detail=True,
        )
        # Integrate the two components of the radiation-stress gradient.
        sx, sxy = fields[:, :, 4], fields[:, :, 6]
        gy = np.gradient(sxy, y, axis=1)
        he = np.maximum(0.5 * (h[1:] + h[:-1]), 0.25)
        forcing = -(np.diff(sx, axis=0) + 10 * 0.5 * (gy[1:] + gy[:-1])) / he
        up = np.pad(np.cumsum(np.maximum(forcing, 0), axis=0), ((1, 0), (0, 0)))
        down = np.pad(np.cumsum(np.minimum(forcing, 0), axis=0), ((1, 0), (0, 0)))
        eta = SETUP[0] * up + SETUP[1] * down
        head = float(np.mean(up[43:48, 5:20] + CURRENT[1] * down[43:48, 5:20]))
        period_bar = max(float(np.mean(fields[35:41, 5:20, 1])), 1.0)
        speed = -CURRENT[0] * np.sqrt(max(head, 0.0)) * np.sqrt(7.0 / period_bar)
        xc, wx, yc, wy = JET
        X, Y = np.meshgrid(x, y, indexing="ij")
        u = speed * np.exp(-0.5 * ((X - xc) / wx) ** 2 - 0.5 * ((Y - yc) / wy) ** 2)
        fields = add_reflection(fields, details, h, freq, theta, x, y)
        # Local diffraction closure at the finite structure tip.  The unresolved
        # wall-adjacent phase gradient is constrained by the directional products.
        base_direction = fields[:, 78:79, 2].copy()
        base_height = fields[:, 78:79, 0].copy()
        if tide <= 0.08:
            shore = 560 * (1 - (max(0.08 - tide, 0) / 9) ** 1.5)
        else:
            shore = 560 + (tide - 0.08) / 0.05
        wet_length = max(0.0, min(shore, 600.0) - 520.0)
        cap, midpoint, transition, edge_width, length_scale, turn = EDGE
        smooth = expit((X - 520) / 5) * np.exp(-0.5 * ((Y - 850) / 25) ** 2)
        edge_limit = cap * (1 - np.exp(-wet_length / length_scale))
        midpoint = midpoint - turn * np.clip(base_direction, -25, 25)
        onset = expit((X - midpoint) / transition) * np.exp(
            -0.5 * ((Y - 850) / edge_width) ** 2
        )
        fields[:, :, 2] += smooth * (base_direction - fields[:, :, 2]) + onset * (
            edge_limit - base_direction
        )
        side_weight = np.clip(
            EDGE_HEIGHT[0] * np.maximum(-np.sin(np.deg2rad(base_direction)), 0), 0, 0.8
        )
        side_weight = (
            side_weight
            * expit((X - 525) / 4)
            * np.exp(-0.5 * ((Y - 850) / EDGE_HEIGHT[1]) ** 2)
        )
        fields[:, :, 0] = (
            fields[:, :, 0] * (1 - side_weight) + base_height * side_weight
        )
        # Convert the source-cell breaker fraction to the cell-center diagnostic.
        gradient = np.zeros_like(h)
        gradient[1:] = np.log(np.maximum(h[:-1], 0.12) / np.maximum(h[1:], 0.12))
        gradient = np.clip(gradient, -1.5, 1.5)
        qa, qb, qc, qd = QDIAG
        correction = (
            qa + qb * gradient + qc * fields[:, :, 3] * gradient + qd * gradient**2
        )
        qdiagnostic = np.interp(
            details["ratio"] * np.exp(np.clip(correction, -2, 2)), _R, _Q
        )
        H, T = fields[:, :, 0], fields[:, :, 1]
        k = wavenumber(1 / np.maximum(T, 1.0), np.maximum(h, 0.12))
        sig = np.tanh(k * np.maximum(h, 0.12))
        crest = 0.5 * H + np.minimum(0.3 * H, k * H**2 * (3 - sig**2) / (16 * sig**3))
        result = {
            "Hs": H,
            "Tm": T,
            "Dm": fields[:, :, 2],
            "Qb": np.clip(qdiagnostic, 0.0, 1.0),
            "mwl": tide + eta,
            "Cs": crest,
            "u": u,
        }
    for key, value in result.items():
        value = np.asarray(value, dtype=np.float32)
        if not np.all(np.isfinite(value[wet])):
            raise FloatingPointError(
                f"Nonfinite {key} on a wet cell for case {case.get('case')}"
            )
        value[~wet] = np.nan
        result[key] = value
    return result


def main(conditions_file, out_dir):
    data = Path(__file__).resolve().parent / "data"
    with np.load(data / "grid.npz") as grid:
        x, y, freq = (grid[k].copy() for k in ("x", "y", "freq"))
    if (len(x), len(y)) != (61, 101):
        raise ValueError(
            "This calibrated site model requires the supplied 61 x 101 grid"
        )
    with np.load(data / "chart.npz") as chart:
        ix = np.argmin(abs(x[:, None] - chart["x"][None, :]), axis=1)
        iy = np.argmin(abs(y[:, None] - chart["y"][None, :]), axis=1)
        wall = chart["structure"][np.ix_(ix, iy)].astype(bool)
    bed = bathymetry(x, y)
    with open(conditions_file, encoding="utf-8") as stream:
        cases = json.load(stream)
    if isinstance(cases, dict):
        cases = cases.get("cases", [cases])
    if not isinstance(cases, list):
        raise TypeError("Conditions must be a list of cases")
    destination = Path(out_dir)
    destination.mkdir(parents=True, exist_ok=True)
    seen = set()
    for case in cases:
        identifier = str(case["case"])
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", identifier) or identifier in seen:
            raise ValueError(
                "Case identifiers must be unique, simple filename components"
            )
        seen.add(identifier)
        result = predict(case, x, y, freq, bed, wall)
        np.savez_compressed(destination / f"case_{identifier}.npz", **result)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", help="JSON file containing the offshore cases")
    parser.add_argument("out_dir", help="Directory for case_<case>.npz files")
    args = parser.parse_args()
    main(args.conditions, args.out_dir)
