#!/usr/bin/env python3
"""Predict stationary nearshore wave fields on the supplied product grid.

Usage: python predict.py conditions.json out_dir

A calibrated directional wave-action model, with frequency-dependent dispersion,
shoaling, lateral transport/refraction, bottom friction, wind input, whitecapping,
depth-induced breaking, radiation-stress setup, reflection and interference.
A setup-driven rip jet feeds back through the wave-current dispersion relation.
Site coefficients and effective bathymetry were fitted to the 24 supplied hours.
No training or observation files are needed at prediction time.

Dependencies: numpy and scipy. All site files are resolved relative to this file.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
from scipy.special import gammaln, expit
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import RegularGridInterpolator

G = 9.81
DATA_DIR = Path(__file__).resolve().parent / "data"
with np.load(DATA_DIR / "grid.npz") as grid:
    X = np.asarray(grid["x"], dtype=float)
    Y = np.asarray(grid["y"], dtype=float)
    F = np.asarray(grid["freq"], dtype=float)
if (len(X), len(Y), len(F)) != (61, 101, 12):
    raise ValueError("This calibration requires the supplied 61 x 101 product grid.")
XX, YY = np.meshgrid(X, Y, indexing="ij")
with np.load(DATA_DIR / "chart.npz") as chart:
    interp = RegularGridInterpolator(
        (chart["x"], chart["y"]),
        chart["structure"].astype(float),
        bounds_error=False,
        fill_value=0.0,
    )
    SOLID = interp(np.stack((XX, YY), axis=-1)) > 0.5
OUTPUT_KEYS = ("Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u")

# Frozen site calibration. These are physical closure/site parameters, not
# a nearest-neighbour lookup of training conditions or their observed fields.
CALIBRATION = {
    "lim": np.float64(68.5977071464787),
    "ss": np.float64(1.0684706003864917),
    "gain": np.float64(1.0074002933021684),
    "friction": np.float64(0.09996701265958344),
    "doff": 0.0,
    "dir_bias": np.float64(-0.7244416015211632),
    "wind": np.float64(0.5545614491681423),
    "white": 1.0,
    "alpha": np.float64(0.8766153449532832),
    "gamma": np.float64(0.6477247832140649),
    "setup": np.float64(1.2067450412684009),
    "break_power": np.float64(0.0005083928808730396),
    "qstage": np.float64(0.5749052127435795),
    "hmin": np.float64(0.1427958733686712),
    "depth": [
        10.040039321905997,
        9.851979176278434,
        9.667610002372696,
        9.487099548122242,
        9.310615561460546,
        9.138325790321073,
        8.970397982637284,
        8.80699988634265,
        8.648299249370641,
        8.494463819654714,
        8.345661345128342,
        8.20373740438498,
        8.069294842588551,
        7.940637307528245,
        7.816068446993234,
        7.693891908772698,
        7.5724113406558144,
        7.449930390431765,
        7.324752705889728,
        7.195181934818882,
        7.059521725008404,
        6.9194756501118135,
        6.777663745532219,
        6.633764351683189,
        6.487455808978288,
        6.338416457831086,
        6.186324638655151,
        6.030858691864049,
        5.8716969578713485,
        5.7085177770906155,
        5.540999489935419,
        5.375491219115319,
        5.19984507255345,
        4.985993868483739,
        4.674837051095465,
        4.224477581126541,
        3.6771756986629978,
        3.102844929952202,
        2.6612336159363754,
        2.4940591580570812,
        2.5552100434399625,
        2.8256653925347592,
        3.0676949428591964,
        3.1635767823486827,
        3.1079844829211365,
        2.961581884917179,
        2.8021410216734024,
        2.585933650617632,
        2.439194038963775,
        2.261938725450338,
        2.0802698340476753,
        1.882052476488198,
        1.6579181294489467,
        1.4048549930384506,
        1.1336622705510215,
        0.7997978609201835,
        0.31700549166454683,
        -0.24161616957267812,
        -0.7416161695726782,
        -1.2416161695726782,
        -1.7416161695726782,
    ],
    "gap": [
        1.165926946121014,
        380.6167851701095,
        380.1501309137392,
        24.69399585861961,
        36.16950536126934,
    ],
    "reflection": [
        0.35802155768406924,
        519.9483637803601,
        1.2977073036687228,
        850.5100331239292,
        16.595704743475665,
    ],
    "refraction_scale": np.float64(0.9744973850442854),
    "advection_scale": np.float64(1.0733446597504313),
    "crestscale": 1.0,
    "coupling": 2,
    "level_coupling": 1.0,
    "phase_depth": [
        10.827015206658578,
        10.63397025516313,
        10.444419966985054,
        10.258504257678544,
        10.076363042797798,
        9.89813623789701,
        9.723963758530381,
        9.5539855202521,
        9.388341438616369,
        9.227171429177382,
        9.070615407489337,
        8.920483945988158,
        8.77743053833509,
        8.639859481922302,
        8.506175074141964,
        8.374781612386249,
        8.244083394047328,
        8.112484716517367,
        7.9783898771885395,
        7.840203173453017,
        7.696328902702969,
        7.548534080249036,
        7.399387993113253,
        7.248498346254878,
        7.095472844633174,
        6.939919193207396,
        6.781445096936805,
        6.6196582607806596,
        6.45416638969822,
        6.079994982440938,
        5.716163354865584,
        5.383974966813288,
        5.08626812441262,
        4.809782206743691,
        4.504649838666812,
        4.1043436193866345,
        3.6293148141944513,
        3.1249878950878758,
        2.7207561723477984,
        2.5395575099968486,
        2.550259565875103,
        2.752067590311073,
        2.9583893739721687,
        3.0631139144689903,
        3.0495339439699536,
        2.9608000236040297,
        2.860066571346294,
        2.6910899557364925,
        2.5515893922883586,
        2.352340021831311,
        2.131461691614937,
        1.8756287683370956,
        1.5898174711082917,
        1.3376125293042773,
        1.0720778055926292,
        0.7395447007216981,
        0.25603221180904,
        -0.3115780495277726,
        -0.8791782787172661,
        -1.0073708585605947,
        -1.0533555093434661,
    ],
    "phase_h0": np.float64(8.302930445588622),
    "coherence": [
        520.2573028534056,
        213.32678164388267,
        5.999999999985477,
        0.9796968310528875,
    ],
    "coherent": True,
    "current_shape": [
        442.6308023250646,
        380.4969417797461,
        101.43179536416959,
        35.80812055551582,
    ],
    "current_drive": [2.9418832878387318, 479.92187057306904, 5.000000000000001, 0.0],
    "current_direction_response": 0.4,
    "current_refraction_scale": 1.0,
    "inundation_level": [
        -9.890039321905997,
        -9.701979176278433,
        -9.517610002372695,
        -9.337099548122241,
        -9.160615561460546,
        -8.988325790321072,
        -8.820397982637283,
        -8.65699988634265,
        -8.49829924937064,
        -8.344463819654713,
        -8.195661345128341,
        -8.053737404384979,
        -7.919294842588551,
        -7.790637307528245,
        -7.666068446993234,
        -7.543891908772697,
        -7.422411340655814,
        -7.299930390431765,
        -7.174752705889728,
        -7.045181934818881,
        -6.909521725008403,
        -6.769475650111813,
        -6.627663745532218,
        -6.483764351683188,
        -6.337455808978287,
        -6.188416457831086,
        -6.036324638655151,
        -5.880858691864049,
        -5.721696957871348,
        -5.558517777090615,
        -5.390999489935418,
        -5.225491219115319,
        -5.04984507255345,
        -4.8359938684837385,
        -4.5248370510954645,
        -4.074477581126541,
        -3.527175698662998,
        -2.952844929952202,
        -2.5112336159363755,
        -2.3440591580570813,
        -2.4052100434399626,
        -2.6756653925347593,
        -2.9176949428591965,
        -3.0135767823486828,
        -2.9579844829211366,
        -2.811581884917179,
        -2.6521410216734025,
        -2.435933650617632,
        -2.289194038963775,
        -2.111938725450338,
        -1.9,
        -1.65,
        -1.45,
        -1.18,
        -0.88,
        -0.54,
        0.11,
        0.595,
        1.08,
        1.58,
        2.08,
    ],
}


def _parameters():
    return {
        key: np.asarray(value, dtype=float) if isinstance(value, list) else value
        for key, value in CALIBRATION.items()
    }


def dispersion(h, f=F):
    h = np.maximum(np.asarray(h), 0.12)
    om = 2 * np.pi * np.asarray(f)
    d = h[..., None]
    k = np.maximum(om**2 / G, om / np.sqrt(G * d))
    for _ in range(7):
        kd = k * d
        t = np.tanh(kd)
        k -= (G * k * t - om**2) / (G * (t + kd * (1 - t * t)))
    c = om / k
    kh = k * d
    n = 0.5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 600)))
    return k, c, c * n, n


def qb_of_r(r):
    r = np.maximum(r, 1e-8)
    q = np.exp(-1 / r)
    # Fixed point becomes slow near r=1; Newton on log(q) has only one nonzero root.
    l = -1 / r
    for _ in range(12):
        ex = np.exp(l)
        fun = 1 - ex + r * l
        der = -ex + r
        step = fun / np.maximum(der, 1e-10)
        l = np.minimum(l - step, -1e-7)
    return np.where(r >= 0.99999, 1, np.exp(np.clip(l, -700, 0)))


def spectrum(conds, lim=75, scale_s=1, mix=False, ntheta=41, dir_bias=0):
    ef = np.array([c["Ef"] for c in conds])
    sf = np.array([c["sf"] for c in conds])
    df = np.deg2rad(np.array([c["dirf"] for c in conds]) + dir_bias)
    th = np.linspace(-lim, lim, ntheta) * np.pi / 180

    def d(s, mu):
        s = s * scale_s
        delta = (th - mu[..., None] + np.pi) % (2 * np.pi) - np.pi
        return (
            np.maximum(np.cos(delta / 2), 0) ** (2 * s[..., None])
            * np.exp(gammaln(s + 1) - gammaln(s + 0.5))[..., None]
            / (2 * np.sqrt(np.pi))
            * (th[1] - th[0])
        )

    if mix:
        ss = sf.max(1)
        mus = df[np.arange(len(sf)), sf.argmax(1)]
        w = (ss[:, None] - sf) / np.maximum(ss[:, None] - 4, 1e-6)
        shape = (1 - w[..., None]) * d(ss, mus)[:, None, :] + w[..., None] * d(
            ss * 0 + 4, np.deg2rad([c["wdir"] for c in conds])
        )[:, None, :]
    else:
        shape = d(sf, df)
    return 0.02 * ef[..., None] * shape, th


def current_state(kbase, ky, h, U):
    om = 2 * np.pi * F[:, None, None]
    hh = h[None, None, :]
    uu = U[None, None, :]
    K = np.maximum(kbase + np.zeros_like(ky), np.abs(ky) * 1.001)
    for _ in range(7):
        si = np.clip(ky / K, -0.9848, 0.9848)
        co = np.sqrt(1 - si * si)
        kh = K * hh
        t = np.tanh(kh)
        sig = np.sqrt(G * K * t)
        n = 0.5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 600)))
        cg = n * sig / K
        fun = sig + uu * K * co - om
        der = np.maximum(cg + uu / co, 0.08)
        step = np.clip(fun / der, -0.22 * K, 0.22 * K)
        K = np.maximum(K - step, np.abs(ky) * 1.001)
    si = np.clip(ky / K, -0.9848, 0.9848)
    co = np.sqrt(1 - si * si)
    kh = K * hh
    t = np.tanh(kh)
    sig = np.sqrt(G * K * t)
    n = 0.5 * (1 + 2 * kh / np.sinh(np.minimum(2 * kh, 600)))
    cg = n * sig / K
    vel = cg * co + uu
    blocked = (vel < 0.08) | (np.abs(sig + uu * K * co - om) > 0.04 * om)
    vel = np.maximum(vel, 0.08)
    action_vel = vel * om / np.maximum(sig, 0.02)
    alpha = K * (1 - t * t) / np.maximum(t + kh * (1 - t * t), 1e-10)
    return K, si, co, cg, vel, action_vel, alpha, blocked, sig


def interpolate_last(a, at):
    lo = np.clip(np.floor(at).astype(int), 0, a.shape[-1] - 2)
    w = np.clip(at - lo, 0, 1)
    return (
        np.take_along_axis(a, lo, axis=-1) * (1 - w)
        + np.take_along_axis(a, lo + 1, axis=-1) * w
    )


def _solve(
    cond,
    p=None,
    details=False,
    ntheta=41,
    nsub=6,
    background=None,
    iteration=0,
    level=None,
    current=None,
):
    if p is None:
        p = _parameters()
    amp, xc, yc, wx, wy = p["gap"]
    xx = X[:, None]
    yy = Y[None, :]
    depth = p["depth"][:, None] + amp * np.exp(
        -0.5 * ((xx - xc) / wx) ** 2 - 0.5 * ((yy - yc) / wy) ** 2
    )
    h = np.maximum(
        depth
        + cond["tide"]
        + (0 if level is None else p.get("level_coupling", 0) * level),
        p.get("hmin", 0.12),
    )
    k, c, cg, n = dispersion(h)
    # Layout (x,frequency,direction,y); physical fields have a singleton direction.
    k = k.transpose(0, 2, 1)[:, :, None, :]
    cg = cg.transpose(0, 2, 1)[:, :, None, :]
    n = n.transpose(0, 2, 1)[:, :, None, :]
    kh = k * h[:, None, None, :]
    tt = np.tanh(kh)
    alpha = k * (1 - tt * tt) / np.maximum(tt + kh * (1 - tt * tt), 1e-10)
    dhdy = np.gradient(depth, 10, axis=1)
    if current is not None:
        du_dy = np.gradient(current, 10, axis=1)
    E0, theta = spectrum([cond], p["lim"], p["ss"], False, ntheta, p.get("dir_bias", 0))
    E0 = E0[0, :, :, None] * p["gain"] ** 2
    ky = np.broadcast_to(
        k[0] * np.sin(theta)[None, :, None], (len(F), ntheta, len(Y))
    ).copy()
    cost = np.sqrt(np.maximum(1 - (ky / k[0]) ** 2, 0.001))
    vel = cg[0] * cost
    flux = np.broadcast_to(E0 * vel, (len(F), ntheta, len(Y))).copy()
    sh = (61, 12, ntheta, 101)
    ES = np.zeros(sh)
    SIS = np.zeros(sh)
    COS = np.zeros(sh)
    VS = np.zeros(sh)
    qs = np.zeros((61, 101))
    eta = np.zeros((61, 101))
    stress = np.zeros((61, 101))
    disss = np.zeros((61, 101))
    for ix in range(61):
        if ix:
            for sub in range(nsub):
                w = sub / nsub
                km = k[ix - 1] * (1 - w) + k[ix] * w
                am = alpha[ix - 1] * (1 - w) + alpha[ix] * w
                dy = dhdy[ix - 1] * (1 - w) + dhdy[ix] * w
                si = np.clip(ky / km, -0.9848, 0.9848)
                co = np.sqrt(1 - si * si)
                tan = si / co
                ds = 10 / nsub
                force = km * am * dy[None, None, :] / co
                if current is not None:
                    us = current[ix - 1] * (1 - w) + current[ix] * w
                    hs = h[ix - 1] * (1 - w) + h[ix] * w
                    km, si, co, cgs, vgs, vas, am, blocked, sig = current_state(
                        km, ky, hs, us
                    )
                    uys = du_dy[ix - 1] * (1 - w) + du_dy[ix] * w
                    tan = np.clip(cgs * si / vgs, -5.5, 5.5)
                    force = (
                        km * cgs * am * dy[None, None, :]
                        + p.get("current_refraction_scale", 1.0)
                        * km
                        * co
                        * uys[None, None, :]
                    ) / vgs
                    flux[blocked] = 0
                tan *= p["advection_scale"]
                # Conservative upwind transport of normal action flux.
                vf = 0.5 * (tan[:, :, 1:] + tan[:, :, :-1])
                jf = vf * np.where(vf > 0, flux[:, :, :-1], flux[:, :, 1:])
                ff = np.concatenate(
                    [
                        tan[:, :, :1] * flux[:, :, :1],
                        jf,
                        tan[:, :, -1:] * flux[:, :, -1:],
                    ],
                    axis=2,
                )
                jky = jf * np.where(vf > 0, ky[:, :, :-1], ky[:, :, 1:])
                ffky = np.concatenate(
                    [
                        tan[:, :, :1] * flux[:, :, :1] * ky[:, :, :1],
                        jky,
                        tan[:, :, -1:] * flux[:, :, -1:] * ky[:, :, -1:],
                    ],
                    axis=2,
                )
                momentum = flux * ky - ds / 10 * np.diff(ffky, axis=2)
                flux = np.maximum(flux - ds / 10 * np.diff(ff, axis=2), 0)
                if ix > 0:
                    flux[:, :, SOLID[ix - 1]] = 0
                # An impermeable cell carries neither action nor wavevector
                # momentum. Dividing residual momentum by a vanishing flux
                # would create spurious grazing rays beside the structure.
                ky = np.divide(
                    momentum,
                    flux,
                    out=np.zeros_like(momentum),
                    where=flux > 1e-24,
                )
                ky -= ds * force * p["refraction_scale"]
            # The x component adjusts according to the local dispersion relation.
        si = np.clip(ky / k[ix], -0.9848, 0.9848)
        co = np.sqrt(1 - si * si)
        vel = cg[ix] * co
        local_k = k[ix]
        ground_vel = vel
        if current is not None:
            local_k, si, co, cgl, ground_vel, vel, al, blocked, sig = current_state(
                k[ix], ky, h[ix], current[ix]
            )
            flux[blocked] = 0
        en = flux / np.maximum(vel, 0.1)
        bk = 0 if background is None else background[ix]
        total_break = en + bk
        m0 = total_break.sum((0, 1))
        fm = np.sum(total_break * F[:, None, None], axis=(0, 1)) / np.maximum(m0, 1e-12)
        fm = np.maximum(fm, 0.03)
        Hmax = p["gamma"] * h[ix]
        qb = qb_of_r(8 * m0 / Hmax**2)
        if ix:
            bfr = (
                p["friction"]
                * (2 * np.pi * F[:, None, None]) ** 2
                / G**2
                / np.sinh(np.minimum(local_k * h[ix, None, None, :], 60)) ** 2
            )
            diss = 0.25 * p["alpha"] * qb * fm * Hmax**2 / np.maximum(m0, 1e-12)
            align = co * np.cos(np.deg2rad(cond["wdir"])) + si * np.sin(
                np.deg2rad(cond["wdir"])
            )
            cphase = (
                2 * np.pi * F[:, None, None] if current is None else sig
            ) / local_k
            beta = (
                0.0003
                * (2 * np.pi * F[:, None, None])
                * np.maximum(cond["U10"] * align / cphase - 1, 0)
                * p.get("wind", 0)
            )
            kbar = np.maximum(
                np.sum(total_break * local_k, axis=(0, 1)) / np.maximum(m0, 1e-12), 1e-4
            )
            steep = kbar * kbar * m0
            white = (
                2.36e-5 * (2 * np.pi * fm) * (steep / 0.00302) ** 2 * p.get("white", 0)
            )
            rate = (
                bfr
                + diss[None, None, :]
                * (F[:, None, None] / fm[None, None, :]) ** p["break_power"]
                - beta
                + white[None, None, :] * local_k / kbar[None, None, :]
            )
            flux *= np.exp(
                -np.clip(rate * 10 / np.maximum(ground_vel, 0.1), -1.0, 20.0)
            )
            en = flux / np.maximum(vel, 0.1)
            disss[ix] = diss
        m0 = (en + bk).sum((0, 1))
        qs[ix] = (1 - p["qstage"]) * qb_of_r(8 * m0 / Hmax**2) + p["qstage"] * qb
        stress[ix] = np.sum(en * (n[ix] * (1 + co**2) - 0.5), axis=(0, 1))
        if ix:
            eta[ix] = (
                eta[ix - 1]
                - (stress[ix] - stress[ix - 1])
                / np.maximum((h[ix] + h[ix - 1]) / 2, 0.3)
                * p["setup"]
            )
        ES[ix] = en
        SIS[ix] = si
        COS[ix] = co
        VS[ix] = vel
        flux[:, :, SOLID[ix]] = 0
    # Reflected component: reverse the normal propagation at the structure face.
    r, xwall, bback, ywall, blur = p["reflection"]
    travel = cumulative_trapezoid(SIS / COS, X, axis=0, initial=0)
    bfr = (
        p["friction"]
        * (2 * np.pi * F[None, :, None, None]) ** 2
        / G**2
        / np.sinh(np.minimum(k * h[:, None, None, :], 60)) ** 2
    )
    loss = cumulative_trapezoid(bfr / VS, X, axis=0, initial=0)
    i = int(np.clip(xwall // 10, 0, 59))
    w = xwall / 10 - i

    def at(a):
        return a[i] * (1 - w) + a[i + 1] * w

    tw = at(travel)
    lw = at(loss)
    fw = at(ES * VS)
    EREF = np.zeros(sh)
    for ix in range(i + 1):
        dist = tw - travel[ix]
        face_y = Y[None, None, :] - dist
        face_flux = interpolate_last(fw, face_y / 10)
        vis = expit((face_y - ywall) / blur)
        EREF[ix] = (
            r * face_flux / VS[ix] * np.exp(-bback * np.maximum(lw - loss[ix], 0)) * vis
        )
    if iteration < p.get("coupling", 0):
        return _solve(
            cond,
            p,
            details,
            ntheta,
            nsub,
            EREF,
            iteration + 1,
            eta if level is None else 0.5 * (eta + level),
            current,
        )
    total = ES + EREF
    M = total.sum((1, 2))
    Hs = 4 * np.sqrt(M)
    Tm = M / np.maximum(np.sum(total * F[None, :, None, None], axis=(1, 2)), 1e-12)
    Dm = (
        np.rad2deg(
            np.arctan2(
                np.sum(total * SIS, axis=(1, 2)), np.sum((ES - EREF) * COS, axis=(1, 2))
            )
        )
        + p["doff"]
    )
    if p.get("coherent", False):
        wall, coh_length, coh_power, coh_amp = p["coherence"]
        ph = np.maximum(p["phase_depth"] + cond["tide"], 0.15)
        pk = dispersion(ph)[0]
        k0 = dispersion(np.maximum(p["phase_h0"] + cond["tide"], 0.15))[0]
        pky = k0[:, None] * np.sin(theta)[None, :]
        pkx = np.sqrt(np.maximum(pk[:, :, None] ** 2 - pky[None, :, :] ** 2, 1e-5))
        pint = cumulative_trapezoid(pkx, X, axis=0, initial=0)
        wi = int(wall // 10)
        ww = wall / 10 - wi
        atwall = pint[wi] * (1 - ww) + pint[wi + 1] * ww
        phase = 2 * (atwall[None, :, :] - pint)
        envelope = (
            np.exp(-((np.maximum(wall - X, 0) / coh_length) ** coh_power)) * coh_amp
        )
        gate = np.interp(
            Y,
            [0, 810, 820, 830, 840, 850, 860, 1000],
            [0, 0, 0.07, 0.14, 0.22, 0.39, 1, 1],
        )
        coherent = (
            2
            * np.sum(
                np.sqrt(np.maximum(ES * EREF, 0)) * np.cos(phase[:, :, :, None]),
                axis=(1, 2),
            )
            * envelope[:, None]
            * gate[None, :]
        )
        Hs = 4 * np.sqrt(np.maximum(M + coherent, 0.04 * M))

    om = 2 * np.pi / np.maximum(Tm, 1)
    km = om / np.sqrt(G * h)
    for _ in range(8):
        t = np.tanh(km * h)
        km -= (G * km * t - om**2) / (G * (t + km * h * (1 - t * t)))
    t = np.tanh(km * h)
    Cs = Hs * (
        0.5 + np.minimum(0.3, p["crestscale"] * km * Hs * (3 - t * t) / (16 * t**3))
    )
    out = dict(
        Hs=Hs, Tm=Tm, Dm=Dm, Qb=qs, mwl=eta + cond["tide"], Cs=Cs, u=np.zeros_like(Hs)
    )
    if details:
        out.update(
            E=ES,
            Eref=EREF,
            sint=SIS,
            cost=COS,
            h=h,
            depth=depth,
            eta=eta,
            stress=stress,
            diss=disss,
        )
    return out


def _validate_condition(condition):
    """Check the input contract without silently changing a supplied spectrum."""
    if not isinstance(condition, dict):
        raise ValueError("Each case must be a JSON object.")
    for key in ("case", "Ef", "dirf", "sf", "tide", "U10", "wdir"):
        if key not in condition:
            raise ValueError(f"Missing required condition field: {key}")
    c = dict(condition)
    for key in ("Ef", "dirf", "sf"):
        a = np.asarray(c[key], dtype=float)
        if a.shape != F.shape or not np.all(np.isfinite(a)):
            raise ValueError(f"{key} must contain {len(F)} finite numbers.")
        c[key] = a
    if np.any(c["Ef"] < 0) or np.any(c["sf"] < 0):
        raise ValueError(
            "Spectral energy and spreading parameters must be nonnegative."
        )
    for key in ("tide", "U10", "wdir"):
        c[key] = float(c[key])
        if not np.isfinite(c[key]):
            raise ValueError(f"{key} must be finite.")
    if c["U10"] < 0:
        raise ValueError("Wind speed must be nonnegative.")
    c["dirf"] = (c["dirf"] + 180.0) % 360.0 - 180.0
    c["wdir"] = (c["wdir"] + 180.0) % 360.0 - 180.0
    name = str(c["case"])
    if not name or name in (".", "..") or any(ch in name for ch in ("/", "\\", "\x00")):
        raise ValueError("A case identifier must be a valid filename component.")
    return c


def predict_case(condition):
    """Return the seven fields, with a common physical wet/dry mask."""
    c = _validate_condition(condition)
    p = _parameters()
    base = _solve(c, p, ntheta=41, nsub=6)

    # The pressure head behind the bar drives an opposing jet through its gap.
    coefficient, reference_x, width_x, offset = p["current_drive"]
    weight = np.exp(-0.5 * ((X - reference_x) / width_x) ** 2)
    weight[(X < 390) | (X > 520)] = 0
    weight /= weight.sum()
    eta = base["mwl"] - c["tide"]
    head = float(weight @ (eta[:, :18].mean(axis=1) - eta[:, 35:42].mean(axis=1)))
    amplitude = coefficient * np.sqrt(max(head + offset, 0.0))
    xc, yc, wx, wy = p["current_shape"]
    jet = -amplitude * np.exp(
        -0.5 * ((XX - xc) / wx) ** 2 - 0.5 * ((YY - yc) / wy) ** 2
    )

    fields = base
    if amplitude > 0.015:
        fields = _solve(c, p, ntheta=41, nsub=8, current=jet)
        # The measured mean direction resolves less of the narrow surface-jet
        # shear than the action flux does. Retain the calibrated moment response.
        direction_change = (fields["Dm"] - base["Dm"] + 180.0) % 360.0 - 180.0
        fields["Dm"] = base["Dm"] + p["current_direction_response"] * direction_change
    fields["u"] = jet

    # Check the solution before deriving its shoreline, so a numerical failure
    # can never be mistaken for a dry cell.
    for key in OUTPUT_KEYS:
        if not np.all(np.isfinite(fields[key])):
            raise FloatingPointError(f"Nonfinite numerical solution for {key}.")

    # The inundation contour is constrained by the chart and the observed tide-
    # dependent shoreline. Wave setup can inundate additional beach cells.
    wet = (fields["mwl"] > p["inundation_level"][:, None]) & ~SOLID
    energy = float(np.sum(c["Ef"]))
    boundary_period = (
        energy / float(np.sum(c["Ef"] * F)) if energy > 0 else 1.0 / F.mean()
    )
    fields["Tm"] = np.where(fields["Hs"] > 1e-8, fields["Tm"], boundary_period)
    fields["Tm"] = np.clip(fields["Tm"], 1.0 / F.max(), 1.0 / F.min())
    fields["Dm"] = (fields["Dm"] + 180.0) % 360.0 - 180.0
    fields["Qb"] = np.clip(fields["Qb"], 0.0, 1.0)
    for key in ("Hs", "Cs"):
        fields[key] = np.maximum(fields[key], 0.0)
    result = {}
    for key in OUTPUT_KEYS:
        a = np.asarray(fields[key], dtype=float)
        if a.shape != (len(X), len(Y)) or not np.all(np.isfinite(a[wet])):
            raise FloatingPointError(f"Nonfinite prediction in wet cells for {key}.")
        result[key] = np.where(wet, a, np.nan).astype(np.float32)
    return result


def main(condition_file, out_dir):
    with open(condition_file, encoding="utf-8") as stream:
        cases = json.load(stream)
    if isinstance(cases, dict) and "cases" in cases:
        cases = cases["cases"]
    if not isinstance(cases, list):
        raise ValueError("The conditions file must contain a list of cases.")
    output = Path(out_dir)
    output.mkdir(parents=True, exist_ok=True)
    for case in cases:
        c = _validate_condition(case)
        fields = predict_case(c)
        filename = f"case_{c['case']}.npz"
        temporary = output / ("." + filename)
        np.savez_compressed(temporary, **fields)
        os.replace(temporary, output / filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("conditions", help="JSON file in the example input format")
    parser.add_argument("out_dir", help="directory for case_<case>.npz files")
    args = parser.parse_args()
    main(args.conditions, args.out_dir)
