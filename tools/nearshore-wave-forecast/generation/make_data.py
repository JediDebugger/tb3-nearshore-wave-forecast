"""Generate given observations, chart, unseen conditions and their truth."""
import json, os, numpy as np
import sim

ROOT = os.path.dirname(os.path.abspath(__file__))
SOLVER = os.path.join(ROOT, "..", "solver")
rng = np.random.default_rng(20261021)


def jonswap(Hs, Tp, g=3.3):
    f = sim.FREQ
    fp = 1 / Tp
    s = np.where(f <= fp, 0.07, 0.09)
    E = f ** -5 * np.exp(-1.25 * (fp / f) ** 4) * g ** np.exp(-(f - fp) ** 2 / (2 * s ** 2 * fp ** 2))
    return E * (Hs / 4) ** 2 / (E.sum() * sim.DF)


def make_cond(sw, ws, tide, U10, wdir):
    """sw/ws = (Hs, Tp, dir, s) swell / wind sea."""
    E1 = jonswap(sw[0], sw[1], 3.3)
    E2 = jonswap(ws[0], ws[1], 2.0)
    Ef = E1 + E2
    w1 = E1 / np.maximum(Ef, 1e-12)
    # per-frequency circular mean of the two components, spreading blended
    d = np.rad2deg(np.arctan2(w1 * np.sin(np.deg2rad(sw[2])) + (1 - w1) * np.sin(np.deg2rad(ws[2])),
                              w1 * np.cos(np.deg2rad(sw[2])) + (1 - w1) * np.cos(np.deg2rad(ws[2]))))
    s = w1 * sw[3] + (1 - w1) * ws[3]
    return dict(Ef=Ef.tolist(), dirf=d.tolist(), sf=s.tolist(), tide=float(tide), U10=float(U10),
                wdir=float(wdir))


def given_conditions(n=24):
    conds, times = [], []
    t_hours = np.sort(rng.choice(np.arange(0, 10 * 24), n, replace=False)).astype(float)
    for t in t_hours:
        tide = 0.95 * np.cos(2 * np.pi * t / 12.42 + 0.3) + 0.15 * np.cos(2 * np.pi * t / 12.0 + 1.1)
        Hs = rng.uniform(0.45, 1.15)
        if tide < -0.3:
            Hs = min(Hs, 0.55 + 0.25 * rng.random())
        Tp = rng.uniform(7.0, 11.0)
        dr = rng.uniform(-28, 6)
        sp = rng.uniform(8, 20)
        U10 = rng.uniform(2, 10)
        wdir = rng.uniform(-70, 30)
        ws = (rng.uniform(0.1, 0.35) * U10 / 6, rng.uniform(3.2, 4.8), wdir, 4.0)
        conds.append(make_cond((Hs, Tp, dr, sp), ws, tide, U10, wdir))
        times.append(t)
    return conds, times


def unseen_conditions():
    return [
        make_cond((2.6, 12.0, 20.0, 12.0), (0.8, 6.0, 35.0, 4.0), -0.6, 15.0, 35.0),
        make_cond((2.0, 11.0, 12.0, 10.0), (0.7, 5.5, 25.0, 4.0), 0.7, 13.0, 25.0),
        make_cond((1.5, 14.0, 28.0, 20.0), (0.3, 4.5, 10.0, 4.0), -0.2, 6.0, 10.0),
        # hardening 2026-10-22 (see ../hardening/): low-tide storms on the bar and rip channel
        make_cond((2.25, 16.5, 0.0, 18.0), (0.6, 5.5, 15.0, 4.0), -1.0, 13.0, 15.0),   # c3 long-period storm swell
        make_cond((2.2, 12.0, 6.0, 12.0), (0.6, 5.5, 21.0, 4.0), -0.9, 13.0, 21.0),    # c4 near-normal storm
    ]


FIELDS = ["Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u"]


def observe(res, rng):
    o = {}
    wet = res["wet"]
    o["Hs"] = res["Hs"] * (1 + 0.05 * rng.standard_normal(wet.shape)) + 0.02 * rng.standard_normal(wet.shape)
    o["Tm"] = res["Tm"] + 0.25 * rng.standard_normal(wet.shape)
    o["Dm"] = res["Dm"] + 3.0 * rng.standard_normal(wet.shape)
    o["Qb"] = np.clip(res["Qb"] + 0.015 * rng.standard_normal(wet.shape), 0, 1)
    o["mwl"] = res["mwl"] + 0.015 * rng.standard_normal(wet.shape)
    o["Cs"] = res["Cs"] * (1 + 0.05 * rng.standard_normal(wet.shape))
    o["u"] = res["u"] + 0.04 * rng.standard_normal(wet.shape)
    o["v"] = 0.04 * rng.standard_normal(wet.shape)
    for k in o:
        o[k] = np.where(wet, o[k], np.nan).astype(np.float32)
    return o


def chart():
    """Old survey: Dean profile slightly off, continuous bar at the old position, no channel, 20 m grid."""
    P = dict(sim.TRUE, A=0.128, xs=556.0, bar_s=205.0, bar_B=1.3, bar_w=40.0, ch_c=0.0)
    h = sim.bathymetry(P, channel=False)
    xc, yc = sim.X[::2], sim.Y[::2]
    hc = np.round(h[::2, ::2], 1)
    land = sim.wall_mask(P)[::2, ::2]
    return xc, yc, hc, land, P


if __name__ == "__main__":
    os.makedirs(os.path.join(SOLVER, "data", "obs"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "truth"), exist_ok=True)
    conds, times = given_conditions()
    orng = np.random.default_rng(7)
    meta = []
    for i, (c, t) in enumerate(zip(conds, times)):
        r = sim.simulate(sim.TRUE, c)
        o = observe(r, orng)
        np.savez_compressed(os.path.join(SOLVER, "data", "obs", f"hour_{i:02d}.npz"), **o)
        np.savez_compressed(os.path.join(ROOT, "truth", f"given_{i:02d}.npz"),
                            **{k: r[k] for k in FIELDS}, wet=r["wet"])
        meta.append(dict(hour=i, time_h=t, **c))
        print(i, "t=%.0f tide=%.2f Hs0=%.2f Urip=%.3f Qbmax=%.3f" % (
            t, c["tide"], 4 * np.sqrt(np.sum(c["Ef"]) * sim.DF), r["Urip"],
            np.nanmax(np.where(sim.XX < 500, r["Qb"], 0))))
    json.dump(meta, open(os.path.join(SOLVER, "data", "given_conditions.json"), "w"), indent=1)
    un = unseen_conditions()
    ex = [make_cond((1.2, 10.0, 15.0, 12.0), (0.4, 5.0, 20.0, 4.0), 0.2, 8.0, 20.0)]
    json.dump([dict(case=j, **c) for j, c in enumerate(ex)],
              open(os.path.join(SOLVER, "data", "example_new_conditions.json"), "w"), indent=1)
    json.dump([dict(case=j, **c) for j, c in enumerate(un)],
              open(os.path.join(ROOT, "unseen_conditions.json"), "w"), indent=1)
    for j, c in enumerate(un):
        r = sim.simulate(sim.TRUE, c)
        np.savez_compressed(os.path.join(ROOT, "truth", f"unseen_{j}.npz"), **{k: r[k] for k in FIELDS},
                            wet=r["wet"])
        print("unseen", j, "Urip=%.3f" % r["Urip"])
    xc, yc, hc, land, _ = chart()
    np.savez_compressed(os.path.join(SOLVER, "data", "chart.npz"), x=xc, y=yc, depth_msl=hc, structure=land)
    np.savez_compressed(os.path.join(SOLVER, "data", "grid.npz"), x=sim.X, y=sim.Y, freq=sim.FREQ)
