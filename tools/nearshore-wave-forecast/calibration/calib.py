"""Calibration: (a) reference, (b) +-2% draws, (a')/(c) refits from fits/*.json, (d) traces, (e) shortcuts."""
import os, sys, json, glob, numpy as np
import sim, grade
from make_data import FIELDS

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "solver", "data")
UNSEEN = json.load(open(os.path.join(ROOT, "unseen_conditions.json")))
GIVEN = json.load(open(os.path.join(DATA, "given_conditions.json")))
OBS = [dict(np.load(os.path.join(DATA, "obs", f"hour_{i:02d}.npz"))) for i in range(len(GIVEN))]


def run(P, flags=None):
    return [sim.simulate(P, c, flags) for c in UNSEEN]


def summarize(preds):
    ok, rep = grade.grade_all(preds, UNSEEN)
    per = {}
    for j in rep:
        for rn in rep[j]:
            bad = [f"{f}({e / thr:.1f}x)" for f, (e, thr, o) in rep[j][rn].items() if not o]
            if bad:
                per[f"c{j}:{rn}"] = bad
    return dict(pass_=ok, worst_ratio=float(grade.margin(rep)), failures=per)


def chart_h():
    c = np.load(os.path.join(DATA, "chart.npz"))
    from scipy.interpolate import RegularGridInterpolator
    f = RegularGridInterpolator((c["x"], c["y"]), c["depth_msl"], bounds_error=False, fill_value=None)
    return f(np.stack([sim.XX.ravel(), sim.YY.ravel()], -1)).reshape(sim.XX.shape)


def shortcut_spectral():
    P = dict(sim.TRUE, h_override=chart_h(), gamma=0.73, alpha=1.0, Cf=0.038, crest_cap=0.75, R=0.0)
    return run(P, dict(current=False, reflection=False))


def offshore_feats(c):
    Ef = np.array(c["Ef"])
    m0 = Ef.sum() * sim.DF
    m1 = (Ef * sim.FREQ).sum() * sim.DF
    d = np.deg2rad(np.sum(Ef * np.array(c["dirf"])) / Ef.sum())
    return np.array([1.0, 4 * np.sqrt(m0), 16 * m0, m0 / m1, np.sin(d), c["tide"], c["tide"] * 4 * np.sqrt(m0),
                     c["U10"]])


def shortcut_regression(lam=1e-2):
    Xg = np.array([offshore_feats(c) for c in GIVEN])
    mu, sd = Xg[:, 1:].mean(0), Xg[:, 1:].std(0)
    Z = np.c_[np.ones(len(Xg)), (Xg[:, 1:] - mu) / sd]
    preds = [dict() for _ in UNSEEN]
    Zu = np.array([np.r_[1.0, (offshore_feats(c)[1:] - mu) / sd] for c in UNSEEN])
    for f in FIELDS:
        Yv = np.stack([o[f] for o in OBS])  # (H,NX,NY)
        H = Yv.shape[0]
        Yf = Yv.reshape(H, -1)
        out = np.full((len(UNSEEN), Yf.shape[1]), np.nan)
        for col in range(Yf.shape[1]):
            yv = Yf[:, col]
            m = np.isfinite(yv)
            if m.sum() < 10:
                if m.any():
                    out[:, col] = np.nanmean(yv)
                continue
            A = Z[m]
            w = np.linalg.solve(A.T @ A + lam * np.eye(A.shape[1]), A.T @ yv[m])
            out[:, col] = Zu @ w
        for j in range(len(UNSEEN)):
            preds[j][f] = out[j].reshape(sim.XX.shape)
    for j in range(len(UNSEEN)):
        preds[j]["Qb"] = np.clip(preds[j]["Qb"], 0, 1)
    return preds


def shortcut_persistence(mode="last"):
    preds = []
    for c in UNSEEN:
        if mode == "last":
            i = len(GIVEN) - 1
        else:  # most similar offshore condition
            fu = offshore_feats(c)
            i = int(np.argmin([np.sum(((offshore_feats(g) - fu)[[1, 4, 5]] / [0.3, 0.2, 0.5]) ** 2) for g in GIVEN]))
        o = OBS[i]
        p = {f: o[f].astype(float).copy() for f in FIELDS}
        p["mwl"] = p["mwl"] - GIVEN[i]["tide"] + c["tide"]
        preds.append(p)
    return preds


def fill_nan(preds):
    """graders treat NaN in a truth-wet cell as failure; for shortcuts fill with nearest valid along x."""
    for p in preds:
        for f in FIELDS:
            a = p[f]
            for jy in range(a.shape[1]):
                col = a[:, jy]
                good = np.where(np.isfinite(col))[0]
                if len(good):
                    idx = np.searchsorted(good, np.arange(len(col))).clip(0, len(good) - 1)
                    col[~np.isfinite(col)] = col[good[idx]][~np.isfinite(col)]
                else:
                    col[:] = 0.0
    return preds


def traces():
    """per part: signal in given data (truth minus truth-without-part) in noise units, locality."""
    parts = dict(channel_rip=dict(channel=False), breaking=dict(breaking=False), rip_current=dict(current=False),
                 reflection=dict(reflection=False), nonlinear=dict(nonlinear=False))
    noise = dict(Hs=None, Dm=3.0, Qb=0.015, mwl=0.015, Cs=None, u=0.04)
    out = {}
    base = [sim.simulate(sim.TRUE, c) for c in GIVEN]
    for name, fl in parts.items():
        alt = [sim.simulate(sim.TRUE, c, fl) for c in GIVEN]
        best = {}
        frac = []
        for f in noise:
            zs = []
            for b, a in zip(base, alt):
                m = b["wet"] & (sim.XX < 540) & np.isfinite(a[f])
                s = np.sqrt((0.05 * b[f][m]) ** 2 + 0.02 ** 2) if f in ("Hs", "Cs") else noise[f]
                z = np.abs(b[f][m] - a[f][m]) / s
                zs.append(z)
            zall = np.concatenate(zs)
            best[f] = dict(max_z=float(np.max(zall)), p99_z=float(np.percentile(zall, 99)),
                           frac_cells_hours_gt1sigma=float(np.mean(zall > 1)))
        # unseen effect
        un = summarize(fill_nan(run(sim.TRUE, fl)))
        out[name] = dict(given=best, unseen_worst_ratio=un["worst_ratio"], unseen_failures=len(un["failures"]))
    return out


if __name__ == "__main__":
    what = sys.argv[1:] or ["a", "b", "fits", "e", "d"]
    resf = os.path.join(ROOT, "calib_results.json")
    R = json.load(open(resf)) if os.path.exists(resf) else {}
    if "a" in what:
        R["a_reference"] = summarize(run(sim.TRUE))
        print("a", R["a_reference"])
    if "b" in what:
        rng = np.random.default_rng(11)
        keys = ["A", "xs", "bar_s", "bar_B", "bar_w", "ch_y", "ch_w", "ch_c", "gamma", "alpha", "Cf", "R",
                "kappa", "Lr", "rip_wfac", "crest_cap", "wall_x", "wall_y"]
        out = []
        for d in range(10):
            P = dict(sim.TRUE)
            for k in keys:
                P[k] = P[k] * (1 + rng.uniform(-0.02, 0.02))
            # positions: perturb by 2% of their own length scale, not of the coordinate origin
            P["xs"] = sim.TRUE["xs"] + sim.TRUE["bar_s"] * rng.uniform(-0.02, 0.02)
            P["wall_x"] = sim.TRUE["wall_x"]
            P["wall_y"] = sim.TRUE["wall_y"]
            s = summarize(run(P))
            out.append(s)
            print("b", d, s["pass_"], round(s["worst_ratio"], 2), s["failures"])
        R["b_pm2pct"] = out
    if "fits" in what:
        R["fits"] = {}
        for fn in sorted(glob.glob(os.path.join(ROOT, "fits", "*.json"))):
            d = json.load(open(fn))
            P = d["P"]
            name = os.path.basename(fn)[:-5]
            fam = d["info"]["fam"]
            import fit as F
            s = summarize(fill_nan(run(P, F.FAMILIES[fam]["flags"])))
            R["fits"][name] = dict(info=d["info"], grade=s)
            print("fit", name, "chi2_fit=%.3f heldout=%.3f all=%.3f" % (d["info"]["chi2"], d["info"]["chi2_heldout"],
                  d["info"]["chi2_all"]), s["pass_"], round(s["worst_ratio"], 2), s["failures"])
    if "fits2" in what:
        import fit2 as F2
        R["fits2"] = {}
        for fn in sorted(f for f in glob.glob(os.path.join(ROOT, "fits2", "*.json")) if not f.endswith(".graded.json")):
            d = json.load(open(fn))
            name = os.path.basename(fn)[:-5]
            P = F2.full_P(d["par"])
            s = summarize(fill_nan(run(P, F2.JOBS[name]["flags"])))
            R["fits2"][name] = dict(par=d["par"], log=d["log"], chi2_all_given=d.get("chi2_all_given"), grade=s)
            print("fit2", name, "chi2_given=%s" % d.get("chi2_all_given"), s["pass_"], round(s["worst_ratio"], 2),
                  s["failures"])
            print("     par", {k: round(v, 3) for k, v in d["par"].items()})
    if "e" in what:
        R["e"] = {}
        for name, fn in [("spectral_chart_std", shortcut_spectral), ("regression", shortcut_regression),
                         ("persistence_last", lambda: shortcut_persistence("last")),
                         ("persistence_similar", lambda: shortcut_persistence("similar"))]:
            s = summarize(fill_nan(fn()))
            R["e"][name] = s
            print("e", name, s["pass_"], round(s["worst_ratio"], 2), len(s["failures"]), list(s["failures"].items())[:6])
    if "d" in what:
        R["d_traces"] = traces()
        print(json.dumps(R["d_traces"], indent=1))
    json.dump(R, open(resf, "w"), indent=1, default=float)
