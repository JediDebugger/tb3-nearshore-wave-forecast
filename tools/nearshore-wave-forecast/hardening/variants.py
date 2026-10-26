"""Truth + model-variant errors on a conditions file.
usage: python variants.py conds.json outdir group [k0 k1]
groups: truth | fits (fits2 staged refits, incl. omissions) | pm2 (10 +-2% draws) | alt (gd/miche breaking-law refits)"""
import sys, json, glob, os, numpy as np
from hcommon import *
import fit2 as F2
sys.path.insert(0, K + "/audit")
VARS = {}
def pm2_list():
    rng = np.random.default_rng(11)
    keys = ["A", "xs", "bar_s", "bar_B", "bar_w", "ch_y", "ch_w", "ch_c", "gamma", "alpha", "Cf", "R",
            "kappa", "Lr", "rip_wfac", "crest_cap", "wall_x", "wall_y"]
    out = []
    for d in range(10):
        P = dict(sim.TRUE)
        for k in keys: P[k] = P[k] * (1 + rng.uniform(-0.02, 0.02))
        P["xs"] = sim.TRUE["xs"] + sim.TRUE["bar_s"] * rng.uniform(-0.02, 0.02)
        P["wall_x"] = sim.TRUE["wall_x"]; P["wall_y"] = sim.TRUE["wall_y"]
        out.append((f"pm2_{d}", P, {}, sim.simulate))
    return out
def fits_list():
    out = []
    for fn in sorted(glob.glob(K + "/hidden/fits2/*.json")):
        name = os.path.basename(fn)[:-5]
        if name.endswith(".graded"): continue
        d = json.load(open(fn)); out.append((name, F2.full_P(d["par"]), F2.JOBS[name]["flags"], sim.simulate))
    return out
def alt_list():
    import sim_alt as S
    out = []
    for form in ("gd", "miche"):
        p = json.load(open(f"{K}/audit/fit_{form}.json"))
        out.append((f"alt_{form}", dict(S.TRUE, gamma=p["gamma"], alpha=p["alpha"], hmax_form=form), {}, S.simulate))
    return out
def fill_nan(p):
    for f in ("Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u"):
        a = np.array(p[f], float)
        for jy in range(a.shape[1]):
            col = a[:, jy]; good = np.where(np.isfinite(col))[0]
            if len(good):
                idx = np.searchsorted(good, np.arange(len(col))).clip(0, len(good) - 1)
                col[~np.isfinite(col)] = col[good[idx]][~np.isfinite(col)]
            else: col[:] = 0.0
        p[f] = a
    return p
if __name__ == "__main__":
    conds = json.load(open(sys.argv[1])); outd = sys.argv[2]; grp = sys.argv[3]
    k0, k1 = (int(sys.argv[4]), int(sys.argv[5])) if len(sys.argv) > 5 else (0, len(conds))
    os.makedirs(outd, exist_ok=True)
    if grp == "truth":
        os.makedirs(f"{outd}/truth", exist_ok=True)
        for k in range(k0, k1):
            r = sim.simulate(sim.TRUE, conds[k])
            np.savez_compressed(f"{outd}/truth/cand_{k}.npz", **{f: r[f] for f in ("Hs", "Tm", "Dm", "Qb", "mwl", "Cs", "u")},
                                wet=r["wet"], Urip=r["Urip"])
        sys.exit()
    lst = dict(fits=fits_list, pm2=pm2_list, alt=alt_list)[grp]()
    res = {}
    for k in range(k0, k1):
        t = np.load(f"{outd}/truth/cand_{k}.npz"); t = {f: t[f] for f in t.files}
        for name, P, fl, simf in lst:
            r = fill_nan(simf(P, conds[k], fl))
            fe = grade.features(r, t, conds[k]["tide"])
            res.setdefault(name, {})[k] = {rn: {f: float(v[0]) for f, v in fe[rn].items()} for rn in fe}
        print(k, flush=True)
    json.dump(res, open(f"{outd}/var_{grp}_{k0}_{k1}.json", "w"))
