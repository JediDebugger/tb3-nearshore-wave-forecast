import json, glob, numpy as np, sys
from hcommon import *
import grade as G
RB = {("R3_rip", "u"): 0.15}
def thr(rn, f):
    if (rn, f) in RB: return RB[(rn, f)]
    if rn == "R5_seawall" and f in ("Hs", "Cs"): return 0.08
    if f in G.OVERALL and rn == "overall": return G.OVERALL[f]
    return G.BARS[f][1]
def wr(fe):
    return max(fe[rn][f] / thr(rn, f) for rn in fe for f in fe[rn])
def topl(fe, n=3):
    it = sorted(((fe[rn][f] / thr(rn, f), rn, f) for rn in fe for f in fe[rn]), reverse=True)[:n]
    return " ".join(f"{rn[3:8]}:{f}={r:.2f}" for r, rn, f in it)
def load_var(d):
    V = {}
    for fn in glob.glob(f"{d}/var_*.json"):
        for name, dd in json.load(open(fn)).items():
            V.setdefault(name, {}).update({int(k): v for k, v in dd.items()})
    return V
def sub_fe(d, tag, k, conds):
    p = dict(np.load(f"{d}/{tag}/case_{k}.npz")); t = np.load(f"cand_var/truth/cand_{k}.npz"); t = {f: t[f] for f in t.files}
    fe = G.features(p, t, conds[k]["tide"])
    return {rn: {f: float(v[0]) for f, v in fe[rn].items()} for rn in fe}
