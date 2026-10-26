import os, sys, json, numpy as np
K = "/Users/dev/.tb-graders/k-ocean"
sys.path.insert(0, K + "/hidden")
import sim, grade
H = K + "/hardening"
def load_preds(root, n):
    return [dict(np.load(f"{root}/case_{k}.npz")) for k in range(n)]
def grade_one(pred, truth, tide, rbars=None):
    """features with optional region-specific bar overrides {(region, feat): thr}"""
    fe = grade.features(pred, truth, tide)
    if rbars:
        for (rn, f), thr in rbars.items():
            if rn in fe and f in fe[rn]:
                e = fe[rn][f][0]; fe[rn][f] = (e, thr, e <= thr)
    return fe
def worst(fe):
    return max(fe[rn][f][0] / fe[rn][f][1] for rn in fe for f in fe[rn])
def fails(fe, tag=""):
    return [f"{tag}{rn}:{f}={e:.3g}>{thr}" for rn in fe for f, (e, thr, ok) in fe[rn].items() if not ok]
def top(fe, n=4):
    items = sorted(((fe[rn][f][0] / fe[rn][f][1], rn, f) for rn in fe for f in fe[rn]), reverse=True)[:n]
    return ", ".join(f"{rn[:6]}:{f} {r:.2f}" for r, rn, f in items)
