"""Fairness margins on the hardened set, strict: refit predictions exactly as a predict.py would write them
(NaN where the refit's own wet mask is dry; no calibration-side NaN filling), graded by hidden/grade.py."""
import json, sys
from hcommon import *
import fit2 as F2
sys.path.insert(0, K + "/audit")
import sim_alt as S
from variants import pm2_list
U = json.load(open(K + "/hidden/unseen_conditions.json"))
T = [grade.load_truth(j) for j in range(len(U))]
models = []
for n in ("full_s0", "full_s1", "full_s2", "full_s3"):
    models.append((n, F2.full_P(json.load(open(f"{K}/hidden/fits2/{n}.json"))["par"]), sim.simulate))
for form in ("gd", "miche"):
    p = json.load(open(f"{K}/audit/fit_{form}.json"))
    models.append((f"alt_{form}", dict(S.TRUE, gamma=p["gamma"], alpha=p["alpha"], hmax_form=form), S.simulate))
models += [(n, P, f) for n, P, fl, f in pm2_list()]
out = {}
for name, P, f in models:
    row = []
    for j, c in enumerate(U):
        fe = grade.features(f(P, c), T[j], c["tide"])
        row.append((worst(fe), top(fe, 2)))
    out[name] = row
    print(f"{name:10s} worst {max(r[0] for r in row):.2f} | " + " | ".join(f"c{j} {w:.2f}" for j, (w, _) in enumerate(row))
          + f" | top c3: {row[3][1]}; c4: {row[4][1]}", flush=True)
json.dump(out, open(H + "/fairness_strict.json", "w"), indent=1)
