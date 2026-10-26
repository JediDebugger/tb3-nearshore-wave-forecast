"""Append the hardening conditions to hidden/unseen_conditions.json and write their truth (TRUE params).
c0-c2 are kept unchanged (checked bit-for-bit against unseen_conditions_v1.json)."""
import json, os, numpy as np
from hcommon import *
import make_data as MD
V1 = json.load(open(K + "/hidden/unseen_conditions_v1.json"))
un = MD.unseen_conditions()
assert len(un) >= 3
for j in range(3):
    for key in ("Ef", "dirf", "sf", "tide", "U10", "wdir"):
        assert np.allclose(un[j][key], V1[j][key], rtol=0, atol=0), (j, key)
out = [dict(case=j, **c) for j, c in enumerate(un)]
json.dump(out, open(K + "/hidden/unseen_conditions.json", "w"), indent=1)
for j, c in enumerate(un):
    f = K + f"/hidden/truth/unseen_{j}.npz"
    r = sim.simulate(sim.TRUE, c)
    if j < 3:   # existing truth must reproduce
        t = np.load(f)
        assert all(np.array_equal(t[k], r[k], equal_nan=True) for k in MD.FIELDS), j
        continue
    np.savez_compressed(f, **{k: r[k] for k in MD.FIELDS}, wet=r["wet"])
    print("unseen", j, "Urip=%.3f" % r["Urip"])
print(len(out), "conditions")
