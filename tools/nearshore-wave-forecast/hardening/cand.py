"""Candidate pool of hard unseen conditions (same physics): low-tide storms / long-period storm swells."""
import json, numpy as np, sys
from hcommon import *
import make_data as MD
rng = np.random.default_rng(424242)
C = []; meta = []
def add(sw, ws, tide, U10, wdir, tag):
    c = MD.make_cond(sw, ws, tide, U10, wdir); c = dict(case=len(C), **c); C.append(c)
    meta.append(dict(case=c["case"], tag=tag, swell=sw, sea=ws, tide=tide, U10=U10, wdir=wdir))
# structured
for tide in (-1.0, -0.9, -0.8):
    for (H, T, d, s) in [(3.0, 12.0, 0.0, 14.0), (3.0, 14.0, 5.0, 18.0), (2.6, 16.0, 0.0, 25.0), (3.3, 13.0, -8.0, 12.0),
                         (2.8, 12.0, 10.0, 12.0)]:
        add((H, T, d, s), (0.7, 6.0, d + 10, 4.0), tide, 14.0, d + 10, "struct")
# random
for i in range(60):
    tide = float(rng.choice([-1.0, -0.95, -0.9, -0.8, -0.7]))
    H = rng.uniform(2.2, 3.4); T = rng.uniform(10.0, 16.0); d = rng.uniform(-12, 15); s = rng.uniform(8, 25)
    U10 = rng.uniform(8, 18); wd = rng.uniform(-40, 40)
    ws = (rng.uniform(0.04, 0.06) * U10, rng.uniform(5.0, 6.5), wd, 4.0)
    add((round(H, 2), round(T, 1), round(d, 1), round(s, 1)), tuple(round(v, 2) for v in ws), tide, round(U10, 1), round(wd, 1), "rand")
json.dump(C, open(H_ := "cands.json", "w"), indent=0)
json.dump(meta, open("cands_meta.json", "w"), indent=0)
print(len(C))
