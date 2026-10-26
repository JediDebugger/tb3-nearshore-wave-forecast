"""Focused refinement around the best fair candidates: low-tide near-normal storms, long and moderate period."""
import json
from hcommon import *
import make_data as MD
C = []; meta = []
for fam, Hs_l, Tp_l, s in (("long", (2.25, 2.4), (15.0, 16.5), 18.0), ("mid", (2.2, 2.35), (11.0, 12.0), 12.0)):
    for tide in (-1.0, -0.9):
        for H in Hs_l:
            for T in Tp_l:
                for d in (-4.0, 0.0, 4.0) if fam == "long" else (-6.0, 0.0, 6.0):
                    sw = (H, T, d, s); ws = (0.6, 5.5, d + 15, 4.0)
                    c = dict(case=len(C), **MD.make_cond(sw, ws, tide, 13.0, d + 15)); C.append(c)
                    meta.append(dict(case=c["case"], tag=fam, swell=sw, sea=ws, tide=tide, U10=13.0, wdir=d + 15))
json.dump(C, open("cands2.json", "w"), indent=0); json.dump(meta, open("cands2_meta.json", "w"), indent=0)
print(len(C))
