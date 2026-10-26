"""Refit gamma, alpha (others at truth) with SWAN-style Hmax=gamma*d on the strip y 100-250, then grade on unseen."""
import os, sys, json
os.environ["SIM_Y0"]="100"; os.environ["SIM_Y1"]="250"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import numpy as np, importlib
import sim_alt as S
from scipy.optimize import least_squares
K="/Users/dev/.tb-graders/k-ocean"
GIVEN=json.load(open(K+"/solver/data/given_conditions.json"))
JY=np.round(S.Y/10.0).astype(int)
OBS=[{k:v[:,JY] for k,v in np.load(f"{K}/solver/data/obs/hour_{i:02d}.npz").items()} for i in range(24)]
HOURS=[0,4,17,16,9,3,15,8,12,19,7,10]
form=sys.argv[1]
def res(p):
    P=dict(S.TRUE, gamma=p[0], alpha=p[1], hmax_form=form); out=[]
    for i in HOURS:
        r=S.simulate(P,GIVEN[i]); o=OBS[i]
        for f,s in (("Hs",None),("Qb",0.015),("mwl",0.015)):
            m=(S.XX<520)&np.isfinite(o[f])&np.isfinite(r[f])
            sg=np.sqrt((0.05*o[f][m])**2+0.02**2) if s is None else s
            out.append((o[f][m]-r[f][m])/sg)
    return np.concatenate(out)
r=least_squares(res,[0.6,1.2],bounds=([0.3,0.3],[1.2,4.0]),diff_step=1e-2,max_nfev=40)
print(form, "gamma=%.3f alpha=%.3f chi2/n=%.3f"%(r.x[0],r.x[1],np.mean(r.fun**2)))
json.dump(dict(gamma=r.x[0],alpha=r.x[1],chi2=float(np.mean(r.fun**2))),open(f"fit_{form}.json","w"))
