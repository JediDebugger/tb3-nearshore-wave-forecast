"""Staged truth-free refits (what a coastal engineer would do), each stage on an alongshore strip.
stage1  profile: Dean A, shoreline xs, bar (s,B,w), breaking gamma/alpha, friction Cf, crest cap  (strip y 100-250)
stage2  channel + rip: ch_y, ch_w, ch_c, kappa, Lr                                            (strip y 250-520)
stage3  seawall reflection R                                                                  (strip y 700-1000)
stage0  given-data check: chi2 on all 24 hours, full domain
usage: SIM_Y0=.. SIM_Y1=.. python fit2.py <job> <stage>   (driver: run_fit2.sh)"""
import os, sys, json, time, numpy as np
from multiprocessing import Pool
from scipy.optimize import least_squares
import sim

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA = os.path.join(ROOT, "..", "solver", "data")
GIVEN = json.load(open(os.path.join(DATA, "given_conditions.json")))
JY = np.round(sim.Y / 10.0).astype(int)
OBS = [{k: v[:, JY] for k, v in np.load(os.path.join(DATA, "obs", f"hour_{i:02d}.npz")).items()}
       for i in range(len(GIVEN))]
SIG = dict(Hs=None, Tm=0.25, Dm=3.0, Qb=0.015, mwl=0.015, Cs=None, u=0.04)
MASK = sim.XX < 520
BOUNDS = dict(A=(0.10, 0.16), xs=(540, 575), bar_s=(120, 260), bar_B=(0.3, 2.5), bar_w=(15, 70),
              ch_y=(260, 500), ch_w=(12, 80), ch_c=(0.0, 1.0), gamma=(0.4, 1.0), alpha=(0.3, 3.0),
              Cf=(0.0, 0.2), R=(0.0, 0.95), kappa=(0.0, 2.0), Lr=(10, 200), crest_cap=(0.55, 0.95))
STAGE_PARAMS = {1: ["A", "xs", "bar_s", "bar_B", "bar_w", "gamma", "alpha", "Cf", "crest_cap"],
                2: ["ch_y", "ch_w", "ch_c", "kappa", "Lr"],
                3: ["R"]}
STAGE_HOURS = {1: [0, 4, 17, 16, 9, 3, 15, 8, 12, 19], 2: [0, 4, 17, 16, 9, 3, 15, 8, 12, 19],
               3: [0, 4, 12, 19, 7, 14, 20, 23]}
CHART_START = dict(A=0.128, xs=556.0, bar_s=205.0, bar_B=1.3, bar_w=40.0, gamma=0.73, alpha=1.0, Cf=0.038,
                   R=0.3, kappa=0.5, Lr=50.0, crest_cap=0.75, ch_w=40.0, ch_c=0.5, ch_y=400.0)
JOBS = {
    "full_s0": dict(start=dict(CHART_START), flags={}),
    "full_s1": dict(start=dict(CHART_START, ch_y=330.0, gamma=0.6, R=0.6, Cf=0.067, kappa=0.9, bar_s=230.0), flags={}),
    "full_s2": dict(start=dict(CHART_START, ch_y=450.0, gamma=0.85, R=0.1, ch_w=25.0, ch_c=0.3, bar_s=180.0,
                               kappa=0.3, alpha=2.0), flags={}),
    "full_s3": dict(start=dict(CHART_START, ch_y=290.0, gamma=0.5, R=0.8, bar_B=0.7, kappa=1.4, Lr=150.0,
                               bar_w=60.0), flags={}),
    "no_channel": dict(start=dict(CHART_START, ch_c=0.0), flags=dict(channel=False, current=False), skip=[2]),
    "no_breaking": dict(start=dict(CHART_START), flags=dict(breaking=False), drop=["gamma", "alpha"]),
    "no_current": dict(start=dict(CHART_START), flags=dict(current=False), drop=["kappa", "Lr"]),
    "no_reflection": dict(start=dict(CHART_START, R=0.0), flags=dict(reflection=False), skip=[3]),
    "linear": dict(start=dict(CHART_START), flags=dict(nonlinear=False), drop=["crest_cap"]),
}


def full_P(par):
    P = dict(sim.TRUE)
    P.update(rip_wfac=1.0)  # jet width unknown to the fitter: assume channel width
    P.update(par)
    return P


def resid_hour(args):
    P, flags, i = args
    r = sim.simulate(P, GIVEN[i], flags)
    o = OBS[i]
    out = []
    for f in SIG:
        ob = o[f]
        m = MASK & np.isfinite(ob)
        s = np.sqrt((0.05 * ob[m]) ** 2 + 0.02 ** 2) if f in ("Hs", "Cs") else SIG[f]
        rr = (r[f][m].astype(float) - ob[m]) / s
        out.append(np.where(np.isfinite(rr), rr, 0.0))
    return np.concatenate(out)


if __name__ == "__main__":
    job, stage = sys.argv[1], int(sys.argv[2])
    nproc = int(os.environ.get("NPROC", "4"))
    state_f = os.path.join(ROOT, "fits2", f"{job}.json")
    if stage == 9:   # share a stage-1 (profile) fit made with identical strip physics
        src = json.load(open(os.path.join(ROOT, "fits2", f"{sys.argv[3]}.json")))
        state = dict(par=dict(JOBS[job]["start"]), log=[dict(src["log"][0], shared_from=sys.argv[3])])
        state["par"].update(src["log"][0]["fitted"])
        json.dump(state, open(state_f, "w"), indent=1)
        print(job, "stage1 shared from", sys.argv[3])
        sys.exit()
    if stage == 1 or not os.path.exists(state_f):
        state = dict(par=dict(JOBS[job]["start"]), log=[])
    else:
        state = json.load(open(state_f))
    flags = JOBS[job]["flags"]
    class _Serial:
        def map(self, f, it):
            return list(map(f, it))
    pool = Pool(nproc) if nproc > 1 else _Serial()
    if stage == 0:
        P = full_P(state["par"])
        rs = pool.map(resid_hour, [(P, flags, i) for i in range(24)])
        state["chi2_all_given"] = float(np.mean(np.concatenate(rs) ** 2))
        json.dump(state, open(state_f, "w"), indent=1)
        print(job, "chi2_all_given", state["chi2_all_given"])
        sys.exit()
    names = [n for n in STAGE_PARAMS[stage] if n not in JOBS[job].get("drop", [])]
    if stage in JOBS[job].get("skip", []) or not names:
        json.dump(state, open(state_f, "w"), indent=1)
        print(job, "stage", stage, "skipped")
        sys.exit()
    hours = STAGE_HOURS[stage]
    lb = np.array([BOUNDS[n][0] for n in names]); ub = np.array([BOUNDS[n][1] for n in names])
    x0 = np.clip([state["par"][n] for n in names], lb + 1e-6 * (ub - lb), ub - 1e-6 * (ub - lb))
    scale = 0.05 * (ub - lb)
    cache = {}

    def mkP(x):
        par = dict(state["par"]); par.update({n: float(v) for n, v in zip(names, x)})
        return full_P(par)

    def fun(x):
        key = tuple(np.round(x, 10))
        if key not in cache:
            cache[key] = np.concatenate(pool.map(resid_hour, [(mkP(x), flags, i) for i in hours]))
        return cache[key]

    def jac(x):
        f0 = fun(x)
        steps, tasks = [], []
        for j in range(len(x)):
            h = 0.3 * scale[j] if x[j] + 0.3 * scale[j] <= ub[j] else -0.3 * scale[j]
            xx = x.copy(); xx[j] += h; steps.append(h)
            tasks += [(mkP(xx), flags, i) for i in hours]
        rs = pool.map(resid_hour, tasks)
        J = np.zeros((len(f0), len(x))); nh = len(hours)
        for j in range(len(x)):
            J[:, j] = (np.concatenate(rs[j * nh:(j + 1) * nh]) - f0) / steps[j]
        return J

    t0 = time.time()
    c0 = float(np.mean(fun(x0) ** 2))
    res = least_squares(fun, x0, jac=jac, bounds=(lb, ub), x_scale=scale, max_nfev=12, method="trf")
    state["par"].update({n: float(v) for n, v in zip(names, res.x)})
    state["log"].append(dict(stage=stage, chi2_start=c0, chi2_end=float(np.mean(res.fun ** 2)), nfev=int(res.nfev),
                             secs=time.time() - t0, fitted={n: float(v) for n, v in zip(names, res.x)}))
    json.dump(state, open(state_f, "w"), indent=1)
    print(job, json.dumps(state["log"][-1]))
