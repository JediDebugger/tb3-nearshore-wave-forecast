"""Run every earlier blind submission (its own python) on a conditions file.
usage: python run_subs.py conds.json outroot [tags...]  -> outroot/<tag>/case_<k>.npz"""
import sys, os, subprocess, time
TAGS = ["r1", "r2", "r3", "a2"]
ENV = dict(os.environ, OMP_NUM_THREADS="1", MKL_NUM_THREADS="1", OPENBLAS_NUM_THREADS="1", VECLIB_MAXIMUM_THREADS="1")
if __name__ == "__main__":
    conds, outroot = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
    tags = sys.argv[3:] or TAGS
    procs = {}
    for t in tags:
        d = f"/Users/dev/ocean-{t}-probe"
        out = os.path.join(outroot, t); os.makedirs(out, exist_ok=True)
        log = open(os.path.join(outroot, f"{t}.log"), "w")
        procs[t] = (subprocess.Popen([f"{d}/py/bin/python", f"{d}/app/predict.py", conds, out], env=ENV,
                                     stdout=log, stderr=subprocess.STDOUT, cwd="/tmp"), time.time())
    for t, (p, t0) in procs.items():
        rc = p.wait(); print(t, "rc", rc, "secs %.0f" % (time.time() - t0), flush=True)
