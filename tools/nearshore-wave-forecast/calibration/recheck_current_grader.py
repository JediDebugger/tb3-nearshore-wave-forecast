"""I re-grade the calibration candidates with the current task grader, without refitting or calling an LLM.

usage: python recheck_current_grader.py OUT.json   (Python with numpy 2.0.2 and scipy 1.13.1)

It rebuilds the working-folder layout the recorded scripts expect (hidden/, solver/data, audit/) in a
temporary folder of links to this repository's files, so calib.py, fit2.py and sim_alt.py run unchanged
against tests/grade.py, solution/sim.py and the unseen set. Candidates:
  reference          sim.TRUE
  pm2_0 .. pm2_9     the ten +-2 % draws of calib.py (rng seed 11)
  fits2/*            the saved staged refits (four starts, five with one process left out), graded as
                     a predict.py would write them (no NaN filling)
  alt_gd, alt_miche  breaking parameters refitted with SWAN-style Hmax = gamma d, and with the
                     reference's own Miche-type Hmax (audit/fit_gd.py)
  shortcuts          calib.py's chart-based spectral model, regression and persistence (calib.py fills
                     their NaNs along x, as recorded)
  kappa_x*           the reference with the rip-speed coefficient kappa scaled
"""

import json
import os
from pathlib import Path
import sys
import tempfile

sys.dont_write_bytecode = True
HERE = Path(__file__).resolve().parent
TOOLS = HERE.parent
TASK = TOOLS.parents[1] / "tasks/nearshore-wave-forecast"


def layout(tmp):
    hidden = tmp / "hidden"
    (tmp / "solver").mkdir(parents=True)
    (tmp / "audit").mkdir()
    hidden.mkdir()
    links = {
        hidden / "sim.py": TASK / "solution/sim.py",
        hidden / "grade.py": TASK / "tests/grade.py",
        hidden / "unseen_conditions.json": TASK / "tests/unseen_conditions.json",
        hidden / "truth": TASK / "tests/truth",
        hidden / "calib.py": HERE / "calib.py",
        hidden / "fit2.py": HERE / "fit2.py",
        hidden / "fits2": HERE / "fits2",
        hidden / "make_data.py": TOOLS / "generation/make_data.py",
        tmp / "solver/data": TASK / "environment/data",
        tmp / "audit/sim_alt.py": TOOLS / "audit/sim_alt.py",
    }
    for link, target in links.items():
        os.symlink(target, link)
    return hidden


def main(out_path):
    with tempfile.TemporaryDirectory() as d:
        hidden = layout(Path(d))
        sys.path[:0] = [str(hidden), str(Path(d) / "audit")]
        import numpy as np
        import calib
        import fit2 as F2
        import grade
        import sim
        import sim_alt

        unseen = calib.UNSEEN

        def score(preds):
            ok, rep = grade.grade_all(preds, unseen)
            return {"passed": bool(ok), "worst_error_over_tolerance": round(float(grade.margin(rep)), 2),
                    "failures": grade.failures(rep)}

        res = {"grader": "tasks/nearshore-wave-forecast/tests/grade.py (current)",
               "numpy": np.__version__}
        res["reference"] = score(calib.run(sim.TRUE))
        rng = np.random.default_rng(11)
        keys = ["A", "xs", "bar_s", "bar_B", "bar_w", "ch_y", "ch_w", "ch_c", "gamma", "alpha", "Cf", "R",
                "kappa", "Lr", "rip_wfac", "crest_cap", "wall_x", "wall_y"]
        for k in range(10):
            P = dict(sim.TRUE)
            for key in keys:
                P[key] = P[key] * (1 + rng.uniform(-0.02, 0.02))
            P["xs"] = sim.TRUE["xs"] + sim.TRUE["bar_s"] * rng.uniform(-0.02, 0.02)
            P["wall_x"], P["wall_y"] = sim.TRUE["wall_x"], sim.TRUE["wall_y"]
            res[f"pm2_{k}"] = score(calib.run(P))
        for name in ["full_s0", "full_s1", "full_s2", "full_s3",
                     "no_channel", "no_current", "no_reflection", "linear", "no_breaking"]:
            fit = json.loads((HERE / f"fits2/{name}.json").read_text())
            res[f"fit2_{name}"] = score(calib.run(F2.full_P(fit["par"]), F2.JOBS[name]["flags"]))
        for form in ("gd", "miche"):
            p = json.loads((TOOLS / f"audit/fit_{form}.json").read_text())
            P = dict(sim_alt.TRUE, gamma=p["gamma"], alpha=p["alpha"], hmax_form=form)
            res[f"alt_{form}"] = score([sim_alt.simulate(P, c) for c in unseen])
        for name, fn in [("spectral_chart_std", calib.shortcut_spectral),
                         ("regression", calib.shortcut_regression),
                         ("persistence_last", lambda: calib.shortcut_persistence("last")),
                         ("persistence_similar", lambda: calib.shortcut_persistence("similar"))]:
            s = score(calib.fill_nan(fn()))
            s["failures"] = len(s["failures"])
            res[f"shortcut_{name}"] = s
        for f in (0.6, 0.8, 1.2, 1.5):
            preds = calib.run(dict(sim.TRUE, kappa=sim.TRUE["kappa"] * f))
            ok, rep = grade.grade_all(preds, unseen)
            res[f"kappa_x{f}"] = {
                "passed": bool(ok), "worst_error_over_tolerance": round(float(grade.margin(rep)), 2),
                "rip_u_max_m_s": round(max(float(rep[j]["R3_rip"]["u"][0]) for j in rep), 3),
                "rip_u_tolerance_m_s": float(rep[0]["R3_rip"]["u"][1]),
                "rip_Dm_max_deg": round(max(float(rep[j]["R3_rip"]["Dm"][0]) for j in rep), 2),
                "failures": grade.failures(rep)}
        for name, r in res.items():
            if isinstance(r, dict):
                print(name, r["passed"], r["worst_error_over_tolerance"], flush=True)
    Path(out_path).write_text(json.dumps(res, indent=1) + "\n")


if __name__ == "__main__":
    main(sys.argv[1])
