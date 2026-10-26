"""I re-grade saved fitted models without refitting or calling an LLM."""

import json
from pathlib import Path
import sys
import time

sys.dont_write_bytecode = True

ROOT = Path(__file__).resolve().parents[3]
TASK = ROOT / "tasks/nearshore-wave-forecast"
sys.path.insert(0, str(TASK / "solution"))
sys.path.insert(0, str(TASK / "tests"))
import grade
import sim


def main():
    conditions = json.loads((TASK / "tests/unseen_conditions.json").read_text())
    results = {}
    for start in range(4):
        name = f"full_s{start}"
        fit = json.loads((Path(__file__).parent / f"fits2/{name}.json").read_text())
        parameters = dict(sim.TRUE, rip_wfac=1.0)
        parameters.update(fit["par"])
        started = time.monotonic()
        predictions = [sim.simulate(parameters, condition) for condition in conditions]
        passed, report = grade.grade_all(predictions, conditions)
        results[name] = {"passed": bool(passed), "worst_error_over_tolerance": float(grade.margin(report)),
                         "seconds": round(time.monotonic() - started, 2), "failures": grade.failures(report)}
        print(f"{name}: {results[name]}", flush=True)
    output = Path(sys.argv[1])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
