"""Run the full hero demo without the UI and print the results table.

    python -m tools.headless_demo            (from the repo root)
"""
import sys
import time

sys.stdout.reconfigure(encoding="utf-8")

from backend.demo import Director  # noqa: E402
from backend.engine import Simulation  # noqa: E402


def main(verbose="-v" in sys.argv):
    t0 = time.perf_counter()
    sim = Simulation()
    print(f"models trained in {sim.models.train_seconds}s")
    sim.director = Director(sim)
    n = 0
    while not sim.director.done and n < 8000:
        sim.step(0.1)
        n += 1
    if verbose:
        for e in sim.events:
            print(f"{e['clock']} [{e['f']:>3}] {e['title']}  | {e['detail'][:100]}")
    print(f"\nsimulated {sim.clock()} ({sim.t:.0f}s) in {time.perf_counter() - t0:.1f}s wall time\n")
    for r in sim.results()["rows"]:
        mark = {True: "PASS", False: "MISS", None: "    "}[r["pass"]]
        print(f"{mark}  {r['metric']:<28} baseline {r['baseline']:<30} resQnet {r['resqnet']:<34} target {r['target']}")
    return sim


if __name__ == "__main__":
    main()
