"""
evaluate_pure_pursuit.py
------------------------
Programmatic evaluation harness for tuning the Pure Pursuit lateral controller
in Scenic.  Follows the TA's instructions:

  • 100 runs per parameter set, same seeds across all parameter sets
  • Off-road steps get a large penalty (OFF_ROAD_PENALTY) instead of raw CTE
  • Score = mean of per-run means (lower is better)
  • Results written to results.csv for easy inspection

Usage
-----
  # Quick smoke-test (5 runs):
  python evaluate_pure_pursuit.py --runs 5

  # Full sweep:
  python evaluate_pure_pursuit.py --runs 100

  # Single parameter combo (useful for debugging):
  python evaluate_pure_pursuit.py --runs 10 --K_dd 0.5 --min_ld 3 --max_ld 15 --clwbr 0.72

Requirements
------------
  pip install scenic          # already installed in your env
  The kyle.scenic file must be reachable at SCENIC_FILE below.
  Adjust MAP_PATH if needed.
"""

import argparse
import csv
import itertools
import os
import random
import sys
import traceback
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import List, Optional

# ── adjust these paths to match your repo layout ─────────────────────────────
SCENIC_FILE  = Path("examples/driving/kyle.scenic")   # your scenario file
MAP_PATH     = "assets/maps/CARLA/Town05.xodr"         # relative path used inside kyle.scenic
RESULTS_CSV  = Path("results.csv")
# ─────────────────────────────────────────────────────────────────────────────

OFF_ROAD_PENALTY = 10.0   # metres added for every off-road timestep
MAX_STEPS        = 600    # safety cap (60 s at 10 Hz)
SEED_BASE        = 0      # seeds will be SEED_BASE … SEED_BASE+N_RUNS-1


# ── parameter grid to sweep ──────────────────────────────────────────────────
# Add/remove values here.  Every combination is evaluated.
PARAM_GRID = {
    "K_dd":   [0.3, 0.5, 0.7],
    "min_ld": [2.0, 3.0, 5.0],
    "max_ld": [10.0, 15.0, 20.0],
    "clwbr":  [0.72],          # fixed at the value your TA confirmed is good
}
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class RunResult:
    seed: int
    K_dd: float
    min_ld: float
    max_ld: float
    clwbr: float
    mean_cte: float          # mean |CTE| over the run (with penalties)
    off_road_steps: int
    total_steps: int
    terminated_early: bool   # True if the scenario violated a requirement
    error: str = ""


def make_param_overrides(K_dd, min_ld, max_ld, clwbr) -> dict:
    """
    Build a Scenic param-override dict.

    kyle.scenic reads `param map` from the scenario file; we override it here
    so the harness is self-contained.  The controller parameters are injected
    as global Scenic params and read inside behaviors.scenic / controllers.py
    via a small shim (see note below).
    """
    return {
        "map":    MAP_PATH,
        # Pure Pursuit knobs — read in getPurePursuitControllers() shim
        "pp_K_dd":   K_dd,
        "pp_min_ld": min_ld,
        "pp_max_ld": max_ld,
        "pp_clwbr":  clwbr,
    }


def run_single(scenic_scenario, seed: int, params: dict) -> RunResult:
    """
    Execute one simulation and return a RunResult.

    scenic_scenario is a pre-compiled Scenic Scenario object (compiled once
    and reused across runs for speed).
    """
    import scenic.syntax.veneer as veneer  # noqa – needed for simulator hooks

    K_dd   = params["K_dd"]
    min_ld = params["min_ld"]
    max_ld = params["max_ld"]
    clwbr  = params["clwbr"]

    cte_accumulator: List[float] = []
    off_road_steps = 0
    terminated_early = False
    error_msg = ""

    try:
        scene, _ = scenic_scenario.generate(maxIterations=50, feedback=None)

        # Patch the simulator factory so getPurePursuitControllers uses our params.
        # We monkey-patch the DrivingSimulation class method BEFORE the simulation
        # is created, then restore it afterward.
        from scenic.domains.driving.simulators import DrivingSimulation
        from scenic.domains.driving.controllers import PurePursuitLateralController, PIDLongitudinalController

        _original = DrivingSimulation.getPurePursuitControllers

        def _patched_getPurePursuitControllers(self_sim, agent, cl=4.5, ld=7, clwbr_=0.72):
            dt = self_sim.timestep
            lon = PIDLongitudinalController(K_P=0.5, K_D=0.1, K_I=0.7, dt=dt)
            lat = PurePursuitLateralController(
                cl=agent.length,
                ld=ld,
                dt=dt,
                clwbr=clwbr,       # from outer scope
                K_dd=K_dd,         # from outer scope
                min_ld=min_ld,     # from outer scope
                max_ld=max_ld,     # from outer scope
            )
            return lon, lat

        DrivingSimulation.getPurePursuitControllers = _patched_getPurePursuitControllers

        try:
            simulation = scenic_scenario.simulatorFactory().simulate(
                scene,
                maxSteps=MAX_STEPS,
                seed=seed,
                raiseGuardViolations=False,  # don't crash on require failures
            )
        finally:
            DrivingSimulation.getPurePursuitControllers = _original  # always restore

        if simulation is None:
            # scenario rejected (initial condition not satisfied) — skip
            return RunResult(
                seed=seed, K_dd=K_dd, min_ld=min_ld, max_ld=max_ld, clwbr=clwbr,
                mean_cte=float("nan"), off_road_steps=0, total_steps=0,
                terminated_early=True, error="scene_rejected",
            )

        # ── collect per-step CTE from simulation records ──────────────────
        # Scenic stores records in simulation.result if you set up monitors;
        # we use the monitor output that kyle.scenic already prints.
        # A cleaner approach: read from simulation.records (if available) or
        # recompute from trajectory.
        ego_records = simulation.result.records.get("ego", []) if simulation.result else []

        # Fallback: recompute CTE from the saved trajectory.
        # simulation.trajectory is a list of dicts {object: state} per step.
        # network = scene.workspace  # PolylineRegion network proxy — see note
        
        # The cleanest approach given your existing monitor: parse the printed
        # CTE values.  But since we're running programmatically, we recompute:
        for state_dict in simulation.trajectory:
            ego_state = state_dict.get(scene.egoObject)
            if ego_state is None:
                continue

            position = ego_state.position
            on_road  = ego_state.onRoad  if hasattr(ego_state, "onRoad")  else None
            lane     = ego_state.lane    if hasattr(ego_state, "lane")    else None

            if lane is not None:
                cte = abs(lane.centerline.signedDistanceTo(position))
            else:
                cte = OFF_ROAD_PENALTY  # no lane info → treat as off-road

            if on_road is False:
                cte += OFF_ROAD_PENALTY
                off_road_steps += 1

            cte_accumulator.append(cte)

        terminated_early = (
            simulation.result is not None
            and not simulation.result.terminationType.isNormalCompletion
        ) if simulation.result else False

    except Exception as exc:
        error_msg = repr(exc)
        traceback.print_exc()

    mean_cte = (
        sum(cte_accumulator) / len(cte_accumulator)
        if cte_accumulator
        else float("nan")
    )

    return RunResult(
        seed=seed,
        K_dd=K_dd,
        min_ld=min_ld,
        max_ld=max_ld,
        clwbr=clwbr,
        mean_cte=mean_cte,
        off_road_steps=off_road_steps,
        total_steps=len(cte_accumulator),
        terminated_early=terminated_early,
        error=error_msg,
    )


def evaluate_params(scenic_scenario, params: dict, seeds: List[int]) -> dict:
    """Run N seeds for one parameter combo and return aggregate stats."""
    results = []
    for seed in seeds:
        r = run_single(scenic_scenario, seed, params)
        results.append(r)
        status = "OFF-ROAD" if r.off_road_steps > 0 else "ok"
        print(
            f"  seed={seed:4d}  mean_cte={r.mean_cte:6.3f}  "
            f"off_road={r.off_road_steps:3d}/{r.total_steps}  [{status}]"
            + (f"  ERR: {r.error[:60]}" if r.error else "")
        )

    valid = [r for r in results if not (r.error or r.mean_cte != r.mean_cte)]
    aggregate_mean_cte = sum(r.mean_cte for r in valid) / len(valid) if valid else float("nan")
    total_off_road     = sum(r.off_road_steps for r in valid)
    total_steps        = sum(r.total_steps    for r in valid)

    return {
        **params,
        "aggregate_mean_cte": aggregate_mean_cte,
        "total_off_road_steps": total_off_road,
        "total_steps": total_steps,
        "off_road_fraction": total_off_road / total_steps if total_steps else float("nan"),
        "n_valid_runs": len(valid),
        "all_results": results,
    }


def compile_scenario(params_override: dict):
    """Compile the Scenic scenario file once (expensive) and return it."""
    import scenic
    scenario = scenic.scenarioFromFile(
        str(SCENIC_FILE),
        params=params_override,
        mode2D=True,
    )
    return scenario


def write_csv(all_agg: list, path: Path):
    if not all_agg:
        return
    keys = [k for k in all_agg[0] if k != "all_results"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        for row in all_agg:
            writer.writerow({k: row[k] for k in keys})
    print(f"\nResults written to {path}")


def main():
    parser = argparse.ArgumentParser(description="Pure Pursuit parameter sweep for Scenic")
    parser.add_argument("--runs",   type=int,   default=100,  help="Number of seeds per param combo")
    parser.add_argument("--K_dd",   type=float, default=None, help="Fix K_dd (skips grid sweep)")
    parser.add_argument("--min_ld", type=float, default=None, help="Fix min_ld")
    parser.add_argument("--max_ld", type=float, default=None, help="Fix max_ld")
    parser.add_argument("--clwbr",  type=float, default=None, help="Fix clwbr (default 0.72)")
    args = parser.parse_args()

    # Build seeds — same seeds used for every parameter combo
    seeds = list(range(SEED_BASE, SEED_BASE + args.runs))

    # Build parameter combos
    if any(v is not None for v in [args.K_dd, args.min_ld, args.max_ld, args.clwbr]):
        # Single combo override from CLI
        combos = [{
            "K_dd":   args.K_dd   if args.K_dd   is not None else 0.5,
            "min_ld": args.min_ld if args.min_ld is not None else 3.0,
            "max_ld": args.max_ld if args.max_ld is not None else 15.0,
            "clwbr":  args.clwbr  if args.clwbr  is not None else 0.72,
        }]
    else:
        grid = PARAM_GRID
        combos = [
            dict(zip(grid.keys(), vals))
            for vals in itertools.product(*grid.values())
        ]

    print(f"Evaluating {len(combos)} parameter combo(s) × {args.runs} seeds each")
    print(f"Scenic file : {SCENIC_FILE}")
    print(f"Seeds       : {seeds[0]} … {seeds[-1]}\n")

    # Compile scenario once with dummy params (the important map param is fixed)
    dummy_params = make_param_overrides(
        K_dd=combos[0]["K_dd"],
        min_ld=combos[0]["min_ld"],
        max_ld=combos[0]["max_ld"],
        clwbr=combos[0]["clwbr"],
    )
    print("Compiling Scenic scenario (this takes ~2-5 seconds)...")
    scenic_scenario = compile_scenario(dummy_params)
    print("Compiled.\n")

    all_agg = []
    best_score = float("inf")
    best_params = None

    for i, combo in enumerate(combos):
        label = f"K_dd={combo['K_dd']} min_ld={combo['min_ld']} max_ld={combo['max_ld']} clwbr={combo['clwbr']}"
        print(f"[{i+1}/{len(combos)}] {label}")

        agg = evaluate_params(scenic_scenario, combo, seeds)
        all_agg.append(agg)

        score = agg["aggregate_mean_cte"]
        print(
            f"  → aggregate_mean_cte={score:.4f}  "
            f"off_road_fraction={agg['off_road_fraction']:.3f}  "
            f"valid_runs={agg['n_valid_runs']}/{args.runs}\n"
        )

        if score < best_score:
            best_score = score
            best_params = combo

    print("=" * 60)
    print(f"Best parameters: {best_params}")
    print(f"Best score (mean |CTE| with penalties): {best_score:.4f} m")
    print("=" * 60)

    write_csv(all_agg, RESULTS_CSV)


if __name__ == "__main__":
    main()
