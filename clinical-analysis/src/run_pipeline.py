"""Clean, logged, stop-on-error orchestrator for the full analysis pipeline.

This is a thin runner: it invokes the phase modules in the canonical order, one
subprocess each, in the SAME interpreter (sys.executable) so everything stays
inside the uv-managed .venv. It contains no analysis logic of its own.

Every analysis phase is checkpoint-resumable (output/archived/*.pkl), so an
interrupted or failed run can be restarted; use --from N to resume at phase N.

Phase order:
    1 pairwise        -> SuppFig Y-a / Y-b   (pairwise_v2_checkpoint.pkl)
    2 multivariate    -> Figure_main + Y-c   (multivariate_v2_checkpoint.pkl)
    3 volume_only     -> SuppFig X-a 3rd bar  (augments multivariate ckpt)
    4 supplementary   -> SuppFig X-b, Z-a, Z-b
    5 render_main     -> Figure_main, SuppFig Y-a/b/c + 2 xlsx tables
    6 render_3c_v3    -> SuppFig X-a (three-model bar chart)
    7 cohort_table1_sex -> "Cohort summary by sex" sheet in the supplementary
                        workbook (full-cohort M-vs-F Table 1; skips under a sex filter)

Usage:
    uv run python src/run_pipeline.py            # full pipeline
    uv run python src/run_pipeline.py --from 4   # resume from phase 4
"""
from __future__ import annotations

import argparse
import datetime as _dt
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))      # src/
PROJECT = os.path.dirname(HERE)                        # project root
# Honour CCT_OUT_DIR so a sex-stratified rerun logs under its own output folder
# (output-male/-female). Unset => the canonical "output" folder, unchanged.
OUT_DIR = os.environ.get("CCT_OUT_DIR", os.path.join(PROJECT, "output"))
LOG_DIR = os.path.join(OUT_DIR, "_run_logs")

PHASES: list[tuple[str, list[str]]] = [
    ("pairwise",      ["pairwise_analysis.py", "200"]),
    ("multivariate",  ["multivariate_analysis.py", "200"]),
    ("volume_only",   ["volume_only_analysis.py", "200"]),
    ("supplementary", ["supplementary_analyses.py"]),
    ("render_main",   ["render_main_figures.py"]),
    ("render_3c_v3",  ["render_figure_3c_v3.py"]),
    ("cohort_table1_sex", ["cohort_table1_by_sex.py"]),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--from", dest="start", type=int, default=1,
                    help="1-based phase index to start from (default 1)")
    args = ap.parse_args()

    os.makedirs(LOG_DIR, exist_ok=True)
    stamp = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    logpath = os.path.join(LOG_DIR, f"pipeline_{stamp}.log")

    def log(msg: str) -> None:
        line = f"{_dt.datetime.now():%H:%M:%S}  {msg}"
        print(line, flush=True)
        with open(logpath, "a", encoding="utf-8") as fh:
            fh.write(line + "\n")

    log(f"PIPELINE START  python={sys.version.split()[0]}  seeds-config-driven  log={logpath}")
    t0 = time.time()

    for i, (name, cmd) in enumerate(PHASES, 1):
        if i < args.start:
            log(f"[{i}/{len(PHASES)}] {name:<14s} SKIPPED (--from {args.start})")
            continue
        script = os.path.join(HERE, cmd[0])
        full = [sys.executable, script, *cmd[1:]]
        log(f"[{i}/{len(PHASES)}] {name:<14s} START  ({' '.join(cmd)})")
        phase_t = time.time()
        # Run from PROJECT so CatBoost's catboost_info/ lands at the project root
        # (regenerable). Script dir (src/) is auto-added to sys.path[0] regardless
        # of cwd, so the flat `import config` etc. resolve.
        # PYTHONUTF8=1 forces UTF-8 stdout so the phase scripts' unicode prints
        # (eta, Delta, R-squared, box-drawing) don't crash on Windows' cp1252
        # console. This is an environment setting only; analysis code is untouched.
        child_env = {**os.environ, "PYTHONUTF8": "1", "PYTHONIOENCODING": "utf-8"}
        with open(logpath, "a", encoding="utf-8") as fh:
            result = subprocess.run(full, cwd=PROJECT, stdout=fh,
                                    stderr=subprocess.STDOUT, env=child_env)
        dt = time.time() - phase_t
        if result.returncode != 0:
            log(f"[{i}/{len(PHASES)}] {name:<14s} FAILED rc={result.returncode} after {dt:.0f}s")
            log("PIPELINE FAILED")
            sys.exit(1)
        log(f"[{i}/{len(PHASES)}] {name:<14s} OK in {dt:.0f}s")

    log(f"PIPELINE COMPLETE in {time.time() - t0:.0f}s  ({(time.time()-t0)/60:.1f} min)")


if __name__ == "__main__":
    main()
