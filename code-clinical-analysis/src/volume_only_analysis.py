"""Volume-only model — features = the 8 raw substructure volumes only
(8 features, NO demographics and NO ratios). Fitted under identical 5-fold CV /
inner-validation early stopping as the baseline and full models.

Adds three keys to each entry of the multivariate_v2_checkpoint:
  * r2_vol_only      — out-of-fold R²
  * ci_vol_only      — percentile bootstrap 95 % CI of the volume-only R²
  * pred_vol_only    — pooled OOF predictions (for the figure)

Resumable: skips any target that already has r2_vol_only stored.

Usage:
    python3 volume_only_analysis.py [MAX_PER_CALL]
"""
import os, sys, pickle, time, warnings
warnings.filterwarnings("ignore")
import numpy as np, pandas as pd
from sklearn.metrics import r2_score

from config import (
    CKPT_DIR, RAW_VOLUMES, CLINICAL_TARGETS, TARGET_DISPLAY,
    N_BOOT, N_MIN, RANDOM_STATE_PRIMARY,
)
from core import (
    load_cohort, fit_catboost_oof_repeated,
)
from multivariate_analysis import CKPT          # same checkpoint file

VOLUME_ONLY_FEATURES = RAW_VOLUMES              # 8 features, no demographics, no ratios
MAX_PER_CALL_DEFAULT = 3


def run(max_per_call):
    if not os.path.exists(CKPT):
        raise RuntimeError(
            "multivariate checkpoint not found — run multivariate_analysis.py first")
    with open(CKPT, "rb") as f: state = pickle.load(f)

    df = load_cohort()
    pending = [t for t in CLINICAL_TARGETS
               if t in state["results"]
               and not state["results"][t].get("skipped")
               and "r2_vol_only" not in state["results"][t]]
    print(f"Volume-only fits pending: {len(pending)} / {len(state['results'])}")
    if not pending:
        return state

    processed = 0
    for target in pending:
        if processed >= max_per_call: break
        r = state["results"][target]
        # Build the same complete-case cohort the baseline+full models used,
        # plus the volumes (which were already in the full feature set anyway).
        df_sub = df.dropna(subset=VOLUME_ONLY_FEATURES + [target])
        if len(df_sub) < N_MIN: continue
        t0 = time.time()
        vol = fit_catboost_oof_repeated(df_sub, VOLUME_ONLY_FEATURES, target)
        y, pv = vol["y"], vol["mean_pred"]

        r2_vol = float(r2_score(y, pv))

        # Percentile bootstrap 95% CI for R²_vol_only: resample patients with
        # replacement and take the 2.5/97.5 percentiles of the bootstrapped R².
        # A one-sample percentile interval is appropriate here — there is no
        # paired comparator, unlike the BCa interval used for nested-model ΔR².
        rng = np.random.default_rng(RANDOM_STATE_PRIMARY)
        boots = []
        for _ in range(N_BOOT):
            idx = rng.integers(0, len(y), len(y))
            if np.var(y[idx]) < 1e-12: continue
            boots.append(r2_score(y[idx], pv[idx]))
        ci_lo, ci_hi = np.percentile(boots, [2.5, 97.5]).tolist()

        r["r2_vol_only"] = r2_vol
        r["ci_vol_only"] = [float(ci_lo), float(ci_hi)]
        r["pred_vol_only"] = pv.tolist()
        with open(CKPT, "wb") as f: pickle.dump(state, f)

        dt = time.time() - t0
        print(f"  {r['display']:<18s} n={r['n']:>5d}  "
              f"R²(vol-only) = {r2_vol:+.3f} [{ci_lo:+.2f}, {ci_hi:+.2f}]  ({dt:.0f}s)")
        processed += 1

    remaining = sum(1 for t in pending if "r2_vol_only" not in state["results"][t])
    print(f"\nProcessed {processed} this call. Remaining: {remaining}")
    return state


if __name__ == "__main__":
    n = int(sys.argv[1]) if len(sys.argv) > 1 else MAX_PER_CALL_DEFAULT
    run(n)
