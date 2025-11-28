import pandas as pd

# === 1) Load CSVs ===
no_fix = pd.read_csv("eval_summary_no_fix.csv")
fix_max = pd.read_csv("eval_fix_max.csv")
fix_topk = pd.read_csv("eval_fix_topk.csv")
events = pd.read_csv("validation_events.csv")

print("Columns in eval_summary_no_fix:", no_fix.columns.tolist())
print("Columns in eval_fix_max:", fix_max.columns.tolist())
print("Columns in eval_fix_topk:", fix_topk.columns.tolist())
print("Columns in validation_events:", events.columns.tolist())

# === 2) SET THE COLUMN NAMES ONCE HERE ===
# metric column in the eval_summary_* files (e.g. 'ap', 'AP', 'mAP', etc.)
METRIC_COL = "ap"          # <<< ### SET THIS ###

# event-frame probability columns in validation_events.csv
PROB_NO_FIX_COL  = "prob_no_fix"   # <<< ### SET THIS ###
PROB_FIX_COL     = "prob_fix"      # <<< ### SET THIS ###

# optional ID column to join on if needed (usually something like 'sample', 'video_id', etc.)
ID_COL = "sample"          # <<< ### SET THIS IF YOU WANT JOINS, OR IGNORE BELOW JOIN PART ###


# === 3) Overall performance numbers (Results – Overall AP/mAP section) ===
base_mean = no_fix[METRIC_COL].mean()
fix_max_mean = fix_max[METRIC_COL].mean()
fix_topk_mean = fix_topk[METRIC_COL].mean()

delta_max_abs = fix_max_mean - base_mean
delta_topk_abs = fix_topk_mean - base_mean

delta_max_rel = 100.0 * delta_max_abs / base_mean
delta_topk_rel = 100.0 * delta_topk_abs / base_mean

print("\n--- OVERALL PERFORMANCE (fill in Results: Overall Performance) ---")
print("BASELINE mAP (no fixation)                [P_BASE_MAP]        =", base_mean)
print("mAP with fixation (max lambda)           [P_FIX_MAX_MAP]      =", fix_max_mean)
print("mAP with fixation (top-k lambda avg)     [P_FIX_TOPK_MAP]     =", fix_topk_mean)
print("Absolute gain (max)                      [P_DELTA_MAX_ABS]    =", delta_max_abs)
print("Relative gain (max, %)                   [P_DELTA_MAX_REL]    =", delta_max_rel)
print("Absolute gain (top-k)                    [P_DELTA_TOPK_ABS]   =", delta_topk_abs)
print("Relative gain (top-k, %)                 [P_DELTA_TOPK_REL]   =", delta_topk_rel)


# === 4) Event-frame probability analysis (Results – Qualitative / Case Study section) ===
# Assumes validation_events has prob at event frame for both conditions.
events["delta_prob"] = events[PROB_FIX_COL] - events[PROB_NO_FIX_COL]

mean_delta = events["delta_prob"].mean()
pos_events = (events["delta_prob"] > 0).sum()
neg_events = (events["delta_prob"] < 0).sum()
zero_events = (events["delta_prob"] == 0).sum()
total_events = len(events)

frac_pos = 100.0 * pos_events / total_events
frac_neg = 100.0 * neg_events / total_events
frac_zero = 100.0 * zero_events / total_events

print("\n--- EVENT-FRAME ANALYSIS (fill in Results: Event-frame Saliency Effects) ---")
print("Mean Δ accident prob at event frame      [P_MEAN_DELTA_PROB]  =", mean_delta)
print("#events with improved prob (Δ>0)         [P_N_POS_EVENTS]     =", pos_events,
      f"({frac_pos:.2f}%)  [P_FRAC_POS_EVENTS]")
print("#events with reduced prob (Δ<0)          [P_N_NEG_EVENTS]     =", neg_events,
      f"({frac_neg:.2f}%)  [P_FRAC_NEG_EVENTS]")
print("#events unchanged (Δ=0)                  [P_N_ZERO_EVENTS]    =", zero_events,
      f"({frac_zero:.2f}%) [P_FRAC_ZERO_EVENTS]")


# === 5) (Optional) Join by ID and compute per-sample gains (Results – Per-sample statistics) ===
if ID_COL in no_fix.columns and ID_COL in fix_max.columns:
    merged = no_fix[[ID_COL, METRIC_COL]].merge(
        fix_max[[ID_COL, METRIC_COL]],
        on=ID_COL,
        suffixes=("_nofix", "_fix")
    )
    merged["delta_metric"] = merged[f"{METRIC_COL}_fix"] - merged[f"{METRIC_COL}_nofix"]

    improved = (merged["delta_metric"] > 0).sum()
    worsened = (merged["delta_metric"] < 0).sum()
    same = (merged["delta_metric"] == 0).sum()
    total = len(merged)

    frac_improved = 100.0 * improved / total
    frac_worsened = 100.0 * worsened / total
    frac_same = 100.0 * same / total

    print("\n--- PER-SAMPLE GAINS (fill in Results: Per-video statistics) ---")
    print("#videos where fixation helps             [P_N_IMPROVED]      =", improved,
          f"({frac_improved:.2f}%) [P_FRAC_IMPROVED]")
    print("#videos where fixation hurts             [P_N_WORSENED]      =", worsened,
          f"({frac_worsened:.2f}%) [P_FRAC_WORSENED]")
    print("#videos unchanged                        [P_N_SAME]          =", same,
          f"({frac_same:.2f}%)   [P_FRAC_SAME]")
else:
    print("\n[INFO] ID_COL not found in eval summaries; skipping per-sample join.")

