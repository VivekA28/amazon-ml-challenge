from pathlib import Path
import pandas as pd

PRED = Path("experiments/predictions_validation_top2_085_final.tsv")
GT = Path("data/validation/val_ground_truth.tsv")
FEATURES = Path("experiments/features_validation_top2.tsv")
SCORES = Path("experiments/scores_validation_top2.tsv")
OUT = Path("experiments/v2_overmatch_diagnostics.tsv")

# ------------------------------------------------------------
# Load predictions
# ------------------------------------------------------------
pred = pd.read_csv(
    PRED,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

pred_map = {}

for _, row in pred.iterrows():
    ids = (
        set(row["matched_entity_ids"].split(","))
        if row["matched_entity_ids"]
        else set()
    )
    pred_map[row["source1_entity_id"]] = ids

# ------------------------------------------------------------
# Load ground truth
# ------------------------------------------------------------
gt = pd.read_csv(
    GT,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

gt_map = {}

for _, row in gt.iterrows():
    ids = (
        set(row["matched_entity_ids"].split(","))
        if row["matched_entity_ids"]
        else set()
    )
    gt_map[row["source1_entity_id"]] = ids

# ------------------------------------------------------------
# Identify overmatched S1s
# ------------------------------------------------------------
overmatched = {}

for s1, true_ids in gt_map.items():
    pred_ids = pred_map.get(s1, set())

    if (
        len(true_ids) > 0
        and len(pred_ids) > len(true_ids)
        and len(pred_ids & true_ids) > 0
    ):
        overmatched[s1] = {
            "true": true_ids,
            "pred": pred_ids,
        }

print(f"Overmatched S1s: {len(overmatched):,}")

if not overmatched:
    raise SystemExit("No overmatched entities found.")

# ------------------------------------------------------------
# Load scores + features
# ------------------------------------------------------------
scores = pd.read_csv(
    SCORES,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

scores["score"] = scores["score"].astype(float)

selected_pairs = []

for s1, info in overmatched.items():
    for cid in info["pred"]:
        selected_pairs.append((s1, cid))

selected = pd.DataFrame(
    selected_pairs,
    columns=["source1_entity_id", "candidate_entity_id"],
)

# Merge scores first.
selected = selected.merge(
    scores,
    on=["source1_entity_id", "candidate_entity_id"],
    how="left",
)

# Merge features.
features = pd.read_csv(
    FEATURES,
    sep="\t",
    dtype=str,
    keep_default_na=False,
)

feature_cols = [
    "source1_entity_id",
    "candidate_entity_id",
    "name_exact",
    "name_core_exact",
    "name_jaccard",
    "name_overlap",
    "name_ratio",
    "address_exact",
    "address_jaccard",
    "address_overlap",
    "address_ratio",
    "address_number_equal",
    "country_equal",
    "name_address_mean",
    "name_address_min",
]

features = features[feature_cols]

selected = selected.merge(
    features,
    on=["source1_entity_id", "candidate_entity_id"],
    how="left",
)

# ------------------------------------------------------------
# Add truth label + prediction type
# ------------------------------------------------------------
def classify(row):
    true_ids = overmatched[row["source1_entity_id"]]["true"]

    if row["candidate_entity_id"] in true_ids:
        return "TRUE_MATCH"
    return "FALSE_MATCH"


selected["match_type"] = selected.apply(
    classify,
    axis=1,
)

# Sort so false matches are easy to inspect.
selected = selected.sort_values(
    ["source1_entity_id", "match_type", "score"],
    ascending=[True, True, False],
)

OUT.parent.mkdir(parents=True, exist_ok=True)
selected.to_csv(
    OUT,
    sep="\t",
    index=False,
)

print()
print("=" * 78)
print("V2 OVERMATCH DIAGNOSTICS")
print("=" * 78)
print(f"Overmatched S1s       : {len(overmatched):,}")
print(f"Selected pairs        : {len(selected):,}")
print(
    f"True selected pairs   : "
    f"{(selected['match_type'] == 'TRUE_MATCH').sum():,}"
)
print(
    f"False selected pairs  : "
    f"{(selected['match_type'] == 'FALSE_MATCH').sum():,}"
)
print(f"Output                : {OUT}")
