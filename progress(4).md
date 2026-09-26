# Amazon ML Challenge 2026 — Progress Report

**Project:** Business Entity Resolution Challenge  
**Team:** 3 members  
**Repository:** `VivekA28/amazon-ml-challenge`  
**Last updated:** 26 September 2026 — validation model frozen; test candidate pipeline completed through V2 ∪ F2 union; final pairwise test generation pending

---

## 1. Challenge Understanding

### Problem
Build an ML pipeline that resolves business records across three independent noisy sources.

- Source 1 = deduplicated reference source.
- For every Source 1 entity, find all matching records from Source 2 and Source 3.
- A Source 1 entity may have zero, one, or multiple matches.
- Matching uses noisy business names, addresses, and country labels.

**Source:** Official challenge problem statement.

### Evaluation
- Metric: **F_0.5**
- Precision-heavy: precision is weighted more strongly than recall.
- Evaluation is macro-averaged per Source 1 entity.
- Correctly identifying a singleton/no-match entity earns full credit for that entity.
- False merges are particularly costly.

### Important restrictions
- No external business/entity lookup.
- No commercial ER APIs.
- No government-registration lookup.
- No geocoding APIs.
- No external internet data augmentation.
- Final model must be MIT/Apache 2.0 licensed and <= 8B parameters.

---

# 2. Team Setup

## Roles

### Member 1 — Vivek
**Primary ownership:**
- Blocking/candidate-generation experiments
- ML/modeling
- Pipeline integration
- Final system
- Final submission

### Member 2 — Windows
**Primary ownership:**
- Preprocessing / normalization
- Reusable name/address/country normalization pipeline
- Processed-data preparation

### Member 3 — Mac
**Primary ownership:**
- Validation
- Threshold optimization
- Experiment comparison
- Error analysis

> Roles are ownership areas, not restrictions. All members can experiment once the baseline is available.

---

# 3. Repository Setup

## Status: COMPLETE

Repository created and connected to GitHub using SSH.

Current intended structure:

```text
amazon-ml-challenge/
├── code/
│   └── business_entity_resolution/
│       ├── notebooks/
│       │   └── .gitkeep
│       ├── requirements.txt
│       └── src/
│           └── .gitkeep
├── data/
│   ├── test/
│   └── train/
├── documentation/
├── experiments/
│   └── .gitkeep
├── output/
│   └── .gitkeep
├── .gitignore
└── README.md
```

### Git/data policy
- Competition dataset is ignored by Git.
- Dataset must NOT be pushed to GitHub.
- Code, configuration, documentation and experiment records should be version controlled.
- Team members should clone the repository and obtain the challenge dataset separately.

---

# 4. Dataset Acquisition

## Status: COMPLETE

Official challenge dataset obtained from the challenge package.

### Training files

```text
data/train/train_source1.tsv
data/train/train_source2.tsv
data/train/train_source3.tsv
data/train/train_ground_truth.tsv
```

### Test files

```text
data/test/test_source1.tsv
data/test/test_source2.tsv
data/test/test_source3.tsv
```

All files are TSV.

### Dataset size observed

| File | Lines |
|---|---:|
| train_source1.tsv | 2,206,822 |
| train_source2.tsv | 5,034,617 |
| train_source3.tsv | 5,285,604 |
| train_ground_truth.tsv | 2,206,822 |
| test_source1.tsv | 1,732,545 |
| test_source2.tsv | 4,887,274 |
| test_source3.tsv | 5,082,317 |
| **Total** | **26,436,001** |

`wc -l` includes one header line per file.

Approximate data size on disk:

- Training: ~1.3 GB
- Test: ~1.2 GB
- Total: ~2.5 GB

### Actual record counts

| File | Records |
|---|---:|
| Train S1 | 2,206,821 |
| Train S2 | 5,034,616 |
| Train S3 | 5,285,603 |
| Test S1 | 1,732,544 |
| Test S2 | 4,887,273 |
| Test S3 | 5,082,316 |
| **Total records** | **24,429,173** |

---

# 5. Dataset Schema

## Status: VERIFIED

Source files use:

```text
entity_id
business_name
business_address
country
```

Entity ID prefix identifies source:

```text
S1-...
S2-...
S3-...
```

Ground truth uses:

```text
source1_entity_id
matched_entity_ids
```

Example:

```text
S1-965667    S2-681193310,S2-743505751,S3-775321672,...
```

---

# 6. Important Dataset Observations

## Status: INITIAL INSPECTION COMPLETE

### Scale
The dataset contains approximately **24.4 million records**.

Naive all-pairs matching is impossible.

Approximate S1-to-S2/S3 comparison space:

```text
2.2M × (5.0M + 5.3M)
≈ 22.8 trillion pair comparisons
```

Therefore, **blocking/candidate generation is a core part of the solution.**

### Ground truth
Training ground truth contains one row for every training Source 1 entity.

A Source 1 entity can have:
- zero matches
- one match
- multiple S2/S3 matches

### Noise expected
Business names may contain:
- abbreviations
- legal suffix differences
- DBA/trade names
- punctuation differences
- word-order changes
- typos
- transliterations

Addresses may contain:
- abbreviations
- transliteration variants
- missing components
- landmarks
- different numbering
- reordered components

### Country
Training contains US and India.

Test additionally contains France.

Therefore:

**DO NOT hard-code the country set to `{US, India}`.**

---

# 7. Architecture — CURRENT PLAN

## Status: DESIGN PHASE

Initial intended pipeline:

```text
Raw TSV
   │
   ▼
Normalization
   │
   ├── business_name normalization
   ├── business_address normalization
   └── country normalization
   │
   ▼
Blocking / Candidate Generation
   │
   ▼
Candidate Pairs
   │
   ▼
Pairwise Feature Generation
   │
   ├── Name similarities
   ├── Address similarities
   ├── Country agreement
   └── Other engineered features
   │
   ▼
Matching Model
   │
   ▼
Threshold / Decision Logic
   │
   ▼
matching_results.tsv
```

`candidate_pairs.tsv` must represent the final candidate set immediately before the matching model scores the candidates.

---

# 8. Candidate Generation / Blocking

## Status: CURRENT VALIDATION CANDIDATE SET FROZEN — MODEL PHASE STARTING

Blocking has now been evaluated on both the full training ground truth and the entity-safe validation split. The current practical candidate set is frozen for the first pairwise-model iteration.

### EXP-001 — Exact normalized business-name blocking

Normalization used for this experiment:

```python
def normalize_name(name):
    return name.lower().strip().replace("&", "and")
```

Results:

| Metric | Result |
|---|---:|
| Training S1 entities | 2,206,821 |
| S1 with candidates | 1,151,939 |
| Average candidates / S1 | 5.61 |
| True matches | 7,638,365 |
| Recovered true matches | 821,811 |
| Blocking recall | **10.7590%** |

Conclusion: exact normalized-name blocking is very selective but misses most true matches. It is useful as a complementary blocking layer, not as the only blocker.

### EXP-002 — Rare-token business-name blocking

Configuration:

- Minimum token length: 4 characters
- Maximum token document frequency: 50,000
- At most 2 rarest usable tokens per S1 record
- Token indexes built separately for Source 2 and Source 3
- Evaluation performed against the complete training ground truth

Results:

| Metric | Result |
|---|---:|
| Unique tokens | 1,810,688 |
| Usable tokens | 1,810,646 |
| Training S1 entities | 2,206,821 |
| S1 with candidates | 2,142,984 |
| Average candidates / S1 | **18,766.29** |
| True matches | 7,638,365 |
| Recovered true matches | 5,934,101 |
| Blocking recall | **77.6881%** |
| Evaluation time | ~63.2 min |

Conclusion: token blocking increases recall dramatically (**10.76% → 77.69%**) but creates an impractically large candidate set (~18.8K candidates/S1). It is retained as a diagnostic upper-bound reference, not as the practical candidate generator.

### EXP-003 — Country-aware rare-token blocking

Configuration:

- Country-aware token index built separately for Source 2 and Source 3
- Minimum token length: 4 characters
- Maximum token document frequency: 50,000
- One rarest usable name token per Source-1 record
- Candidates restricted to the same country
- Evaluation performed against the complete training ground truth

Results:

| Metric | Result |
|---|---:|
| Unique tokens | 1,810,688 |
| Usable tokens | 1,810,646 |
| Training S1 entities | 2,206,821 |
| S1 with candidates | 2,142,978 |
| Average candidates / S1 | **5,487.38** |
| True matches | 7,638,365 |
| Recovered true matches | 5,315,250 |
| Blocking recall | **69.5862%** |
| Evaluation time | ~19.2 min |

Conclusion: country restriction reduces candidate volume substantially compared with EXP-002, but one token loses true matches. Not sufficient as the sole blocker.

### EXP-004 — Country-aware + 2 rare tokens

**Status: COMPLETE — diagnostic only**

Configuration:

- Same-country candidate retrieval
- Two rarest usable name tokens
- OR/union across the two token postings

Full training-set result:

| Metric | Result |
|---|---:|
| Training S1 entities | 2,206,821 |
| S1 with candidates | 2,206,821 |
| Candidate coverage | **97.1073%** |
| Total candidate pairs | **32,736,494,225** |
| Average candidates / S1 | **14,834.23** |
| Recovered true matches | 5,934,102 |
| Blocking recall | **77.6881%** |
| Evaluation time | ~45.4 min |
| Total runtime | ~47.6 min |

Conclusion: country restriction reduced candidate volume by about 21% versus EXP-002 but produced ~32.7B pairs and did not improve recall over EXP-002. It is not practical for pairwise scoring.

### EXP-005 — Exact-name rule-based validation baseline

This was a matching/scoring baseline rather than a new blocker: exact normalized-name candidates were emitted directly as final matches on the validation split.

| Metric | Result |
|---|---:|
| Macro F0.5 | **19.3429%** |
| Macro precision | 25.8305% |
| Macro recall | 15.7295% |
| Singleton mean F0.5 | 67.9793% |
| Non-singleton mean F0.5 | 16.4493% |

This is a sanity-check baseline, not a candidate final model.

### EXP-006 — Address blocking pilot

A 100K-S1 pilot tested complementary address-aware rules with name constraints. The pilot used approximate posting-frequency controls rather than a full global-frequency pass.

| Variant | Candidate coverage | Avg candidates/S1 | Recovered matches | Blocking recall |
|---|---:|---:|---:|---:|
| A — exact address | 24.09% | 0.35 | 28,441 | 8.22% |
| B — country + address number + name | 83.38% | 17.31 | 172,659 | 49.89% |
| C — country + address token + name | 90.83% | 4.76 | 206,087 | 59.55% |
| D — exact name + B | 93.05% | 26.63 | 198,382 | 57.32% |
| E — exact name + C | **96.63%** | **13.99** | **222,345** | **64.25%** |

The pilot demonstrated useful name/address complementarity, but no address-only variant was selected as the final candidate generator.

### EXP-007 — Practical validation candidate generator v1

The current generator combines three bounded layers:

1. exact normalized name;
2. same-country + rare name token + address-number constraint;
3. same-country + rare name token + address-token constraint.

It was run against the entity-safe validation split using the full training Source 2 + Source 3 pools.

Results:

| Metric | Result |
|---|---:|
| Validation S1 entities | **331,023** |
| S1 with candidates | **321,852** |
| Candidate coverage | **97.2295%** |
| Total candidate pairs | **10,251,775** |
| Average candidates / S1 | **30.97** |
| True matches | **1,144,444** |
| Recovered true matches | **769,629** |
| Blocking recall | **67.2492%** |

This is currently the practical candidate set for the first pairwise-model iteration because it keeps the validation candidate volume around 10.25M pairs instead of tens of billions.

### Abandoned candidate-generator variants

Two follow-up broad variants were generated and rejected on computational-volume grounds:

- **v2:** produced a ~22 GB validation candidate file; not used for modeling and deleted.
- **v3 bounded:** completed on 331,023 validation S1s with 100% S1 coverage, but produced **985,873,678 candidate pairs** (~2,978/S1) in 13.9 min. It was deleted and will not be used for the first model iteration.

These runs confirm that maximizing S1 coverage without controlling pair volume is not sufficient.

### Current blocking decision

**Freeze EXP-007 / `output/candidate_pairs_validation.tsv` for the first model iteration.**

Measured validation characteristics:

```text
97.2295% S1 coverage
10,251,775 candidate pairs
30.97 candidates/S1
67.2492% pairwise blocking recall
```

The remaining ~32.75% blocking recall gap is known and will be revisited only if model/error analysis shows that it is the dominant bottleneck worth additional compute.

Do not feed the 32.7B-pair EXP-004, ~22GB v2, or ~986M-pair v3 candidate sets into the model.

# 9. Feature Engineering

## Status: NOT IMPLEMENTED

Candidate-pair features to investigate:

### Name
- Exact normalized equality
- Jaccard similarity
- Character n-gram similarity
- Levenshtein/edit similarity
- Token overlap
- TF-IDF cosine similarity

### Address
- Exact normalized equality
- Token overlap
- Character similarity
- Address-number agreement
- Component-level similarity
- TF-IDF cosine similarity

### Cross-field
- Country equality
- Name + address combined similarity
- Name/address confidence interactions

### Potential semantic features
Pretrained text embeddings may be tested later if computationally practical.

No external business data may be used.

---

# 10. Model Candidates

## Status: NOT IMPLEMENTED

Baseline candidates to evaluate:

### Baseline A
Rule-based similarity + threshold.

### Baseline B
Classical pairwise features + tree-based classifier/ranker.

Potential models:

- LightGBM
- XGBoost
- CatBoost

### Baseline C
Classical features + pretrained text embedding similarities.

Possible text embedding models will only be considered after checking:
- license
- parameter count
- local computational cost
- actual validation benefit

### Important
Do NOT install every ML framework blindly.

Choose the smallest useful stack after EDA and baseline requirements are known.

---

# 11. Validation Strategy

## Status: IMPLEMENTED

Because the test set has no ground truth, validation is created from training data.

Plan:

```text
Training data
      │
      ├── train split
      │
      └── validation split
```

Important:

- Avoid leakage between training and validation.
- Reproduce the challenge's entity-level evaluation logic.
- Measure F_0.5.
- Track precision and recall separately.
- Track singleton performance separately.
- Evaluate blocking recall before model performance.

---

# 12. Threshold Optimization

## Status: PLANNED / NOT YET TUNED

The validation framework is available; threshold optimization itself has not yet been run. Because F_0.5 is precision-heavy:

- Do not automatically use 0.5 as the classification threshold.
- Test thresholds using validation data.
- Analyze false merges.
- Analyze missed matches.
- Analyze singleton predictions.
- Select threshold based on validation performance.

Final threshold must be documented.

---

# 13. Experiments

## Status: BLOCKING ITERATION COMPLETE FOR FIRST MODEL PASS

Every meaningful experiment should be recorded.

| ID | Blocking / baseline | Avg candidates/S1 | Blocking recall | Model | F0.5 | Status / Notes |
|---|---|---:|---:|---|---:|---|
| EXP-001 | Exact normalized name | 5.61 | 10.7590% | — | — | Complete; very selective |
| EXP-002 | Rare name tokens (2 rarest) | 18,766.29 | 77.6881% | — | — | Complete; diagnostic, too large |
| EXP-003 | Country + 1 rare token | 5,487.38 | 69.5862% | — | — | Complete; too much recall loss |
| EXP-004 | Country + 2 rare tokens | 14,834.23 | 77.6881% | — | — | Complete; 32.7B pairs, rejected |
| EXP-005 | Exact-name final-match baseline | 5.61 | 10.7590% | None — direct rule | **19.3429%** | Complete; validation sanity check |
| EXP-006 | Address blocking pilot | 4.76–26.63 | 8.22–64.25% | — | — | Complete; 100K-S1 pilot |
| EXP-007 | Practical multi-layer candidate generator v1 | **30.97** | **67.2492%** | — | — | **Frozen for first model pass; 10.25M validation pairs** |

### Experiment lessons

- Blocking quality must be evaluated using both **recall** and **candidate volume**.
- A high-recall blocker is not automatically usable if it produces an excessive number of pairwise comparisons.
- The current practical blocker trades some recall for a manageable 10.25M validation-pair workload.
- Pairwise ML scoring is the next major experiment.

Keep experiment scripts, configurations and results reproducible.

---

# 14. Submission Outputs

## Status: NOT IMPLEMENTED

Required:

```text
output/
├── matching_results.tsv
└── candidate_pairs.tsv
```

### matching_results.tsv
Requirements:

- Exactly one row per test S1 entity.
- Empty match list when there is no match.
- No duplicate IDs.
- Only S2/S3 IDs from the test data.
- Every test S1 entity must appear.

### candidate_pairs.tsv
Requirements:

- One row per test S1 entity.
- Contains final candidate set fed into the matching model.
- Final predicted matches must be a subset of candidates.

---

# 15. Official Validator

## Status: AVAILABLE / NOT YET INTEGRATED

Official validator:

```text
utils/validate_submission.py
```

Expected command:

```bash
python3 utils/validate_submission.py     --matching output/matching_results.tsv     --candidate output/candidate_pairs.tsv     --test-dir dataset/test
```

We should copy/include the official validator in the project and run it before every submission.

---

# 16. Leaderboard / Submission Management

## Status: NOT STARTED

Challenge allows a limited number of submissions per day.

Therefore:

- Do not waste submissions on formatting mistakes.
- Validate locally first.
- Maintain submission version history.
- Record public leaderboard score alongside experiment ID.
- Keep the final pipeline reproducible.

Suggested naming:

```text
SUB-001
SUB-002
SUB-003
...
```

---

# 17. Team Environment

## Current assumption

| Member | OS | Python |
|---|---|---|
| Vivek | Arch Linux | To verify/standardize |
| Member 2 | Windows | Python 3.12.6 |
| Member 3 | macOS | Python 3.12.x assumed |

### Environment goal

Standardize:

- Python 3.12
- package versions
- requirements.txt
- project structure
- reproducible commands

OS/IDE/hardware can remain different.

### Status
**Environment setup intentionally postponed until EDA determines actual dependencies.**

---

# 18. Compute Strategy

## Status: PLANNING

Dataset is large enough that memory efficiency matters.

Avoid:

```text
Load all 24M records into pandas at once
```

Prefer:

- chunked reading
- efficient data types
- indexes
- streaming/grouped processing
- disk-backed intermediate data when required
- candidate generation before expensive similarity calculations

GPU will only be introduced if an actual model benefits from it.

---

# 19. Immediate Next Tasks

## Priority 1 — Data profiling
**STATUS: COMPLETE**

Streaming profiling has been completed for all train/test source files and the training ground truth.

## Priority 2 — Validation framework
**STATUS: COMPLETE**

Entity-safe validation split, challenge-style macro F_0.5 scorer, error buckets, threshold-sweep infrastructure, and validation wrapper are implemented.

## Priority 3 — Blocking / candidate generation
**STATUS: COMPLETE AND FROZEN**

Validation blocking for the first model pass is frozen as **V2 ∪ F2**:

- V2: top-2 rare name-token blocking
- F2: address min-shared-token blocking with bounded postings

No further blocker experiments are planned for this submission pipeline.

### Validation final candidate set

```text
V2 candidates:             18,792,789
F2 candidates:             27,328,347
V2 ∪ F2 candidates:        45,321,035
Validation blocking recall: 88.2230%
```

## Priority 4 — Pairwise feature engineering
**STATUS: COMPLETE FOR VALIDATION**

The pairwise feature pipeline is implemented with 13 features:

```text
name_exact
name_core_exact
name_jaccard
name_overlap
name_ratio
address_exact
address_jaccard
address_overlap
address_ratio
address_number_equal
country_equal
name_address_mean
name_address_min
```

Validation feature generation completed for **45,321,035 pairs** with **0 missing records**.

## Priority 5 — First ML model
**STATUS: COMPLETE / FROZEN**

LightGBM pairwise model trained on the validation-union feature set:

```text
Training positives: 1,009,663
Training negatives: 3,028,989
Training rows:      4,038,652
Features:           13
```

Model artifact:

```text
output/pairwise_lgbm_union.txt
```

## Priority 6 — Threshold tuning
**STATUS: COMPLETE / FROZEN**

Validation threshold sweep established **0.95** as the selected threshold.

At threshold 0.95:

```text
Macro F0.5:    85.8274%
Macro precision: 91.2531%
Macro recall:    77.0325%
```

The model and threshold are now frozen for the test pipeline.

## Priority 7 — Test candidate pipeline
**STATUS: COMPLETE**

Test V2 and F2 candidate generation completed:

```text
Test S1:                 1,732,544
V2 pairs:              163,428,982
F2 pairs:              180,192,389
V2 ∪ F2 unique pairs:  338,406,680
Duplicates removed:      5,214,691
```

Final candidate union:

```text
output/candidate_pairs_test_union.tsv
```

Properties:

```text
Union file size:          4.08 GB
S1s with candidates:      1,724,832
S1s with zero candidates:     7,712
```

The 7,712 zero-candidate S1 entities must still appear in the final submission with an empty match list.

The completed union is also retained in:

```text
output/candidate_union_test.duckdb
```

DuckDB union database size observed: ~12 GB.

## Priority 8 — Test pairwise feature generation
**STATUS: PENDING**

The remaining heavy computation is pairwise feature generation for the **338.4M test candidate pairs**.

A validation timing reference is approximately:

```text
45.32M pairs → ~30 min
```

giving a rough test-scale estimate of several hours. This is suitable for an overnight run or can be moved to the Windows teammate's machine if hardware is preferable.

No test pairwise generation has been started yet.

## Priority 9 — Test scoring / prediction
**STATUS: PENDING**

Planned:

```text
test pairwise features
        ↓
frozen LightGBM model
        ↓
threshold 0.95
        ↓
matching_results.tsv
```

Final prediction writer must output **all 1,732,544 test S1 entities**, including the 7,712 with no candidate matches.

## Priority 10 — Final validation / submission
**STATUS: PENDING**

Run the official validator on:

```text
matching_results.tsv
candidate_pairs.tsv
```

Verify:

- exactly one row per test S1
- no duplicate predicted IDs
- predictions are S2/S3 IDs from test data
- every prediction is contained in the candidate set
- empty match lists are retained where appropriate

---

# 20. Current Overall Status

```text
Repository setup          ██████████ 100%  ✅
GitHub setup              ██████████ 100%  ✅
Dataset acquisition       ██████████ 100%  ✅
Dataset placement         ██████████ 100%  ✅
Dataset schema check      ██████████ 100%  ✅
Initial scale analysis    ██████████ 100%  ✅
Data profiling            ██████████ 100%  ✅

Validation framework      ██████████ 100%  ✅
Blocking                  ██████████ 100%  ✅  (V2 ∪ F2 frozen)
Feature engineering       ██████████ 100%  ✅  (validation complete)
Baseline model            ██████████ 100%  ✅  (LightGBM frozen)
Threshold optimization    ██████████ 100%  ✅  (0.95 frozen)
Test candidate generation ██████████ 100%  ✅
Test candidate union      ██████████ 100%  ✅  (338.4M pairs)

Test pairwise features    ░░░░░░░░░░   0%  ⬜  (next heavy job)
Test scoring              ░░░░░░░░░░   0%  ⬜
Final predictions         ░░░░░░░░░░   0%  ⬜
Final submission          ░░░░░░░░░░   0%  ⬜
```

# 21. Critical Rules — QUICK REFERENCE

1. **Do not push the dataset to GitHub.**
2. **Do not use external business/entity lookup.**
3. **Do not use geocoding or external address databases.**
4. **Do not hard-code only US/India.**
5. **Every test S1 must appear in the output.**
6. **Final matches must be a subset of candidate pairs.**
7. **Optimize for F_0.5, not accuracy alone.**
8. **Do not neglect singleton/no-match entities.**
9. **Validate locally before using a leaderboard submission.**
10. **Keep every serious experiment reproducible.**

---

# 22. Current Decision

**The validation/modeling pipeline is frozen. The test candidate set is complete.**

Current final test candidate set:

```text
V2 + F2 union
338,406,680 unique candidate pairs
1,724,832 S1 entities with candidates
7,712 S1 entities with zero candidates
```

The candidate union has been successfully materialized after an initial DuckDB OOM during unpartitioned final aggregation. The union itself was already complete; the final TSV was written safely using 16 hash partitions.

Current output:

```text
output/candidate_pairs_test_union.tsv
```

The remaining work is intentionally limited to:

1. Generate test pairwise features.
2. Score using the frozen LightGBM model.
3. Apply threshold 0.95.
4. Produce a submission containing every test S1.
5. Run the official submission validator.

The test pairwise feature job is the next major compute step and can be run overnight or transferred to the Windows teammate if that machine is more suitable.

**No further blocking experiments or model/threshold experimentation is planned unless a concrete implementation failure requires it.**

