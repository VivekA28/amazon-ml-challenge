# Amazon ML Challenge 2026 — Progress Report

**Project:** Business Entity Resolution Challenge  
**Team:** 3 members  
**Repository:** `VivekA28/amazon-ml-challenge`  
**Last updated:** 25 September 2026 — validation framework + blocking experiments updated

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
- ML/modeling
- Pipeline integration
- Final system
- Final submission

### Member 2 — Mac
**Primary ownership:**
- Data analysis / EDA
- Normalization
- Blocking/candidate-generation experiments

### Member 3 — Windows
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

## Status: INITIAL BLOCKING EXPERIMENTS COMPLETE — ITERATION IN PROGRESS

Blocking is now being tested empirically on the full training ground truth.

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

Conclusion: exact normalized-name blocking is very selective but misses most true matches. It is useful as a possible complementary blocking layer, not as the only blocker.

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

Conclusion: token blocking increases blocking recall dramatically (**10.76% → 77.69%**) but creates an impractically large candidate set (~18.8K candidates/S1). It is therefore a diagnostic result rather than a final candidate-generation strategy.

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

Conclusion: country restriction reduces candidate volume substantially compared with EXP-002 (18,766.29 → 5,487.38 candidates/S1), but using only one token reduces blocking recall (77.6881% → 69.5862%). This is a promising efficiency trade-off but not sufficient as the sole blocker.

### Current interpretation

```text
Exact normalized name
    ↓
Very low candidate volume
    ↓
10.76% recall

Rare name tokens (2)
    ↓
77.69% recall
    ↓
18,766 candidates/S1

Country + rarest name token (1)
    ↓
69.59% recall
    ↓
5,487 candidates/S1
```

The next blocking experiment should test **country + two rarest usable tokens** to determine whether recall can be recovered while keeping candidate volume below the unrestricted two-token blocker.

### EXP-004 — Country-aware + 2 rare tokens

**Status: RUNNING**

Configuration:

- Country-aware token index
- Minimum token length: 4 characters
- Maximum token frequency: 50,000
- Two rarest usable name tokens per Source-1 record
- Candidate retrieval uses OR/union across the two tokens
- Candidates restricted to the same country
- Evaluation against the complete training ground truth

Purpose: determine whether the second rare token recovers the recall lost in EXP-003 while keeping candidate volume below EXP-002.

Results are **not recorded yet** because the experiment is still running.

### Current blocking comparison

| Experiment | Strategy | Avg candidates/S1 | Blocking recall | Runtime | Status |
|---|---|---:|---:|---:|---|
| EXP-001 | Exact normalized name | 5.61 | 10.7590% | — | Complete |
| EXP-002 | Rare name tokens (2) | 18,766.29 | 77.6881% | ~63.2 min | Complete |
| EXP-003 | Country + 1 rare token | 5,487.38 | 69.5862% | ~19.2 min | Complete |
| EXP-004 | Country + 2 rare tokens | TBD | TBD | Running | Running |
| EXP-005 | Exact normalized name → all candidates emitted as matches (no filtering) | n/a (matching metric, not blocking) | n/a (matching metric, not blocking) | untimed | Complete — first real end-to-end VAL-split F_0.5: 19.3429%. Precision 25.8305%, recall 15.7295% (macro-averaged). Singletons scored much higher (67.9793%, n=18,588) than non-singletons (16.4493%, n=312,435), as expected — with zero filtering, exact-name matches are either exactly right or noisy, and most singletons correctly got zero candidates. Sanity-checks the scorer on real imperfect output; not a real model. |

### Blocking ideas still to investigate

- Exact normalized address components
- Address-number + name-token blocks
- Character n-gram based blocking
- Multiple complementary blocking rules
- Approximate/phonetic blocks where useful
- Progressive/multi-stage candidate generation

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

## Status: BLOCKING EXPERIMENTS STARTED

Every meaningful experiment should be recorded.

| ID | Blocking | Avg candidates/S1 | Blocking recall | Model | F0.5 | Notes |
|---|---|---:|---:|---|---:|---|
| EXP-001 | Exact normalized name | 5.61 | 10.7590% | — | — | Very selective; insufficient recall alone |
| EXP-002 | Rare name tokens (2 rarest) | 18,766.29 | 77.6881% | — | — | High recall but candidate explosion; diagnostic only |

| EXP-005 | Exact normalized name (EXP-001's blocker) | 5.61 | 10.7590% (micro, train, from EXP-001) | None — every candidate emitted as a final match, no filtering | 19.3429% | First real end-to-end F_0.5 on VAL split (331,023 entities). Precision 25.83%, recall 15.73% (macro). Singleton mean F_0.5: 67.98% (n=18,588). Non-singleton mean F_0.5: 16.45% (n=312,435). Confirms the scorer behaves correctly on real, imperfect predictions — trivial rule-based baseline, not a candidate for the final pipeline. |

### Experiment lessons

- Blocking quality must be evaluated using both **recall** and **candidate volume**.
- A high-recall blocker is not automatically usable if it produces an excessive number of pairwise comparisons.
- Multiple complementary blockers are likely to be preferable to relying on one broad blocker.
- Pairwise ML scoring has not started yet.

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

Observed key findings:

- Source 2 and Source 3 contain missing addresses; names and countries are present.
- Test contains France in addition to US and India.
- Training ground truth contains 2,206,821 S1 entities.
- 123,247 training S1 entities (5.58%) have zero matches.
- 94.42% have at least one match.
- Most matched S1 entities have multiple S2/S3 matches; the maximum observed match count is 11.

## Priority 2 — Validation framework
**STATUS: IMPLEMENTED — SANITY CHECK PENDING**

Mac implemented an entity-safe validation split and F_0.5 scorer. Source-1 entities are split into train/validation while the full Source-2/Source-3 pools remain available to both splits.

Implemented scripts:
- `create_validation_split.py`
- `score_f05.py`

The scorer computes per-S1 precision, recall and F_0.5, then macro-averages them, with special handling for no-match/singleton entities.

Next: run the validation pipeline end-to-end and use it for blocker/model experiments.

## Priority 3 — Blocking refinement
**IN PROGRESS — EXP-004 RUNNING**

Current results:

- Exact normalized name: 10.7590% recall, 5.61 candidates/S1.
- Rare-token name blocking (2 tokens): 77.6881% recall, 18,766.29 candidates/S1.
- Country + rarest token (1 token): 69.5862% recall, 5,487.38 candidates/S1.

Next: test country + two rare tokens, then combine complementary blockers while controlling candidate volume.

## Priority 4 — Similarity baseline

Implement classical string similarity features.

## Priority 5 — First ML model

Train a pairwise matching model.

## Priority 6 — Threshold tuning

Optimize for validation F_0.5.

## Priority 7 — Error analysis
### Error Analysis — EXP-005 (preliminary)

Categorized `val_errors.tsv` (331,023 val entities) into failure buckets
using `error_analysis_exp005.py`. Full breakdown in
`experiments/val_errors_categorized.tsv`.

| Category | Count | % of total | Mean F_0.5 |
|---|---:|---:|---:|
| nonsingleton_total_miss | 145,718 | 44.0% | 0.000 |
| nonsingleton_mixed_errors | 100,075 | 30.2% | 0.068 |
| nonsingleton_under_matched | 63,481 | 19.2% | 0.667 |
| singleton_correct | 12,636 | 3.8% | 1.000 |
| singleton_false_positive | 5,952 | 1.8% | 0.000 |
| nonsingleton_correct | 1,936 | 0.6% | 1.000 |
| nonsingleton_over_matched | 1,225 | 0.4% | 0.254 |

**Key findings:**
- 44% of non-singleton entities received zero candidates from exact-name
  blocking — this is a pure blocking-recall gap, not fixable by feature
  engineering. Directly consistent with EXP-001's 10.76% pooled recall.
- India shows a higher rate of total misses (48.6% vs US 45.3%) and
  mixed errors (36.2% vs US 29.3%), and a higher singleton
  false-positive rate (35.4% vs US 29.8%) — consistent with the
  challenge's documented transliteration/DBA-name noise being more
  pronounced in Indian records.
- The US shows relatively more under-matched (24.04% vs India 14.75%)
  and over-matched entities — once a candidate is found, US errors lean
  toward generic/chain-name precision confusion rather than complete misses.
- Implication for Priority 4: address- and country-weighted features are
  needed, not just name similarity — 1.8% of true singletons already get
  a false match on name alone.
- Implication for Priority 3: EXP-003/EXP-004's country-aware token
  blocking should specifically target the `nonsingleton_total_miss`
  bucket. Once EXP-004 finishes, its recall should be checked against
  this same bucket, not just the full-training-set aggregate.
Study:
- false positives
- false negatives
- singleton mistakes
- country-specific behavior
- difficult names
- difficult addresses

## Priority 8 — Improvements

Test additional blocking/features/embeddings only where validation indicates a benefit.

## Priority 9 — Final pipeline

Generate:

```text
matching_results.tsv
candidate_pairs.tsv
```

## Priority 10 — Submission validation

Run official validator and package the final reproducible solution.

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

Validation framework      ████████░░  80%  🟡
Blocking                  ████░░░░░░  40%  🟡
Feature engineering       ░░░░░░░░░░   0%  ⬜
Baseline model            ░░░░░░░░░░   0%  ⬜
Threshold optimization    ░░░░░░░░░░   0%  ⬜
Error analysis            ░░░░░░░░░░   0%  ⬜
Final pipeline             ░░░░░░░░░░   0%  ⬜
Submission                 ░░░░░░░░░░   0%  ⬜
```

---

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

**We are NOT choosing the final ML model yet.**

Data profiling is complete and blocking experiments have begun.

Current evidence:

> Exact normalized-name blocking gives low candidate volume but only 10.76% recall. Two-token rare-name blocking reaches 77.69% recall but produces ~18.8K candidates/S1. Country + one rare token reduces this to ~5.49K candidates/S1, but recall falls to 69.59%.

Validation infrastructure is now implemented, including an entity-safe S1 split and challenge-style F_0.5 scorer.

Therefore, the next technically correct steps are:

> **Run the validation pipeline end-to-end, test country + two rare tokens, then move into pairwise feature engineering and a baseline matching model.**

The team should not treat either current blocker as the final solution yet.
