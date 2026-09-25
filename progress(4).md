# Amazon ML Challenge 2026 — Progress Report

**Project:** Business Entity Resolution Challenge  
**Team:** 3 members  
**Repository:** `VivekA28/amazon-ml-challenge`  
**Last updated:** 25 September 2026 — preprocessing completed, validation framework + blocking experiments in progress

---

# 1. Challenge Understanding

## Problem

Build an ML pipeline that resolves business records across three independent noisy sources.

- Source 1 = deduplicated reference source.
- For every Source 1 entity, find all matching records from Source 2 and Source 3.
- A Source 1 entity may have zero, one, or multiple matches.
- Matching uses noisy business names, addresses, and country labels.

## Evaluation

- Metric: **F_0.5**
- Precision-heavy: precision is weighted more strongly than recall.
- Evaluation is macro-averaged per Source 1 entity.
- Correctly identifying a singleton/no-match entity earns full credit for that entity.
- False merges are particularly costly.

## Important restrictions

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

Current project structure includes:

```text
amazon-ml-challenge/
├── code/
│   └── business_entity_resolution/
│       ├── notebooks/
│       ├── requirements.txt
│       └── src/
│           ├── preprocess.py
│           └── ...
├── data/
│   ├── test/
│   └── train/
├── documentation/
├── experiments/
├── output/
├── processed/
├── .gitignore
└── README.md
