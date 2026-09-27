# Business Entity Resolution — ML Challenge (72-Hour Hackathon)

Matches business records across three noisy, independently-sourced datasets using only
`business_name`, `business_address`, and `country` — no shared identifiers, no external
data lookups. Built for a 72-hour hackathon with a ~11.7M-record dataset and a
precision-weighted (F0.5) evaluation metric.

**Validation results:** Precision **0.968** · Recall **0.809** · **F0.5 = 0.931** · ROC-AUC **0.991**

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Approach](#approach)
- [Results](#results)
- [Output Visualizations](#output-visualizations)
- [Project Structure](#project-structure)
- [Setup & Installation](#setup--installation)
- [How to Run](#how-to-run)
- [Design Decisions](#design-decisions--trade-offs)
- [Compliance](#compliance--fair-play)
- [Full Methodology](#full-methodology)

---

## Problem Overview

Given business records from three independent sources — Source 1 (deduplicated reference),
Source 2, and Source 3 — with noisy, inconsistent name/address fields and no shared keys,
determine which Source 2/3 records refer to the same real-world business entity as each
Source 1 record. A Source 1 entity may match zero, one, or many candidate records.

| Source | Rows | India | US | France |
|---|---|---|---|---|
| Source 1 (reference) | 1,732,544 | 809,986 | 663,106 | 259,452 |
| Source 2 | 4,887,273 | 2,312,565 | 1,871,330 | 703,378 |
| Source 3 | 5,082,316 | 2,405,000 | 1,945,701 | 731,615 |

**Total scale: ~11.7M records.** A naive pairwise comparison of Source 1 against Source 2 + 3
would require ~17 trillion comparisons — infeasible within the time budget, making blocking
strategy the primary scalability constraint.

**Evaluation metric:** F0.5, which weights precision 2× over recall. Correctly predicting
"no match" (a singleton) scores full credit; any false merge is penalized twice as hard as
a missed match. The solution is deliberately tuned to be conservative rather than exhaustive.

**Hard constraint:** No external APIs, geocoding services, or business registry lookups at
any stage. Any ML model used must be MIT/Apache-2.0 licensed and ≤8B parameters.

---

## Approach

```
Raw records
     │
     ▼
Normalization        Unicode NFKC · legal-suffix canonicalization · address abbreviation expansion
     │
     ▼
Blocking             Country-sharded key-bucketing (sorted name tokens + Metaphone phonetic code)
     │
     ▼
Feature Engineering  Fuzzy ratio, Jaro-Winkler, phonetic match, country match, name length delta
     │
     ▼
Classification        LightGBM (Apache-2.0) + hard-negative mining
     │
     ▼
Threshold Tuning     Decision threshold selected by maximizing F0.5 on held-out validation
     │
     ▼
Conflict Resolution  Greedy highest-confidence-first assignment (preserves one-to-many matching)
     │
     ▼
matching_results.tsv + candidate_pairs.tsv
```

**Key design choices:**
- **Blocking** uses deterministic key-bucketing (no embeddings/vector index) — scales linearly
  to 11.7M records on CPU alone, with no GPU dependency.
- **Model** is a single LightGBM classifier refined via one round of hard-negative mining — a
  stacked ensemble was built and tested but *rejected* after it underperformed on validation
  F0.5 (0.926 vs. 0.931); see [Design Decisions](#design-decisions--trade-offs).
- **Threshold** is chosen by directly optimizing F0.5, not the default 0.5 cutoff, since the
  metric weights precision twice as heavily as recall.
- **Conflict resolution** uses greedy assignment rather than bipartite matching, since the
  latter enforces 1-to-1 pairing, which would violate the problem's one-to-many matching rule.

---

## Results

| Metric | Score |
|---|---|
| Precision | 0.968 |
| Recall | 0.809 |
| **F0.5 (competition metric)** | **0.931** |
| ROC-AUC | 0.991 |

All metrics computed on a held-out 20% validation split of the training data, stratified by
label, not seen during training or hard-negative mining.

---

## Output Visualizations

### Validation Confusion Matrix & Precision-Recall Curve

![Validation Performance](assets/validation_performance.png)

Confusion matrix at the selected threshold (0.917) and the precision-recall curve, with the
chosen F0.5-optimal operating point marked.

### Feature Importance

![Feature Importance](assets/feature_importance.png)

`name_fuzzy`, `name_jaro`, and `addr_fuzzy` dominate, confirming string similarity as the
primary match signal. `country_match` shows near-zero importance — expected, since blocking
is already country-sharded, so the feature has no variance within a shard.

### Predicted Match Distribution (Final Test Set Submission)

![Match Distribution](assets/final_match_distribution.png)

~50.8% of Source 1 test entities predicted as singletons, consistent with the F0.5 metric's
incentive to avoid false merges when evidence is weak.

---

## Project Structure

```
.
├── README.md                          ← you are here
├── assets/                            ← output charts embedded above
│   ├── validation_performance.png
│   ├── feature_importance.png
│   └── final_match_distribution.png
├── Documentation_template.docx        ← full methodology write-up (candidate generation,
│                                          feature engineering, model architecture, compliance)
├── output/
│   ├── matching_results.tsv           ← final matches (leaderboard submission)
│   └── candidate_pairs.tsv            ← blocking candidate set fed to the model
└── code/
    └── business_entity_resolution/
        ├── README.md
        ├── requirements.txt
        └── src/
            └── entity_resolution_pipeline.py   ← full runnable pipeline
```

---

## Setup & Installation

```bash
git clone <this-repo-url>
cd <repo-folder>
pip install -r code/business_entity_resolution/requirements.txt
```

**Dependencies:** `pandas`, `numpy`, `scikit-learn`, `lightgbm`, `rapidfuzz`, `jellyfish`, `joblib`

---

## How to Run

Expects challenge data at `dataset/train/` and `dataset/test/` relative to the working
directory (matching the challenge's provided folder structure):

```bash
python code/business_entity_resolution/src/entity_resolution_pipeline.py
```

This runs the full pipeline end-to-end: loads train/test data, trains the model with
hard-negative mining, tunes the F0.5-optimal threshold, and writes both `matching_results.tsv`
and `candidate_pairs.tsv` to `output/`.

---

## Design Decisions & Trade-offs

- **Embedding-based blocking (sentence-transformers + FAISS) was prototyped and replaced**
  with key-bucketing — comparable recall at a fraction of the compute/infrastructure cost,
  and removes any GPU dependency for reproducibility.
- **A stacked ensemble (GBM + logistic meta-learner) was tested and discarded** — it
  underperformed the single hard-negative-mined model on validation F0.5 (0.926 vs. 0.931).
  Reported transparently as evidence that model complexity was evaluated empirically, not
  assumed to help.
- **Hard-negative mining** (retraining on the model's own high-confidence errors) improved
  precision from 0.965 → 0.968 with only a marginal recall trade-off.

---

## Compliance & Fair Play

- ✅ No external APIs, geocoding services, or business registries used at any stage.
- ✅ No commercial entity-resolution services used.
- ✅ All address normalization implemented as local, rule-based string transformations.
- ✅ Model used (LightGBM) is Apache-2.0 licensed, with a negligible parameter count —
  well within the 8B-parameter constraint.
- ✅ All training signal derived exclusively from the provided `train_source1/2/3.tsv` and
  `train_ground_truth.tsv`.

---

## Full Methodology

See [`Documentation_template.docx`](./Documentation_template.docx) for the complete
methodology write-up, covering:
- Candidate generation / blocking strategy (with recall-ceiling validation methodology)
- Feature engineering rationale
- Model architecture, hard-negative mining, and the ensemble rejection analysis
- Threshold selection and conflict resolution reasoning
- Full fair-play compliance statement
