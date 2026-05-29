# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Task

**ECS 172 HW3 — Hybrid Recommender: Top-10 Next-Item Prediction** on the Steam game catalog.

Goal: predict the 10 most likely next games for 10,000 test users. Scored on `0.5 * (Recall@10 + NDCG@10)`.

## Data Files

| File | Description |
|------|-------------|
| `train.csv` | 122,366 interactions — `user_id, item_id, playtime_minutes, timestamp` |
| `item_metadata.csv` | 32,132 items — `item_id, title, tags, genres, specs, developer, publisher, release_year, sentiment` |
| `test_users.csv` | 10,000 users to predict for |
| `sample_submission.csv` | Format: `user_id, rank_1, …, rank_10` |
| `evaluation_code.py` | Local eval: `evaluate(submission.csv, ground_truth.csv, train.csv)` |

## Evaluation

```bash
python evaluation_code.py submission.csv ground_truth.csv train.csv
python evaluation_code.py submission.csv ground_truth.csv   # without train filter
```

Validate format before submitting:
```python
from evaluation_code import validate_submission
validate_submission("submission.csv", "test_users.csv", "item_metadata.csv")
```

## Key Domain Notes

- **Local validation split**: hold out each user's **last 5** interactions by timestamp; train on the rest. Random splits won't correlate with leaderboard.
- **75% of catalog is cold** — never appears in training interactions. Pure CF cannot reach them; content-based retrieval is required.
- **Playtime is skewed** — use `log1p` if used as a feature.
- **Tags are imbalanced** — apply IDF or drop top tags (`Indie`, `Action`, `Adventure`).
- IDs are anonymized (`u_NNNNN` / `i_NNNNN`); never use raw Steam appids.
- Per-user Recall@10 only takes values in {0, 0.2, 0.4, 0.6, 0.8, 1.0} (denominator is always 5).

## Architecture

The system uses two retrieval pipelines merged before a final re-ranker:

1. **Content-based retrieval (sentence transformers)** — embed item text (title + tags + genres + specs) with a sentence-transformer model; for each user, average the embeddings of their history and retrieve nearest neighbors via cosine similarity.
2. **Collaborative filtering retrieval** — item-based CF using co-occurrence or matrix factorization; retrieves items similar to what the user has interacted with.
3. **Hybrid merge + re-rank** — combine candidate sets from both pipelines, score, and output top-10 per user.

Submission must have exactly one row per test user, 10 unique valid `item_id`s per row, no items outside `item_metadata.csv`.
