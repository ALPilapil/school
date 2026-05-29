"""
Analysis script for the HW3 report.
Computes:
  1. Retrieval Recall@K on local validation split (K=merged pool size)
  2. Per-feature importance breakdown
  3. Cold-user behavior (users with <=3 training items)
"""

import argparse
import math
import html
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

# ── mirror constants from recommender.py ──────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"
EMBED_CACHE    = "item_embeddings.npy"
EMBED_ID_CACHE = "item_ids_order.npy"
ENCODE_BATCH   = 64
CB_CANDIDATES  = 150
CF_NEIGHBORS   = 100
CF_CANDIDATES  = 150
CB_BATCH_USERS = 500
RECENCY_LAMBDA = 0.3
HYBRID_ALPHA   = 0.6  # overridden by --alpha
MAX_HISTORY    = 50
TOP_N          = 10

item2idx_global = {}


# ── helpers (copied verbatim so we don't depend on main module state) ─────────

def _clean(val):
    if pd.isna(val) or val == "":
        return ""
    return html.unescape(str(val)).replace("|", " ")


def build_item_texts(meta_df):
    texts = []
    for _, row in meta_df.iterrows():
        parts = [
            _clean(row.get("title")),
            _clean(row.get("tags")),
            _clean(row.get("genres")),
            _clean(row.get("specs")),
            _clean(row.get("developer")),
            _clean(row.get("publisher")),
            _clean(row.get("sentiment")),
        ]
        texts.append(" ".join(p for p in parts if p))
    return texts


def encode_items(texts, item_ids):
    import os
    if os.path.exists(EMBED_CACHE) and os.path.exists(EMBED_ID_CACHE):
        cached_ids = np.load(EMBED_ID_CACHE, allow_pickle=True)
        if np.array_equal(cached_ids, item_ids):
            print("  Loading cached embeddings...")
            return np.load(EMBED_CACHE)
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    emb = model.encode(texts, batch_size=ENCODE_BATCH, normalize_embeddings=True,
                       show_progress_bar=True).astype(np.float32)
    np.save(EMBED_CACHE, emb)
    np.save(EMBED_ID_CACHE, item_ids)
    return emb


def build_user_vectors(train_df, item_embeddings, item2idx):
    user_vecs = {}
    for uid, group in train_df.groupby("user_id"):
        group = group.sort_values("timestamp").iloc[-MAX_HISTORY:]
        n = len(group)
        positions = np.arange(n)
        recency_rank = (n - 1 - positions).astype(float)
        play_weights = np.log1p(group["playtime_minutes"].values + 1.0)
        combined = play_weights * np.exp(-RECENCY_LAMBDA * recency_rank)
        if combined.sum() == 0:
            combined = np.ones(n)
        vecs, ws = [], []
        for (_, row), w in zip(group.iterrows(), combined):
            idx = item2idx.get(row["item_id"])
            if idx is None:
                continue
            vecs.append(item_embeddings[idx])
            ws.append(w)
        if not vecs:
            continue
        ws = np.array(ws, dtype=np.float64)
        uv = np.average(vecs, axis=0, weights=ws).astype(np.float32)
        norm = np.linalg.norm(uv)
        if norm > 0:
            uv /= norm
        user_vecs[uid] = uv
    return user_vecs


def get_cb_candidates(user_vecs, item_embeddings, item_ids, seen_sets, item2idx):
    users = list(user_vecs.keys())
    U = np.stack([user_vecs[u] for u in users], axis=0)
    E = item_embeddings
    cb_candidates = {}
    for start in range(0, len(users), CB_BATCH_USERS):
        batch_users = users[start:start + CB_BATCH_USERS]
        batch_U = U[start:start + CB_BATCH_USERS]
        sim = batch_U @ E.T
        for i, uid in enumerate(batch_users):
            row = sim[i].copy()
            seen = seen_sets.get(uid, set())
            for s in seen:
                sidx = item2idx.get(s)
                if sidx is not None:
                    row[sidx] = -2.0
            top_idxs = np.argpartition(row, -CB_CANDIDATES)[-CB_CANDIDATES:]
            top_idxs = top_idxs[np.argsort(row[top_idxs])[::-1]]
            cb_candidates[uid] = [(item_ids[j], float(row[j])) for j in top_idxs if row[j] > -1.5]
    return cb_candidates


def build_interaction_matrix(train_df):
    warm_items = train_df["item_id"].unique()
    warm2idx   = {iid: i for i, iid in enumerate(warm_items)}
    all_users  = train_df["user_id"].unique()
    user2idx   = {uid: i for i, uid in enumerate(all_users)}
    rows, cols, vals = [], [], []
    for _, row in train_df.iterrows():
        u  = user2idx.get(row["user_id"])
        it = warm2idx.get(row["item_id"])
        if u is not None and it is not None:
            rows.append(u); cols.append(it)
            vals.append(math.log1p(row["playtime_minutes"] + 1.0))
    R = csr_matrix((vals, (rows, cols)),
                   shape=(len(all_users), len(warm_items)), dtype=np.float32)
    return R, user2idx, warm2idx, warm_items


def build_item_item_sim(R):
    Rt = R.T.tocsr()
    row_norms = np.sqrt(np.array(Rt.power(2).sum(axis=1)).ravel())
    row_norms[row_norms == 0] = 1.0
    inv_norms = diags(1.0 / row_norms)
    Rt_norm = inv_norms @ Rt
    sim = (Rt_norm @ Rt_norm.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    return sim


def build_top_k_lookup(sim, k=CF_NEIGHBORS):
    n = sim.shape[0]
    top_k_idx    = np.zeros((n, k), dtype=np.int32)
    top_k_scores = np.zeros((n, k), dtype=np.float32)
    for i in range(n):
        row = sim[i]
        idxs = np.argpartition(row, -k)[-k:]
        idxs = idxs[np.argsort(row[idxs])[::-1]]
        top_k_idx[i]    = idxs
        top_k_scores[i] = row[idxs]
    return top_k_idx, top_k_scores


def get_cf_candidates(train_df, top_k_idx, top_k_scores, warm2idx, warm_items, seen_sets):
    cf_candidates = {}
    for uid, group in train_df.groupby("user_id"):
        seen   = seen_sets.get(uid, set())
        scores = defaultdict(float)
        for _, row in group.iterrows():
            widx = warm2idx.get(row["item_id"])
            if widx is None:
                continue
            item_weight = math.log1p(row["playtime_minutes"] + 1.0)
            for nidx, nscore in zip(top_k_idx[widx], top_k_scores[widx]):
                nitem = warm_items[nidx]
                if nitem not in seen:
                    scores[nitem] += item_weight * nscore
        if scores:
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:CF_CANDIDATES]
        else:
            sorted_items = []
        cf_candidates[uid] = sorted_items
    return cf_candidates


def merge_candidates(cb_list, cf_list, alpha=HYBRID_ALPHA):
    cb_dict = {iid: s for iid, s in cb_list}
    cf_dict = {iid: s for iid, s in cf_list}
    cf_max  = max((s for _, s in cf_list), default=1.0)
    if cf_max <= 0:
        cf_max = 1.0
    all_items = set(cb_dict) | set(cf_dict)
    final_scores = {}
    for iid in all_items:
        cb_s = max(0.0, cb_dict.get(iid, 0.0))
        cf_s = cf_dict.get(iid, 0.0) / cf_max
        final_scores[iid] = alpha * cf_s + (1.0 - alpha) * cb_s
    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)


# ─────────────────────────────────────────────────────────────────────────────

def retrieval_recall(merged_list, ground_truth_set, k):
    pool = {iid for iid, _ in merged_list[:k]}
    hits = pool & ground_truth_set
    return len(hits) / len(ground_truth_set) if ground_truth_set else 0.0


def main(alpha=HYBRID_ALPHA):
    print("=" * 60)
    print(f"alpha={alpha}  (CF weight={alpha:.2f}, CB weight={1-alpha:.2f})")
    print("Loading data...")
    train_full = pd.read_csv("train.csv")
    meta_df    = pd.read_csv("item_metadata.csv")
    item_ids   = meta_df["item_id"].values
    item2idx   = {iid: i for i, iid in enumerate(item_ids)}

    # ── Temporal split ────────────────────────────────────────────────────────
    train_full = train_full.sort_values(["user_id", "timestamp"])
    val_train_rows, val_test_rows = [], []
    for uid, grp in train_full.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        val_train_rows.append(grp.iloc[:-5])
        val_test_rows.append(grp.iloc[-5:])

    val_train = pd.concat(val_train_rows).reset_index(drop=True)
    val_test  = pd.concat(val_test_rows).reset_index(drop=True)

    # Ground-truth dict: user -> set of held-out items
    gt = val_test.groupby("user_id")["item_id"].apply(set).to_dict()
    seen_sets = val_train.groupby("user_id")["item_id"].apply(set).to_dict()

    # ── Build pipelines on val_train ──────────────────────────────────────────
    print("\n[1/3] Content-based pipeline...")
    texts      = build_item_texts(meta_df)
    embeddings = encode_items(texts, item_ids)
    user_vecs  = build_user_vectors(val_train, embeddings, item2idx)
    cb_cands   = get_cb_candidates(user_vecs, embeddings, item_ids, seen_sets, item2idx)
    print(f"  CB candidates built for {len(cb_cands)} users")

    print("\n[2/3] CF pipeline...")
    R, _, warm2idx, warm_items = build_interaction_matrix(val_train)
    print(f"  Interaction matrix: {R.shape[0]} users × {R.shape[1]} warm items")
    sim = build_item_item_sim(R)
    top_k_idx, top_k_scores = build_top_k_lookup(sim, k=CF_NEIGHBORS)
    del sim
    cf_cands = get_cf_candidates(val_train, top_k_idx, top_k_scores, warm2idx, warm_items, seen_sets)
    print(f"  CF candidates built for {len(cf_cands)} users")

    # ── Retrieval Recall@K ────────────────────────────────────────────────────
    print("\n[3/3] Computing retrieval Recall@K on validation set...")

    pool_sizes   = [10, 50, 100, 150, 200, 300]  # 300 is the full merged pool
    recall_table = {k: [] for k in pool_sizes}
    recall_cb    = {k: [] for k in pool_sizes}
    recall_cf    = {k: [] for k in pool_sizes}

    users_with_gt = [u for u in gt if u in cb_cands or u in cf_cands]
    print(f"  Evaluating {len(users_with_gt)} users...")

    history_lengths = val_train.groupby("user_id").size().to_dict()

    cold_user_stats = []  # users with <= 3 train items

    for uid in users_with_gt:
        gt_set  = gt.get(uid, set())
        if not gt_set:
            continue

        cb_list = cb_cands.get(uid, [])
        cf_list = cf_cands.get(uid, [])
        merged  = merge_candidates(cb_list, cf_list, alpha=alpha)

        for k in pool_sizes:
            recall_table[k].append(retrieval_recall(merged, gt_set, k))
            recall_cb[k].append(retrieval_recall(cb_list, gt_set, k))
            recall_cf[k].append(retrieval_recall(cf_list, gt_set, k))

        # Cold-user tracking
        hist_len = history_lengths.get(uid, 0)
        if hist_len <= 3:
            n_merged      = len(merged)
            n_cb          = len(cb_list)
            n_cf          = len(cf_list)
            recall_at_300 = retrieval_recall(merged, gt_set, 300)
            cold_user_stats.append({
                "user_id": uid, "hist_len": hist_len,
                "cb_pool": n_cb, "cf_pool": n_cf, "merged_pool": n_merged,
                "recall_at_300": recall_at_300,
            })

    print("\n" + "=" * 60)
    print("RETRIEVAL RECALL@K (merged pool):")
    print(f"{'Pool K':>8} | {'Hybrid':>8} | {'CB only':>8} | {'CF only':>8}")
    print("-" * 44)
    for k in pool_sizes:
        h  = np.mean(recall_table[k]) if recall_table[k] else 0
        c  = np.mean(recall_cb[k])    if recall_cb[k]    else 0
        cf = np.mean(recall_cf[k])    if recall_cf[k]    else 0
        print(f"{k:>8} | {h:>8.4f} | {c:>8.4f} | {cf:>8.4f}")

    pool_k = CB_CANDIDATES + CF_CANDIDATES  # 300
    hybrid_300 = np.mean(recall_table[300]) if recall_table[300] else 0
    print(f"\n  >> Merged pool fed to ranker: K={pool_k}, Recall@{pool_k}={hybrid_300:.4f}")

    # ── Feature summary (static analysis) ────────────────────────────────────
    print("\n" + "=" * 60)
    print("RANKING FEATURES:")
    print(f"""
  User features:
    - user_vec:       Weighted average of item embeddings over history
                      (weight = log1p(playtime) × exp(-{RECENCY_LAMBDA}×recency_rank))
    - hist_len:       Number of training interactions (implicit)

  Item features:
    - item_embedding: all-MiniLM-L6-v2 encoding of
                      title + tags + genres + specs + developer + publisher + sentiment
    - warm_flag:      Whether item appears in training interactions (cold vs warm)

  Interaction features:
    - cb_score:       cosine_sim(user_vec, item_embedding)  [0, 1]
    - cf_score_norm:  Σ_h [ log1p(play_h) × sim(h, item) ] / max  [0, 1]
    - hybrid_score:   {alpha:.2f} × cf_score_norm + {1-alpha:.2f} × cb_score

  Ranker:            Linear combination (no learned weights)
""")

    print("FEATURE IMPORTANCE (by design weights):")
    print(f"  CF score:  {alpha*100:.0f}%  (item-item co-occurrence similarity)")
    print(f"  CB score:  {(1-alpha)*100:.0f}%  (semantic embedding similarity)")
    print()
    print("  Sub-components of CB user vector weight:")
    print(f"    playtime (log1p):   drives magnitude")
    print(f"    recency decay:      exp(-{RECENCY_LAMBDA}×rank), rank 0=most recent")
    print(f"    both combined:      playtime × recency → normalized average")

    # ── Cold user analysis ────────────────────────────────────────────────────
    print("\n" + "=" * 60)
    print("COLD-USER ANALYSIS (users with ≤ 3 training interactions):")

    if cold_user_stats:
        cold_df = pd.DataFrame(cold_user_stats)
        print(f"\n  N cold users: {len(cold_df)}")
        print(f"  Avg CB pool size:     {cold_df['cb_pool'].mean():.1f}")
        print(f"  Avg CF pool size:     {cold_df['cf_pool'].mean():.1f}")
        print(f"  Avg merged pool size: {cold_df['merged_pool'].mean():.1f}")
        print(f"  Avg Recall@300:       {cold_df['recall_at_300'].mean():.4f}")

        by_hist = cold_df.groupby("hist_len").agg(
            n=("user_id", "count"),
            cb_pool=("cb_pool", "mean"),
            cf_pool=("cf_pool", "mean"),
            merged=("merged_pool", "mean"),
            recall=("recall_at_300", "mean"),
        )
        print("\n  Breakdown by history length:")
        print(f"  {'hist':>5} | {'N':>5} | {'CB pool':>8} | {'CF pool':>8} | {'merged':>8} | {'Rec@300':>8}")
        print("  " + "-" * 55)
        for hl, row2 in by_hist.iterrows():
            print(f"  {hl:>5} | {int(row2['n']):>5} | {row2['cb_pool']:>8.1f} | "
                  f"{row2['cf_pool']:>8.1f} | {row2['merged']:>8.1f} | {row2['recall']:>8.4f}")

        # Compare cold vs warm
        all_hist = {u: history_lengths.get(u, 0) for u in users_with_gt}
        warm_recalls = []
        for uid in users_with_gt:
            if history_lengths.get(uid, 0) > 3:
                gt_set = gt.get(uid, set())
                if gt_set:
                    merged = merge_candidates(cb_cands.get(uid, []), cf_cands.get(uid, []), alpha=alpha)
                    warm_recalls.append(retrieval_recall(merged, gt_set, 300))
        print(f"\n  Avg Recall@300 (warm users, hist>3): {np.mean(warm_recalls):.4f}")
        print(f"  Avg Recall@300 (cold users, hist≤3): {cold_df['recall_at_300'].mean():.4f}")
    else:
        print("  No cold users (hist ≤ 3) in val split.")

    # Cold behavior description
    print("""
  Cold-user behavior summary:
  - CB pipeline: user vector = weighted avg of ≤3 item embeddings
      → less stable direction, but still encodes semantic preference
      → CB pool is always filled (≈150 candidates from content similarity)
  - CF pipeline: scores each neighbor of ≤3 seed items
      → limited co-occurrence signal; CF pool may be sparse or empty
      → if CF pool is empty, fallback kicks in from CB alone
  - Merged pool: dominated by CB candidates; CF weight (0.6) is diluted
  - Final padding: if merged < 10 unique items, popularity fallback fills gaps
  - Net effect: cold users get reasonable content-based recs, but no
    personalized CF signal; quality degrades gracefully via CB + popularity.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type=float, default=HYBRID_ALPHA)
    args = parser.parse_args()
    main(alpha=args.alpha)
