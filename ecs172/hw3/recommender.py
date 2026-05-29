"""
HW3 Hybrid Recommender System
Two pipelines: content-based (sentence transformers) + item-based CF.
Usage:
  python recommender.py --mode submit    # generate submission.csv
  python recommender.py --mode validate  # local Recall@10 / NDCG@10
"""

import argparse
import html
import math
import os
import tempfile
from collections import defaultdict

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix, diags

# ── Constants ──────────────────────────────────────────────────────────────────
EMBED_MODEL    = "all-MiniLM-L6-v2"
EMBED_CACHE    = "item_embeddings.npy"
EMBED_ID_CACHE = "item_ids_order.npy"
ENCODE_BATCH   = 64
CB_CANDIDATES  = 150
CF_NEIGHBORS   = 100
CF_CANDIDATES  = 150
CB_BATCH_USERS = 500
RECENCY_LAMBDA = 0.3
HYBRID_ALPHA   = 0.6  # 60% CF, 40% CB
MAX_HISTORY    = 50   # cap history for CB user vector (avoids old history diluting signal)
TOP_N          = 10


# ── Section 1: Data Loading ────────────────────────────────────────────────────

def load_data(train_csv, meta_csv, test_users_csv):
    train_df   = pd.read_csv(train_csv)
    meta_df    = pd.read_csv(meta_csv)
    test_users = pd.read_csv(test_users_csv)["user_id"].tolist()

    # item_ids ordered consistently with meta_df rows
    item_ids = meta_df["item_id"].values
    item2idx = {iid: i for i, iid in enumerate(item_ids)}

    return train_df, meta_df, test_users, item_ids, item2idx


def build_seen_sets(train_df):
    return train_df.groupby("user_id")["item_id"].apply(set).to_dict()


def build_popularity_list(train_df, item_ids):
    counts = train_df["item_id"].value_counts()
    # sort by count desc, then item_id for determinism among ties
    all_items = pd.DataFrame({"item_id": item_ids})
    all_items["count"] = all_items["item_id"].map(counts).fillna(0).astype(int)
    all_items = all_items.sort_values(["count", "item_id"], ascending=[False, True])
    return all_items["item_id"].tolist()


# ── Section 2: Content-Based Pipeline ─────────────────────────────────────────

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


def encode_items(texts, item_ids, no_cache=False):
    if not no_cache and os.path.exists(EMBED_CACHE) and os.path.exists(EMBED_ID_CACHE):
        cached_ids = np.load(EMBED_ID_CACHE, allow_pickle=True)
        if np.array_equal(cached_ids, item_ids):
            print("Loading cached item embeddings...")
            return np.load(EMBED_CACHE)
        else:
            print("Cache item_id mismatch — re-encoding.")

    print(f"Encoding {len(texts)} items with {EMBED_MODEL}...")
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(EMBED_MODEL)
    embeddings = model.encode(
        texts,
        batch_size=ENCODE_BATCH,
        normalize_embeddings=True,
        show_progress_bar=True,
    ).astype(np.float32)
    np.save(EMBED_CACHE, embeddings)
    np.save(EMBED_ID_CACHE, item_ids)
    return embeddings


def build_user_vectors(train_df, item_embeddings, item2idx):
    user_vecs = {}
    for uid, group in train_df.groupby("user_id"):
        group = group.sort_values("timestamp")
        # limit to most recent MAX_HISTORY items
        group = group.iloc[-MAX_HISTORY:]
        n = len(group)
        positions = np.arange(n)
        recency_rank = (n - 1 - positions).astype(float)  # 0 = most recent
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
        user_vec = np.average(vecs, axis=0, weights=ws).astype(np.float32)
        norm = np.linalg.norm(user_vec)
        if norm > 0:
            user_vec /= norm
        user_vecs[uid] = user_vec
    return user_vecs


def get_cb_candidates(user_vecs, item_embeddings, item_ids, seen_sets):
    users = list(user_vecs.keys())
    U = np.stack([user_vecs[u] for u in users], axis=0)  # [N, 384]
    E = item_embeddings  # [32132, 384]

    cb_candidates = {}
    for start in range(0, len(users), CB_BATCH_USERS):
        batch_users = users[start:start + CB_BATCH_USERS]
        batch_U = U[start:start + CB_BATCH_USERS]
        sim = batch_U @ E.T  # [batch, 32132]

        for i, uid in enumerate(batch_users):
            row = sim[i]
            seen = seen_sets.get(uid, set())
            # mask seen items
            seen_idxs = [item2idx_global.get(s) for s in seen if s in item2idx_global]
            row_copy = row.copy()
            for sidx in seen_idxs:
                if sidx is not None:
                    row_copy[sidx] = -2.0
            top_idxs = np.argpartition(row_copy, -CB_CANDIDATES)[-CB_CANDIDATES:]
            top_idxs = top_idxs[np.argsort(row_copy[top_idxs])[::-1]]
            cb_candidates[uid] = [(item_ids[j], float(row_copy[j])) for j in top_idxs if row_copy[j] > -1.5]
    return cb_candidates


# Global item2idx reference (set before CB scoring)
item2idx_global = {}


# ── Section 3: Collaborative Filtering Pipeline ────────────────────────────────

def build_interaction_matrix(train_df):
    warm_items = train_df["item_id"].unique()
    warm2idx = {iid: i for i, iid in enumerate(warm_items)}
    all_users = train_df["user_id"].unique()
    user2idx = {uid: i for i, uid in enumerate(all_users)}

    rows, cols, vals = [], [], []
    for _, row in train_df.iterrows():
        u = user2idx.get(row["user_id"])
        it = warm2idx.get(row["item_id"])
        if u is not None and it is not None:
            rows.append(u)
            cols.append(it)
            vals.append(math.log1p(row["playtime_minutes"] + 1.0))

    R = csr_matrix(
        (vals, (rows, cols)),
        shape=(len(all_users), len(warm_items)),
        dtype=np.float32,
    )
    return R, user2idx, warm2idx, warm_items


def build_item_item_sim(R):
    print("Building item-item similarity matrix...")
    Rt = R.T.tocsr()  # [n_items, n_users]
    # row-normalize
    row_norms = np.sqrt(np.array(Rt.power(2).sum(axis=1)).ravel())
    row_norms[row_norms == 0] = 1.0
    inv_norms = diags(1.0 / row_norms)
    Rt_norm = inv_pnorms_mat(inv_norms, Rt)
    sim = (Rt_norm @ Rt_norm.T).toarray().astype(np.float32)
    np.fill_diagonal(sim, 0.0)
    return sim


def inv_pnorms_mat(inv_norms, Rt):
    return inv_norms @ Rt


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
    play_lookup = {}
    for _, row in train_df.iterrows():
        play_lookup.setdefault(row["user_id"], {})[row["item_id"]] = row["playtime_minutes"]

    cf_candidates = {}
    for uid, group in train_df.groupby("user_id"):
        seen = seen_sets.get(uid, set())
        scores = defaultdict(float)
        for _, row in group.iterrows():
            iid = row["item_id"]
            widx = warm2idx.get(iid)
            if widx is None:
                continue
            item_weight = math.log1p(row["playtime_minutes"] + 1.0)
            neighbors = top_k_idx[widx]
            neighbor_scores = top_k_scores[widx]
            for nidx, nscore in zip(neighbors, neighbor_scores):
                nitem = warm_items[nidx]
                if nitem not in seen:
                    scores[nitem] += item_weight * nscore

        if scores:
            sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:CF_CANDIDATES]
        else:
            sorted_items = []
        cf_candidates[uid] = sorted_items
    return cf_candidates


# ── Section 4: Hybrid Merge ────────────────────────────────────────────────────

def merge_candidates(cb_list, cf_list, alpha=HYBRID_ALPHA):
    cb_dict = {iid: score for iid, score in cb_list}
    cf_dict = {iid: score for iid, score in cf_list}

    cf_max = max((s for _, s in cf_list), default=1.0)
    if cf_max <= 0:
        cf_max = 1.0

    all_items = set(cb_dict) | set(cf_dict)
    final_scores = {}
    for iid in all_items:
        cb_s = max(0.0, cb_dict.get(iid, 0.0))  # cosine in [0,1] after normalization
        cf_s = cf_dict.get(iid, 0.0) / cf_max
        final_scores[iid] = alpha * cf_s + (1.0 - alpha) * cb_s

    return sorted(final_scores.items(), key=lambda x: x[1], reverse=True)


def predict_all(test_users, cb_candidates, cf_candidates, seen_sets, popularity_list,
                alpha=HYBRID_ALPHA, warm_items_set=None):
    pop_set = popularity_list  # ordered list, pre-computed

    rows = []
    for uid in test_users:
        seen = seen_sets.get(uid, set())
        cb_list = cb_candidates.get(uid, [])
        cf_list = cf_candidates.get(uid, [])

        merged = merge_candidates(cb_list, cf_list, alpha=alpha)
        filtered = [iid for iid, _ in merged if iid not in seen][:TOP_N]

        # pad with popularity fallback
        if len(filtered) < TOP_N:
            rec_set = set(filtered)
            for iid in pop_set:
                if len(filtered) >= TOP_N:
                    break
                if iid not in seen and iid not in rec_set:
                    filtered.append(iid)
                    rec_set.add(iid)

        assert len(filtered) == TOP_N, f"User {uid}: only {len(filtered)} recs"
        assert len(set(filtered)) == TOP_N, f"User {uid}: duplicate recs"
        rows.append([uid] + filtered)

    cols = ["user_id"] + [f"rank_{i}" for i in range(1, TOP_N + 1)]
    return pd.DataFrame(rows, columns=cols)


# ── Section 5: Validation Harness ─────────────────────────────────────────────

def run_validation(train_csv, meta_csv, alpha=HYBRID_ALPHA, no_cache=False):
    print("=== Running local validation (temporal holdout of last 5 interactions) ===")
    train_df = pd.read_csv(train_csv)
    meta_df  = pd.read_csv(meta_csv)

    # temporal split per user
    train_df = train_df.sort_values(["user_id", "timestamp"])
    val_train_rows, val_test_rows = [], []
    for uid, grp in train_df.groupby("user_id"):
        grp = grp.sort_values("timestamp")
        val_train_rows.append(grp.iloc[:-5])
        val_test_rows.append(grp.iloc[-5:])

    val_train_df = pd.concat(val_train_rows).reset_index(drop=True)
    val_test_df  = pd.concat(val_test_rows).reset_index(drop=True)

    test_users = val_train_df["user_id"].unique().tolist()

    with tempfile.TemporaryDirectory() as tmpdir:
        val_train_csv = os.path.join(tmpdir, "val_train.csv")
        val_gt_csv    = os.path.join(tmpdir, "val_gt.csv")
        val_sub_csv   = os.path.join(tmpdir, "val_submission.csv")

        val_train_df.to_csv(val_train_csv, index=False)
        val_test_df[["user_id", "item_id"]].to_csv(val_gt_csv, index=False)

        submission = run_pipeline(
            train_csv=val_train_csv,
            meta_csv=meta_csv,
            test_users_override=test_users,
            alpha=alpha,
            no_cache=no_cache,
        )
        submission.to_csv(val_sub_csv, index=False)

        from evaluation_code import evaluate
        evaluate(val_sub_csv, val_gt_csv, val_train_csv)


# ── Section 6: Main Pipeline ───────────────────────────────────────────────────

def run_pipeline(train_csv, meta_csv, test_users_csv=None, test_users_override=None,
                 alpha=HYBRID_ALPHA, no_cache=False):
    global item2idx_global

    train_df, meta_df, test_users_from_file, item_ids, item2idx = load_data(
        train_csv, meta_csv, test_users_csv or "test_users.csv"
    )
    if test_users_override is not None:
        test_users = test_users_override
    else:
        test_users = test_users_from_file

    item2idx_global = item2idx

    seen_sets      = build_seen_sets(train_df)
    popularity_list = build_popularity_list(train_df, item_ids)

    # Content-based
    print("\n[1/4] Building content-based pipeline...")
    texts      = build_item_texts(meta_df)
    embeddings = encode_items(texts, item_ids, no_cache=no_cache)
    print("Building user vectors...")
    user_vecs  = build_user_vectors(train_df, embeddings, item2idx)
    print(f"  User vectors built: {len(user_vecs)}")
    print("Computing CB candidates...")
    cb_cands   = get_cb_candidates(user_vecs, embeddings, item_ids, seen_sets)

    # Collaborative filtering
    print("\n[2/4] Building collaborative filtering pipeline...")
    R, user2idx_cf, warm2idx, warm_items = build_interaction_matrix(train_df)
    print(f"  Interaction matrix: {R.shape[0]} users × {R.shape[1]} warm items")
    sim = build_item_item_sim(R)
    print("  Computing top-K item neighbors...")
    top_k_idx, top_k_scores = build_top_k_lookup(sim, k=CF_NEIGHBORS)
    del sim  # free 258 MB
    print("  Computing CF candidates...")
    cf_cands = get_cf_candidates(train_df, top_k_idx, top_k_scores, warm2idx, warm_items, seen_sets)

    # Hybrid merge
    print("\n[3/4] Merging and ranking...")
    warm_items_set = set(warm_items)
    submission = predict_all(test_users, cb_cands, cf_cands, seen_sets, popularity_list,
                             alpha=alpha, warm_items_set=warm_items_set)

    print(f"\n[4/4] Done. {len(submission)} users × {TOP_N} items.")
    return submission


def main():
    parser = argparse.ArgumentParser(description="HW3 Hybrid Recommender")
    parser.add_argument("--mode", choices=["submit", "validate"], default="submit")
    parser.add_argument("--train",      default="train.csv")
    parser.add_argument("--meta",       default="item_metadata.csv")
    parser.add_argument("--test_users", default="test_users.csv")
    parser.add_argument("--output",     default="submission.csv")
    parser.add_argument("--alpha",      type=float, default=HYBRID_ALPHA,
                        help="Weight on CF score (1-alpha goes to CB). Default=0.5")
    parser.add_argument("--no_cache",   action="store_true",
                        help="Re-encode item embeddings even if cache exists")
    args = parser.parse_args()

    if args.mode == "validate":
        run_validation(args.train, args.meta, alpha=args.alpha, no_cache=args.no_cache)
    else:
        submission = run_pipeline(
            train_csv=args.train,
            meta_csv=args.meta,
            test_users_csv=args.test_users,
            alpha=args.alpha,
            no_cache=args.no_cache,
        )
        submission.to_csv(args.output, index=False)

        from evaluation_code import validate_submission
        validate_submission(args.output, args.test_users, args.meta)
        print(f"Saved → {args.output}")


if __name__ == "__main__":
    main()
