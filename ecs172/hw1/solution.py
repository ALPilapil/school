import numpy as np
import pandas as pd
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize
import scipy.sparse as sp

DATA_DIR = "/home/aarpila/classes/ecs172/hw1"

# ─────────────────────────────────────────────
# Step 1: Load Data
# ─────────────────────────────────────────────

print("Loading data...")

train = pd.read_csv(f"{DATA_DIR}/train.csv")
test  = pd.read_csv(f"{DATA_DIR}/test.csv")
meta  = pd.read_csv(f"{DATA_DIR}/item_metadata.csv")

print(f"  train: {len(train)} rows | test: {len(test)} rows | items: {len(meta)} rows")


# ─────────────────────────────────────────────
# Step 2: Build Item Text Corpus
# ─────────────────────────────────────────────

print("Building item text corpus")

# Fill NaN values in text columns with empty strings.
# Columns to use: title, description, features, categories, main_category
meta_cols = list(meta.columns)
for col in meta_cols:
    meta[col] = meta[col].fillna("")

print("Concatenating data")
# For each item, concatenate the text columns into a single string.
meta['text'] = meta['title'] + " " + meta['description'] + " " + meta['features'] + " " + meta['categories'] + " " + meta['main_category'] + " " + meta['store'] + " " + meta['price'].astype(str) + " " + meta['average_rating'].astype(str)

# item_texts: list of strings, one per item, in the same order as meta
item_texts = meta['text'].tolist() 


# ─────────────────────────────────────────────
# Step 3: TF-IDF Vectorization
# ─────────────────────────────────────────────

print("Fitting TF-IDF vectorizer")

vectorizer = TfidfVectorizer(
    max_features=50_000,
    ngram_range=(1, 2),
    sublinear_tf=True,
    min_df=2,
)

# Fit the vectorizer on item_texts and transform to get item_tfidf.
item_tfidf = vectorizer.fit_transform(item_texts)
print("vecotrized shape: ", item_tfidf.shape)


# Build a lookup: item_id -> row index in item_tfidf
item_id_to_idx = {item_id: idx for idx, item_id in enumerate(meta["item_id"])}


# ─────────────────────────────────────────────
# Step 4: Build User Profiles
# ─────────────────────────────────────────────

print("Building user profiles...")

# Group training data by user
user_groups = train.groupby("user_id")

user_profiles = {}  # user_id -> sparse profile vector (1 x vocab_size)

for user_id, history in user_groups:

    # --- 4a. Item-content-based profile ---
    # TODO: For each row in history, look up the item's TF-IDF vector and
    # weight it by the rating. Sum them up and divide by total rating weight.
    #
    # Pseudocode:
    #   weighted_sum = sum(rating_i * item_tfidf[idx_i] for each item in history
    #                      if item_id is in item_id_to_idx)
    #   item_profile = weighted_sum / sum_of_weights
    weighted_sum = 0
    sum_of_weights = 0
    for _, row in history.iterrows():
        item = row['item_id']
        if item not in item_id_to_idx:
            continue
        item_idx = item_id_to_idx[item]
        item_vector = item_tfidf[item_idx]
        weighted_item_vector = row['rating'] * item_vector
        weighted_sum += weighted_item_vector
        sum_of_weights += row['rating']

    item_profile = weighted_sum / sum_of_weights  

    # --- 4b. Review-text-based profile ---
    # then transform it with the already-fitted vectorizer to get a sparse vector.
    # Hint: vectorizer.transform([combined_review_text])

    combined_review_list = history['review_text'].tolist()
    combined_review_text = "".join(combined_review_list)
    review_profile = vectorizer.transform([combined_review_text])

    # --- 4c. Blend the two profiles 50/50 ---
    # Handle the case where either could be None (e.g., all items missing from meta).
    # Fall back to whichever profile is available.
    if item_profile is not None and review_profile is not None:
        blended_profile = (item_profile + review_profile) / 2.0
    elif item_profile is None:
        blended_profile = review_profile
    else:
        blended_profile = item_profile

    user_profiles[user_id] = blended_profile  

# ─────────────────────────────────────────────
# Step 5: Score and Rank Test Candidates
# ─────────────────────────────────────────────

print("Ranking test candidates...")

# Build a lookup: average_rating per item (used as fallback for unknown users)
avg_rating = meta.set_index("item_id")["average_rating"].fillna(0).to_dict()

results = []  # list of dicts: {user_id, rank_1, ..., rank_5}

for user_id, candidates in test.groupby("user_id"):
    candidate_ids = candidates["item_id"].tolist()  # exactly 5 items

    profile = user_profiles.get(user_id)

    if profile is None:
        # Edge case: user has no training history — rank by average_rating
        ranked = sorted(candidate_ids, key=lambda iid: avg_rating.get(iid, 0), reverse=True)
    else:
        # Compute cosine similarity between the user profile and each candidate vector.
        # Sort candidate_ids by descending similarity score.
        #
        # Hint: cosine_similarity(profile, candidate_matrix) returns a (1 x n) array.
        # Items not in item_id_to_idx get similarity score 0.0 (rank last).

        similarity_score = []
        for candidate in candidate_ids:
            # get tf-idf vector
            if candidate in item_id_to_idx:
                candidate_idx = item_id_to_idx[candidate]
                candidate_vector = item_tfidf[candidate_idx]
                similarity_score.append(cosine_similarity(profile, candidate_vector))
            else:
                similarity_score.append(0.0)
            
        ranked = [cid for _, cid in sorted(zip(similarity_score, candidate_ids), reverse=True)]

    results.append({
        "user_id": user_id,
        "rank_1": ranked[0],
        "rank_2": ranked[1],
        "rank_3": ranked[2],
        "rank_4": ranked[3],
        "rank_5": ranked[4],
    })


# ─────────────────────────────────────────────
# Step 6: Validate and Write Submission
# ─────────────────────────────────────────────

submission = pd.DataFrame(results, columns=["user_id", "rank_1", "rank_2", "rank_3", "rank_4", "rank_5"])

assert len(submission) == 2000, f"Expected 2000 rows, got {len(submission)}"
assert set(submission["user_id"]) == set(test["user_id"]), "user_id mismatch"

out_path = f"{DATA_DIR}/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Submission written to {out_path} ({len(submission)} rows)")


# ─────────────────────────────────────────────
# (Optional) Validation — MRR & Hit@1
# ─────────────────────────────────────────────
#
# To validate before submitting:
#   1. Sort each user's train history by timestamp.
#   2. Hold out the last interaction as the "ground truth" item.
#   3. Pick 4 random other items as decoys.
#   4. Use your pipeline to rank all 5.
#   5. Check where the held-out item lands.
#
# Metrics:
#   Hit@1  = fraction of users where held-out item is rank_1
#   MRR    = mean of 1/rank of held-out item across all users
#
def validate():
    # Sort by timestamp, hold out last interaction per user
    train_sorted = train.sort_values("timestamp")
    last_idx = train_sorted.groupby("user_id").tail(1).index
    train_val = train_sorted.drop(index=last_idx)
    val = train_sorted.loc[last_idx]

    # Build user profiles on train_val (same logic as step 4)
    val_profiles = {}
    for user_id, history in train_val.groupby("user_id"):
        weighted_sum = 0
        sum_of_weights = 0
        for _, row in history.iterrows():
            item = row['item_id']
            if item not in item_id_to_idx:
                continue
            item_idx = item_id_to_idx[item]
            item_vector = item_tfidf[item_idx]
            weighted_sum += row['rating'] * item_vector
            sum_of_weights += row['rating']
        item_profile = weighted_sum / sum_of_weights if sum_of_weights > 0 else None

        combined_review_text = " ".join(history['review_text'].fillna("").tolist())
        review_profile = vectorizer.transform([combined_review_text])

        if item_profile is not None and review_profile is not None:
            blended = (item_profile + review_profile) / 2.0
        elif item_profile is None:
            blended = review_profile
        else:
            blended = item_profile

        val_profiles[user_id] = blended

    # Score and rank for each user
    all_item_ids = meta["item_id"].tolist()
    hit1_count = 0
    mrr_total = 0.0
    num_users = len(val)

    for _, row in val.iterrows():
        user_id = row['user_id']
        ground_truth = row['item_id']
        decoys = random.sample([iid for iid in all_item_ids if iid != ground_truth], 4)
        candidate_ids = [ground_truth] + decoys

        profile = val_profiles.get(user_id)

        if profile is None:
            ranked = sorted(candidate_ids, key=lambda iid: avg_rating.get(iid, 0), reverse=True)
        else:
            similarity_score = []
            for candidate in candidate_ids:
                if candidate in item_id_to_idx:
                    candidate_idx = item_id_to_idx[candidate]
                    candidate_vector = item_tfidf[candidate_idx]
                    similarity_score.append(float(cosine_similarity(profile, candidate_vector)[0, 0]))
                else:
                    similarity_score.append(0.0)
            ranked = [cid for _, cid in sorted(zip(similarity_score, candidate_ids), reverse=True)]

        rank = ranked.index(ground_truth) + 1
        if rank == 1:
            hit1_count += 1
        mrr_total += 1.0 / rank

    hit1 = hit1_count / num_users
    mrr = mrr_total / num_users
    print(f"Validation — Hit@1: {hit1:.4f} | MRR: {mrr:.4f}")

validate()
