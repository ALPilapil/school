"""Regenerate submission.csv using BiasedBaseline + TruncatedSVD on full train.csv."""
from pathlib import Path
import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.decomposition import TruncatedSVD

HERE = Path(__file__).parent

train = pd.read_csv(HERE / "train.csv")
test  = pd.read_csv(HERE / "test_pairs.csv")

# --- Biased Baseline (best hyperparams from notebook: lam_u=25, lam_i=10) ---
mu    = train["rating"].mean()
lam_u = 25.0
lam_i = 10.0

us   = train.groupby("user_id")["rating"].agg(["sum", "count"])
b_u  = ((us["sum"] - us["count"] * mu) / (us["count"] + lam_u)).to_dict()

df2         = train.copy()
df2["resid"] = df2["rating"] - mu - df2["user_id"].map(b_u).fillna(0)
its  = df2.groupby("item_id")["resid"].agg(["sum", "count"])
b_i  = (its["sum"] / (its["count"] + lam_i)).to_dict()

def predict_bias(pairs: pd.DataFrame) -> np.ndarray:
    bu = pairs["user_id"].map(b_u).fillna(0).to_numpy()
    bi = pairs["item_id"].map(b_i).fillna(0).to_numpy()
    return np.clip(mu + bu + bi, 1.0, 5.0)

# --- TruncatedSVD (best k=20 from notebook) ---
users  = sorted(train["user_id"].unique())
items  = sorted(train["item_id"].unique())
u2idx  = {u: i for i, u in enumerate(users)}
i2idx  = {it: i for i, it in enumerate(items)}

df2["bias_pred"] = predict_bias(df2)
df2["resid_svd"] = df2["rating"] - df2["bias_pred"]
uidx = df2["user_id"].map(u2idx).to_numpy()
iidx = df2["item_id"].map(i2idx).to_numpy()
R    = sp.csr_matrix((df2["resid_svd"].to_numpy(), (uidx, iidx)),
                     shape=(len(users), len(items)))

print(f"Fitting TruncatedSVD (k=20) on {len(train):,} ratings ...")
svd  = TruncatedSVD(n_components=20, random_state=172)
U_k  = svd.fit_transform(R)   # (n_users, k)
VT_k = svd.components_         # (k, n_items)

def predict_svd(pairs: pd.DataFrame) -> np.ndarray:
    bp   = predict_bias(pairs)
    uidx = pairs["user_id"].map(u2idx)
    iidx = pairs["item_id"].map(i2idx)
    corr = np.zeros(len(pairs))
    mask = (uidx.notna() & iidx.notna()).to_numpy()
    if mask.any():
        ui = uidx[mask].astype(int).to_numpy()
        ii = iidx[mask].astype(int).to_numpy()
        corr[mask] = np.einsum("nk,kn->n", U_k[ui], VT_k[:, ii])
    return np.clip(bp + corr, 1.0, 5.0)

submission = test[["user_id", "item_id"]].copy()
submission["predicted_rating"] = predict_svd(test)

assert len(submission) == len(test)
assert set(submission.columns) == {"user_id", "item_id", "predicted_rating"}
assert submission["predicted_rating"].notna().all()
assert submission["predicted_rating"].between(1.0, 5.0).all()

submission.to_csv(HERE / "submission.csv", index=False)
print(f"Wrote {len(submission):,} predictions to submission.csv")
print(f"Range: {submission['predicted_rating'].min():.3f} – {submission['predicted_rating'].max():.3f}")
print(submission.head())
