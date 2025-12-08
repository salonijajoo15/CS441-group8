#distilbert + pca + xgboost on idealogy_score instead of ideaology itself
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from transformers import DistilBertTokenizerFast, DistilBertModel
import torch
from tqdm import tqdm
import xgboost as xgb
import matplotlib.pyplot as plt
import os

# =========================================================
# 1. LOAD DATA
# =========================================================
df = pd.read_csv("combined_labeled_data.csv")

# Keep only valid ideology scores
df = df[df["ideology_score"].astype(str).str.replace(".", "", 1).str.replace("-", "", 1).str.isnumeric()]
df["ideology_score"] = df["ideology_score"].astype(float)

texts = df["comment"].tolist()
scores = df["ideology_score"].tolist()

# =========================================================
# 2. TRAIN/VAL/TEST SPLIT
# =========================================================
X_train, X_temp, y_train, y_temp = train_test_split(
    texts, scores, test_size=0.30, random_state=42
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=42
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# =========================================================
# 3. LOAD DISTILBERT
# =========================================================
tokenizer = DistilBertTokenizerFast.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()

# =========================================================
# 4. FUNCTION: GET EMBEDDING
# =========================================================
@torch.no_grad()
def get_embedding(text):
    tokens = tokenizer(
        text, truncation=True, padding="max_length", max_length=128, return_tensors="pt"
    )
    outputs = model(**tokens)
    return outputs.last_hidden_state[:, 0, :].squeeze().numpy()  # CLS token, 768-dim


# =========================================================
# 5. EMBEDDING CACHE FUNCTION
# =========================================================
def compute_or_load_embeddings(text_list, save_path):
    """
    Loads embeddings if file exists.
    Computes and saves them if file doesn't exist.
    """
    if os.path.exists(save_path):
        print(f"🔄 Loading cached embeddings: {save_path}")
        return np.load(save_path)

    print(f"⚙️ Generating embeddings for {len(text_list)} samples...")
    all_embeds = []

    for t in tqdm(text_list):
        emb = get_embedding(str(t))
        all_embeds.append(emb)

    all_embeds = np.array(all_embeds)

    np.save(save_path, all_embeds)
    print(f"💾 Saved embeddings → {save_path}")

    return all_embeds


# =========================================================
# 6. COMPUTE OR LOAD EMBEDDINGS
# =========================================================
train_emb = compute_or_load_embeddings(X_train, "train_emb.npy")
val_emb   = compute_or_load_embeddings(X_val,   "val_emb.npy")
test_emb  = compute_or_load_embeddings(X_test,  "test_emb.npy")

print("Embedding shapes:", train_emb.shape, val_emb.shape, test_emb.shape)

# =========================================================
# 7. PCA REDUCTION (100 components)
# =========================================================
pca = PCA(n_components=100)
train_pca = pca.fit_transform(train_emb)
val_pca   = pca.transform(val_emb)
test_pca  = pca.transform(test_emb)

print(f"PCA Explained Variance: {pca.explained_variance_ratio_.sum():.4f}")

# =========================================================
# 8. XGBOOST REGRESSOR
# =========================================================
model_xgb = xgb.XGBRegressor(
    n_estimators=600,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    reg_alpha=1.0,
    reg_lambda=1.0,
    objective="reg:squarederror"
)

model_xgb.fit(train_pca, y_train, eval_set=[(val_pca, y_val)], verbose=False)

# =========================================================
# 9. EVALUATION
# =========================================================
preds = model_xgb.predict(test_pca)

rmse = mean_squared_error(y_test, preds, squared=False)
mae = mean_absolute_error(y_test, preds)
r2 = r2_score(y_test, preds)

print("\n=== CONTINUOUS REGRESSION RESULTS ===")
print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R²:   {r2:.4f}")

# =========================================================
# 10. SCATTER PLOT TRUE vs PRED
# =========================================================
plt.figure(figsize=(7, 7))
plt.scatter(y_test, preds, alpha=0.4)
plt.xlabel("True Ideology Score")
plt.ylabel("Predicted Score")
plt.title("True vs Predicted Ideology (Regression)")
plt.grid()
plt.axline((0, 0), slope=1, color="red", linestyle="--")
plt.show()

# =========================================================
# 11. ERROR DISTRIBUTION
# =========================================================
errors = np.array(preds) - np.array(y_test)

plt.figure(figsize=(7, 5))
plt.hist(errors, bins=50, color="steelblue", alpha=0.7)
plt.title("Prediction Error Distribution")
plt.xlabel("Prediction Error")
plt.ylabel("Count")
plt.grid()
plt.show()

