# distilbert_rf_classifier.py
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt

# ===========================
# 1. LOAD & PREP DATA
# ===========================
df = pd.read_csv("combined_labeled_data.csv")

df["ideology_score"] = pd.to_numeric(df["ideology_score"], errors="coerce")
df = df.dropna(subset=["ideology_score", "comment"])

engagement_features = ["post_score", "post_upvotes", "post_downvotes", "comment_score"]
df[engagement_features] = df[engagement_features].fillna(0)

df["text_combined"] = (
    df["comment"].fillna("") + " " +
    df["post_title"].fillna("") + " " +
    df["post_body"].fillna("") + " " +
    df["subreddit"].fillna("")
)

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ===========================
# 2. DISTILBERT EMBEDDINGS
# ===========================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")

def get_embeddings(texts, batch_size=16):
    embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i+batch_size].tolist()
        tokens = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=256)
        with torch.no_grad():
            outputs = model(**tokens)
        batch_embeds = outputs.last_hidden_state[:, 0, :].numpy()
        embeddings.append(batch_embeds)
    return np.vstack(embeddings)

print("Generating DistilBERT embeddings...")
train_embeddings = get_embeddings(train_df["text_combined"])
val_embeddings = get_embeddings(val_df["text_combined"])
test_embeddings = get_embeddings(test_df["text_combined"])

X_train = np.hstack([train_embeddings, train_df[engagement_features].values])
X_val = np.hstack([val_embeddings, val_df[engagement_features].values])
X_test = np.hstack([test_embeddings, test_df[engagement_features].values])

y_train, y_val, y_test = train_df["ideology_score"], val_df["ideology_score"], test_df["ideology_score"]

# ===========================
# 3. RANDOM FOREST REGRESSOR
# ===========================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    min_samples_split=5,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

y_pred = rf.predict(X_test)

# ===========================
# 4. EVALUATION
# ===========================
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("True Ideology Score")
plt.ylabel("Predicted Ideology Score")
plt.title("DistilBERT + Engagement + Random Forest — Continuous Ideology Prediction")
plt.grid(True)
plt.show()
