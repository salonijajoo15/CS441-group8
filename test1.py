#distilbert + pca + xgboost
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

#load data
df = pd.read_csv("combined_labeled_data.csv")

# clean ideology score
df["ideology_score"] = pd.to_numeric(df["ideology_score"], errors="coerce")
df = df.dropna(subset=["comment", "ideology_score"])

# Combine text fields for embedding
df["text_combined"] = (
    df["comment"].fillna("") + " " +
    df["post_title"].fillna("") + " " +
    df["post_body"].fillna("") + " " +
    df["subreddit"].fillna("")
)

# 70/15/15 split
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print(f"Train = {len(train_df)}, Val = {len(val_df)}, Test = {len(test_df)}")

#distilbert embedding set up
from transformers import DistilBertTokenizer, DistilBertModel
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Using device:", device)

tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
bert_model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)

# function to embed text
def embed_texts(texts):
    embeddings = []

    for t in tqdm(texts, desc="Embedding"):
        encoded = tokenizer(
            t,
            padding="max_length",
            truncation=True,
            max_length=128,
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            output = bert_model(**encoded).last_hidden_state[:, 0, :]  # CLS token

        embeddings.append(output.cpu().numpy().flatten())

    return np.array(embeddings)

#generate embedings

X_train_text = embed_texts(train_df["text_combined"].tolist())
X_val_text   = embed_texts(val_df["text_combined"].tolist())
X_test_text  = embed_texts(test_df["text_combined"].tolist())

y_train = train_df["ideology_score"].values
y_val   = val_df["ideology_score"].values
y_test  = test_df["ideology_score"].values

#pca
pca = PCA(n_components=100)   # reduce from 768 → 100
pca.fit(X_train_text)

X_train_pca = pca.transform(X_train_text)
X_val_pca   = pca.transform(X_val_text)
X_test_pca  = pca.transform(X_test_text)

print("PCA explained variance:", sum(pca.explained_variance_ratio_))

#xgboost

from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=300,
    learning_rate=0.05,
    max_depth=6,
    subsample=0.9,
    colsample_bytree=0.9,
    objective="reg:squarederror",
    random_state=42
)

model.fit(X_train_pca, y_train)

#evaluate regression
y_pred = model.predict(X_test_pca)
y_pred = np.clip(y_pred, -1, 1)

mse = mean_squared_error(y_test, y_pred)
r2  = r2_score(y_test, y_pred)

print("\n=== Regression Results ===")
print(f"Mean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

#binning

def to_bins(val):
    if -1.00 <= val < -0.75:
        return 0
    elif -0.75 <= val < -0.25:
        return 1
    elif -0.25 <= val <= 0.25:
        return 2
    elif 0.25 < val <= 0.75:
        return 3
    elif 0.75 < val <= 1.00:
        return 4
    return 2  # fallback centrist

y_test_bins = np.array([to_bins(v) for v in y_test])
y_pred_bins = np.array([to_bins(v) for v in y_pred])

#classification metrics

print("\n=== Classification Results (Binned) ===")
print("Accuracy:", accuracy_score(y_test_bins, y_pred_bins))
print(classification_report(
    y_test_bins, y_pred_bins,
    target_names=[
        "Highly Right",
        "Right",
        "Centrist",
        "Left",
        "Highly Left"
    ]
))

#confusion matrix

cm = confusion_matrix(y_test_bins, y_pred_bins)

plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, cmap="Blues", fmt="d",
            xticklabels=["HR","R","C","L","HL"],
            yticklabels=["HR","R","C","L","HL"])
plt.title("Confusion Matrix – PCA + XGBoost")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
