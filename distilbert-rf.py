# distilbert_rf_hybrid.py
import pandas as pd
import numpy as np
import torch
from transformers import DistilBertTokenizer, DistilBertModel
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm

# =====================
# 1. Load and prep data
# =====================
df = pd.read_csv("combined_labeled_data.csv")
df["ideology_score"] = pd.to_numeric(df["ideology_score"], errors="coerce")
df = df.dropna(subset=["ideology_score", "comment"])

engagement = ["post_score", "post_upvotes", "post_downvotes", "comment_score"]
df[engagement] = df[engagement].fillna(0)

df["text"] = (
    df["comment"].fillna("") + " " +
    df["post_title"].fillna("") + " " +
    df["post_body"].fillna("") + " " +
    df["subreddit"].fillna("")
)

train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

# ============================
# 2. DistilBERT embeddings
# ============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased").to(device)
model.eval()

def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size].tolist()
        tokens = tokenizer(batch, padding=True, truncation=True, max_length=256,
                           return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**tokens)
            embeddings = outputs.last_hidden_state[:, 0, :].cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

# Encode text
X_train_text = get_embeddings(train_df["text"])
X_val_text = get_embeddings(val_df["text"])
X_test_text = get_embeddings(test_df["text"])

# ============================
# 3. Combine with engagement
# ============================
scaler = StandardScaler()
train_eng = scaler.fit_transform(train_df[engagement])
val_eng = scaler.transform(val_df[engagement])
test_eng = scaler.transform(test_df[engagement])

X_train = np.hstack([X_train_text, train_eng])
X_val = np.hstack([X_val_text, val_eng])
X_test = np.hstack([X_test_text, test_eng])

y_train = train_df["ideology_score"].values
y_val = val_df["ideology_score"].values
y_test = test_df["ideology_score"].values

# ============================
# 4. Train Random Forest
# ============================
rf = RandomForestRegressor(
    n_estimators=300,
    max_depth=20,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train, y_train)

# ============================
# 5. Evaluate
# ============================
y_pred = np.clip(rf.predict(X_test), -1, 1)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ============================
# 6. Classification-style bins
# ============================
def to_bins(y):
    bins = [-1.01, -0.75, -0.25, 0.25, 0.75, 1.01]
    return np.digitize(y, bins) - 1

y_test_bins = to_bins(y_test)
y_pred_bins = to_bins(y_pred)

print("\nClassification-style Evaluation (binned ideology):")
print(f"Accuracy: {accuracy_score(y_test_bins, y_pred_bins):.3f}")
print(classification_report(
    y_test_bins, y_pred_bins,
    target_names=[
        "Highly Right (-1 to -0.75)",
        "Right (-0.75 to -0.25)",
        "Centrist (-0.25 to 0.25)",
        "Left (0.25 to 0.75)",
        "Highly Left (0.75 to 1)"
    ]
))

cm = confusion_matrix(y_test_bins, y_pred_bins)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["H.Right", "Right", "Center", "Left", "H.Left"],
            yticklabels=["H.Right", "Right", "Center", "Left", "H.Left"])
plt.title("Confusion Matrix — DistilBERT + Engagement + Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
