# ============================================
# CS441 - Ideology Prediction (High-Accuracy Run)
# Model: TF-IDF text + engagement features + subreddit one-hot
#        → XGBoost (multi:softprob) with class weights + early stopping
#
# Major changes vs earlier versions:
# 1) Uses subreddit as an explicit categorical feature (one-hot).
# 2) Uses XGBoost "multi:softprob" (probabilities) not "softmax".
# 3) Adds per-class sample weights to fight class imbalance.
# 4) Uses xgboost.train with early_stopping_rounds on a validation set.
# 5) Larger TF-IDF vocab (30k) and tuned tree params.
# ============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from scipy.sparse import hstack
import xgboost as xgb

RANDOM_STATE = 42

# -------------------------------------------------
# 1. Load data and basic cleaning
# -------------------------------------------------
df = pd.read_csv("combined_labeled_data.csv")

# Keep only rows with a numeric ideology_score and non-empty comment
df["ideology_score"] = pd.to_numeric(df["ideology_score"], errors="coerce")
df = df.dropna(subset=["ideology_score", "comment"])

# Fill missing text / subreddit / engagement fields
text_cols = ["comment", "post_title", "post_body"]
for c in text_cols:
    if c not in df.columns:
        df[c] = ""
    df[c] = df[c].fillna("")

if "subreddit" not in df.columns:
    df["subreddit"] = ""
df["subreddit"] = df["subreddit"].fillna("")

engagement_features = ["post_score", "post_upvotes", "post_downvotes", "comment_score"]
for c in engagement_features:
    if c not in df.columns:
        df[c] = 0
    df[c] = df[c].fillna(0)

# -------------------------------------------------
# 2. Label binning (same mapping as earlier work)
# -------------------------------------------------
def ideology_label(score: float) -> int:
    """Map continuous ideology_score into 5 discrete bins.
       0: highly right-leaning
       1: right-leaning
       2: centrist
       3: left-leaning
       4: highly left-leaning
    """
    if -1.00 <= score < -0.75:
        return 0
    elif -0.75 <= score < -0.25:
        return 1
    elif -0.25 <= score <= 0.25:
        return 2
    elif 0.25 < score <= 0.75:
        return 3
    elif 0.75 < score <= 1.00:
        return 4
    else:
        return 2  # fallback to centrist if something weird happens

df["label"] = df["ideology_score"].apply(ideology_label)

# -------------------------------------------------
# 3. Train / val / test split (70 / 15 / 15), stratified
# -------------------------------------------------
X = df.copy()
y = df["label"].values

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, random_state=RANDOM_STATE, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, random_state=RANDOM_STATE, stratify=y_temp
)

print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

# -------------------------------------------------
# 4. Feature engineering
#    - Combined text for TF-IDF
#    - Engagement ratios
#    - Subreddit one-hot
# -------------------------------------------------

# Combined text field (comment + post title/body)
def make_text(df_):
    return (
        df_["comment"].astype(str) + " " +
        df_["post_title"].astype(str) + " " +
        df_["post_body"].astype(str)
    )

X_train_text = make_text(X_train)
X_val_text   = make_text(X_val)
X_test_text  = make_text(X_test)

# Engagement ratios
for frame in [X_train, X_val, X_test]:
    frame["engagement_ratio"] = (frame["comment_score"] + 1) / (frame["post_score"] + 1)
    frame["upvote_ratio"] = (frame["post_upvotes"] + 1) / (frame["post_downvotes"] + 1)

all_engagement_features = engagement_features + ["engagement_ratio", "upvote_ratio"]

# -------------------------------------------------
# 4a. TF-IDF on text (fitted on TRAIN only)
# -------------------------------------------------
tfidf = TfidfVectorizer(
    max_features=30000,        # larger vocab for more ideological keywords
    stop_words="english",
    ngram_range=(1, 2)         # unigrams + bigrams
)

X_train_tfidf = tfidf.fit_transform(X_train_text)
X_val_tfidf   = tfidf.transform(X_val_text)
X_test_tfidf  = tfidf.transform(X_test_text)

print("TF-IDF shape (train):", X_train_tfidf.shape)

# -------------------------------------------------
# 4b. Standard-scaled engagement features
# -------------------------------------------------
scaler = StandardScaler()
X_train_eng = scaler.fit_transform(X_train[all_engagement_features])
X_val_eng   = scaler.transform(X_val[all_engagement_features])
X_test_eng  = scaler.transform(X_test[all_engagement_features])

# -------------------------------------------------
# 4c. One-hot encoded subreddit
# -------------------------------------------------
ohe = OneHotEncoder(handle_unknown="ignore")
X_train_sub = ohe.fit_transform(X_train[["subreddit"]])
X_val_sub   = ohe.transform(X_val[["subreddit"]])
X_test_sub  = ohe.transform(X_test[["subreddit"]])

print("Subreddit one-hot shape (train):", X_train_sub.shape)

# -------------------------------------------------
# 4d. Combine all feature blocks: [TF-IDF | engagement | subreddit]
# -------------------------------------------------
from scipy.sparse import csr_matrix

X_train_all = hstack(
    [X_train_tfidf, csr_matrix(X_train_eng), X_train_sub],
    format="csr"
)
X_val_all = hstack(
    [X_val_tfidf, csr_matrix(X_val_eng), X_val_sub],
    format="csr"
)
X_test_all = hstack(
    [X_test_tfidf, csr_matrix(X_test_eng), X_test_sub],
    format="csr"
)

print("Final feature matrix (train):", X_train_all.shape)

# -------------------------------------------------
# 5. Class weights to handle imbalance
# -------------------------------------------------
from collections import Counter

class_counts = Counter(y_train)
num_classes = 5
total_samples = len(y_train)

# Inverse-frequency weights (normalized so average weight ≈ 1)
class_weights = {}
for c in range(num_classes):
    freq = class_counts.get(c, 1)
    class_weights[c] = total_samples / (num_classes * freq)

print("\nClass counts:", class_counts)
print("Class weights:", class_weights)

# Build sample weights vector for training set
sample_weights = np.array([class_weights[label] for label in y_train])

# -------------------------------------------------
# 6. XGBoost DMatrix objects
# -------------------------------------------------
dtrain = xgb.DMatrix(X_train_all, label=y_train, weight=sample_weights)
dval   = xgb.DMatrix(X_val_all,   label=y_val)
dtest  = xgb.DMatrix(X_test_all,  label=y_test)

# -------------------------------------------------
# 7. XGBoost parameters (multi:softprob + tuned depth/regularization)
# -------------------------------------------------
params = {
    "objective": "multi:softprob",   # probabilities for each of 5 ideology bins
    "num_class": num_classes,
    "eval_metric": "mlogloss",
    "eta": 0.05,                     # learning rate
    "max_depth": 12,                 # deeper trees capture more sparse patterns
    "min_child_weight": 1,
    "subsample": 0.9,
    "colsample_bytree": 0.9,
    "gamma": 1.0,
    "lambda": 1.0,
    "alpha": 0.1,
    "tree_method": "hist",           # faster on CPUs
    "seed": RANDOM_STATE
}

# -------------------------------------------------
# 8. Training with EARLY STOPPING on validation set
# -------------------------------------------------
num_boost_round = 1000
early_stopping_rounds = 50

evals = [(dtrain, "train"), (dval, "val")]

print("\n=== Training XGBoost with early stopping ===")
bst = xgb.train(
    params=params,
    dtrain=dtrain,
    num_boost_round=num_boost_round,
    evals=evals,
    early_stopping_rounds=early_stopping_rounds,
    verbose_eval=25  # print every 25 rounds
)

print(f"\nBest iteration: {bst.best_iteration} with val-mlogloss={bst.best_score:.4f}")

# -------------------------------------------------
# 9. Evaluation on TEST set
# -------------------------------------------------
# bst.predict returns probabilities for all classes (softprob)
probs_test = bst.predict(dtest)
y_pred = np.asarray(probs_test).argmax(axis=1)

test_acc = accuracy_score(y_test, y_pred)
print("\n=== TEST RESULTS ===")
print(f"Test Accuracy: {test_acc:.4f}\n")

target_names = [
    "0: Highly Right",
    "1: Right",
    "2: Centrist",
    "3: Left",
    "4: Highly Left"
]
print(classification_report(y_test, y_pred, target_names=target_names))

# Confusion matrix plot
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[t.split(":")[0] for t in target_names],
    yticklabels=[t.split(":")[0] for t in target_names],
)
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — TF-IDF + Engagement + Subreddit + XGBoost")
plt.tight_layout()
plt.show()
