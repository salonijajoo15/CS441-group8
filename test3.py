import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt

# =============================
# Load + preprocess data
# =============================
df = pd.read_csv("combined_labeled_data.csv")
df = df.dropna(subset=["ideology_score", "comment"])

df["ideology_score"] = pd.to_numeric(df["ideology_score"], errors="coerce")
df = df.dropna(subset=["ideology_score"])

# Binning function
def to_bins(y):
    bins = [-1.01, -0.75, -0.25, 0.25, 0.75, 1.01]
    return np.digitize(y, bins) - 1

df["label"] = to_bins(df["ideology_score"])

engagement_features = ["post_score", "post_upvotes", "post_downvotes", "comment_score"]
df[engagement_features] = df[engagement_features].fillna(0)

df["engagement_ratio"] = (df["comment_score"] + 1) / (df["post_score"] + 1)
df["upvote_ratio"] = (df["post_upvotes"] + 1) / (df["post_downvotes"] + 1)

all_engagement = engagement_features + ["engagement_ratio", "upvote_ratio"]

df["text_combined"] = (
    df["comment"].fillna("") + " " +
    df["post_title"].fillna("") + " " +
    df["post_body"].fillna("") + " " +
    df["subreddit"].fillna("")
)

# Train/Val/Test = 70/15/15
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42)
val_df, test_df = train_df, temp_df = train_test_split(temp_df, test_size=0.50, random_state=42)

print(f"Train: {len(train_df)} Val: {len(val_df)} Test: {len(test_df)}")

# =============================
# TF-IDF + Engagement
# =============================
tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1,2), stop_words="english")

preprocessor = ColumnTransformer(
    [
        ("text", tfidf, "text_combined"),
        ("eng", StandardScaler(), all_engagement)
    ]
)

X_train = preprocessor.fit_transform(train_df)
X_val   = preprocessor.transform(val_df)
X_test  = preprocessor.transform(test_df)

y_train = train_df["label"].values
y_val   = val_df["label"].values
y_test  = test_df["label"].values

# =============================
# Compute class weights
# =============================
class_counts = np.bincount(y_train)
total = len(y_train)
class_weights = {i: total / (len(class_counts) * class_counts[i]) for i in range(len(class_counts))}
print("\nClass Weights:", class_weights)

# =============================
# Prepare DMatrix objects
# =============================
dtrain = xgb.DMatrix(X_train, label=y_train, weight=[class_weights[i] for i in y_train])
dval   = xgb.DMatrix(X_val, label=y_val)
dtest  = xgb.DMatrix(X_test, label=y_test)

# =============================
# XGBoost Parameters (manual early stopping)
# =============================
params = {
    "objective": "multi:softmax",
    "num_class": 5,
    "max_depth": 8,
    "eta": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "eval_metric": "mlogloss",
}

num_rounds = 2000
early_stop_rounds = 50

# =============================
# Train with manual early stopping
# =============================
watchlist = [(dtrain, "train"), (dval, "validation")]

print("\n=== TRAINING WITH MANUAL EARLY STOPPING ===")
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_rounds,
    evals=watchlist,
    early_stopping_rounds=early_stop_rounds
)

# =============================
# Inference
# =============================
y_pred = bst.predict(dtest)

acc = accuracy_score(y_test, y_pred)
print(f"\nTEST ACCURACY: {acc:.4f}\n")
print(classification_report(y_test, y_pred))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7,6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — Manual Early Stopping")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
