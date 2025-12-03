# =====================================
# TF-IDF + Engagement Ratios + XGBoost (Huber Loss)
# Continuous Ideology Prediction (-1 to 1)
# =====================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score, classification_report, accuracy_score, confusion_matrix
from xgboost import XGBRegressor

# ===========================
# 1. LOAD DATA
# ===========================
df = pd.read_csv("combined_labeled_data.csv")

# Ensure ideology_score is numeric
df["ideology_score"] = pd.to_numeric(df["ideology_score"], errors="coerce")
df = df.dropna(subset=["ideology_score", "comment"])  # remove rows missing core fields

# Basic engagement columns
engagement_features = ["post_score", "post_upvotes", "post_downvotes", "comment_score"]
df[engagement_features] = df[engagement_features].fillna(0)

# ===========================
# 2. FEATURE ENGINEERING
# ===========================
# Engagement ratios (captures controversy and agreement)
df["engagement_ratio"] = (df["comment_score"] + 1) / (df["post_score"] + 1)
df["upvote_ratio"] = (df["post_upvotes"] + 1) / (df["post_downvotes"] + 1)

# Combine all engagement features
all_engagement_features = engagement_features + ["engagement_ratio", "upvote_ratio"]

# Combine text fields
df["text_combined"] = (
    df["comment"].fillna("") + " " +
    df["post_title"].fillna("") + " " +
    df["post_body"].fillna("") + " " +
    df["subreddit"].fillna("")
)

# ===========================
# 3. TRAIN/VAL/TEST SPLIT (70/15/15)
# ===========================
train_df, temp_df = train_test_split(df, test_size=0.30, random_state=42, shuffle=True)
val_df, test_df = train_test_split(temp_df, test_size=0.50, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}, Test: {len(test_df)}")

X_train, y_train = train_df[["text_combined"] + all_engagement_features], train_df["ideology_score"]
X_val, y_val = val_df[["text_combined"] + all_engagement_features], val_df["ideology_score"]
X_test, y_test = test_df[["text_combined"] + all_engagement_features], test_df["ideology_score"]

# ===========================
# 4. TF-IDF + XGBOOST PIPELINE
# ===========================
text_transformer = TfidfVectorizer(
    max_features=15000,
    stop_words="english",
    ngram_range=(1, 2)
)

preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_transformer, "text_combined"),
        ("engagement", StandardScaler(), all_engagement_features)
    ]
)

xgb_model = XGBRegressor(
    n_estimators=1000,
    learning_rate=0.03,
    max_depth=8,
    subsample=0.85,
    colsample_bytree=0.85,
    reg_alpha=0.1,
    reg_lambda=1.0,
    objective="reg:pseudohubererror",  # Huber loss (robust to outliers)
    tree_method="hist",
    random_state=42,
    n_jobs=-1
)

model = Pipeline(steps=[
    ("preprocessor", preprocessor),
    ("regressor", xgb_model)
])

# ===========================
# 5. TRAIN & EVALUATE
# ===========================
model.fit(X_train, y_train)
y_pred = np.clip(model.predict(X_test), -1, 1)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f"\nMean Squared Error: {mse:.4f}")
print(f"R² Score: {r2:.4f}")

# ===========================
# 6. CONTINUOUS SCATTER PLOT
# ===========================
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6)
plt.xlabel("True Ideology Score")
plt.ylabel("Predicted Ideology Score")
plt.title("TF-IDF + Engagement Ratios + XGBoost (Huber Loss)")
plt.grid(True)
plt.show()

# ===========================
# 7. BINNED CLASSIFICATION EVAL
# ===========================
def to_bins(y):
    bins = [-1.01, -0.75, -0.25, 0.25, 0.75, 1.01]
    labels = [0, 1, 2, 3, 4]  # 0=Highly Right, 4=Highly Left
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

# Confusion matrix
plt.figure(figsize=(7,6))
cm = confusion_matrix(y_test_bins, y_pred_bins)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["H.Right", "Right", "Center", "Left", "H.Left"],
            yticklabels=["H.Right", "Right", "Center", "Left", "H.Left"])
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix — TF-IDF + XGBoost (Huber Loss + Engagement Ratios)")
plt.show()
