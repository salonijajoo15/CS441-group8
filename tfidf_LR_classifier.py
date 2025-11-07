import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------
# 1. Load dataset
# -------------------------
df = pd.read_csv("combined_labeled_data.csv")

# Combine post title + comment for better context
df["text"] = (
    df["post_title"].fillna("") + " " + df["comment"].fillna("")
).str.lower()

# -------------------------
# 2. Create discrete ideology labels
# -------------------------
df = df.dropna(subset=["ideology_score"])
df["ideology_label"] = pd.qcut(df["ideology_score"], q=5, labels=[0, 1, 2, 3, 4])

# -------------------------
# 3. Define features and target
# -------------------------
X = df[["text", "post_score", "post_upvotes", "comment_score"]]
y = df["ideology_label"]

# -------------------------
# 4. Train / validation / test split
# -------------------------
X_train_full, X_temp, y_train_full, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f"Train size: {len(X_train_full)}, Val size: {len(X_val)}, Test size: {len(X_test)}")

# -------------------------
# 5. Preprocessing: text + numeric
# -------------------------
text_transformer = TfidfVectorizer(stop_words="english", max_features=10000)
numeric_features = ["post_score", "post_upvotes", "comment_score"]
numeric_transformer = StandardScaler()

preprocessor = ColumnTransformer(
    transformers=[
        ("text", text_transformer, "text"),
        ("numeric", numeric_transformer, numeric_features),
    ]
)

# -------------------------
# 6. Build pipeline
# -------------------------
pipeline = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("clf", LogisticRegression(max_iter=1000, multi_class="multinomial")),
    ]
)

# -------------------------
# 7. Grid Search for best hyperparams
# -------------------------
param_grid = {
    "clf__C": [0.1, 1, 5],
    "preprocessor__text__max_features": [5000, 8000, 12000],
}

grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=3,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
    error_score="raise"
)

# -------------------------
# 8. Train
# -------------------------
grid_search.fit(X_train_full, y_train_full)
print(f"Best params: {grid_search.best_params_}")
best_model = grid_search.best_estimator_

# -------------------------
# 9. Evaluate on validation set
# -------------------------
y_val_pred = best_model.predict(X_val)
print("\nValidation Report:\n", classification_report(y_val, y_val_pred))

# -------------------------
# 10. Final test evaluation
# -------------------------
y_pred = best_model.predict(X_test)

print("\nTest Classification Report:\n", classification_report(y_test, y_pred))

# -------------------------
# 11. Confusion matrix
# -------------------------
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — TF-IDF + Engagement + Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.tight_layout()
plt.show()
