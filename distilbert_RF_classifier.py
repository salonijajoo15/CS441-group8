# distilbert_random_forest.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
from transformers import DistilBertTokenizer, DistilBertModel
import torch

# ======================
# 1️⃣ Load and Prepare Data
# ======================
df = pd.read_csv("combined_labeled_data.csv")

# Combine post title + comment text
X_text = df["post_title"].fillna('') + " " + df["comment"].fillna('')
y = df["ideology"].astype("category").cat.codes

# Include engagement features (optional)
engagement_features = df[["post_score", "post_upvotes", "comment_score"]].fillna(0)

# Split into train / val / test
X_train, X_temp, y_train, y_temp, eng_train, eng_temp = train_test_split(
    X_text, y, engagement_features, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test, eng_val, eng_test = train_test_split(
    X_temp, y_temp, eng_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# ======================
# 2️⃣ Generate DistilBERT Embeddings
# ======================
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()

def get_embeddings(texts, batch_size=32):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts.iloc[i:i+batch_size].tolist()
        inputs = tokenizer(batch, padding=True, truncation=True, return_tensors="pt", max_length=128)
        with torch.no_grad():
            outputs = model(**inputs)
        embeddings = outputs.last_hidden_state.mean(dim=1).cpu().numpy()
        all_embeddings.append(embeddings)
    return np.vstack(all_embeddings)

train_emb = get_embeddings(X_train)
val_emb   = get_embeddings(X_val)
test_emb  = get_embeddings(X_test)

# Combine embeddings with engagement metrics
X_train_final = np.hstack([train_emb, eng_train.values])
X_val_final   = np.hstack([val_emb, eng_val.values])
X_test_final  = np.hstack([test_emb, eng_test.values])

# ======================
# 3️⃣ Train Random Forest Classifier
# ======================
rf = RandomForestClassifier(
    n_estimators=200,
    max_depth=25,
    random_state=42,
    n_jobs=-1
)
rf.fit(X_train_final, y_train)

# ======================
# 4️⃣ Evaluate Model
# ======================
y_pred = rf.predict(X_test_final)

print("\nClassification Report:")
print(classification_report(y_test, y_pred))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — DistilBERT Embeddings + Random Forest")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
