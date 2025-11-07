# distilbert_logistic.py

import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from transformers import DistilBertTokenizer, DistilBertModel
from tqdm import tqdm
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# 1️⃣ Load Data
df = pd.read_csv("combined_labeled_data.csv")

# Choose text + labels
X_text = df["post_title"].fillna("") + " " + df["comment"].fillna("")
y = df["ideology"].astype("category").cat.codes  # convert string labels to integers

# 2️⃣ Split Data
# After loading df and defining X_text, y
X_train, X_temp, y_train, y_temp = train_test_split(
    X_text, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

# 3️⃣ Load DistilBERT Model and Tokenizer
tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
model = DistilBertModel.from_pretrained("distilbert-base-uncased")
model.eval()  # disable dropout etc.

# 4️⃣ Function to get embeddings (mean pooled)
def get_embeddings(texts, batch_size=16):
    all_embeddings = []
    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding batches"):
        batch = texts[i:i+batch_size]
        encoded = tokenizer(
            list(batch),
            padding=True,
            truncation=True,
            max_length=128,
            return_tensors="pt"
        )

        with torch.no_grad():
            outputs = model(**encoded)
            # Mean pool over token embeddings
            embeddings = outputs.last_hidden_state.mean(dim=1)
            all_embeddings.append(embeddings.cpu().numpy())

    return np.vstack(all_embeddings)

# 5️⃣ Generate embeddings for train/test
X_train_emb = get_embeddings(X_train)
X_test_emb  = get_embeddings(X_test)

# 6️⃣ Train Multinomial Logistic Regression on DistilBERT embeddings
clf = LogisticRegression(max_iter=200, multi_class="multinomial", solver="lbfgs")
clf.fit(X_train_emb, y_train)

# 7️⃣ Evaluate
y_pred = clf.predict(X_test_emb)
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=df["ideology"].astype("category").cat.categories))

cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Confusion Matrix — DistilBERT Embeddings + Logistic Regression")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()
