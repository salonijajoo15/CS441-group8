import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("Political_tweets.csv", low_memory=False)
df["date"] = pd.to_datetime(df["date"], errors="coerce")

def parse_hashtags(x):
    try: return ast.literal_eval(x) if isinstance(x, str) else []
    except Exception: return []

df["hashtags_parsed"] = df["hashtags"].apply(parse_hashtags)
df["user_followers"] = pd.to_numeric(df["user_followers"], errors="coerce")
df["text"] = df["text"].astype(str)
df["text_len_tokens"] = df["text"].str.split().apply(len)

#top 20 hashtags
all_tags = [t.lower() for tags in df["hashtags_parsed"] for t in tags if isinstance(t, str)]
tag_counts = Counter(all_tags)
top20_tags = tag_counts.most_common(20)
if top20_tags:
    labels, values = zip(*top20_tags)
    plt.figure(figsize=(12,6))
    plt.bar(labels, values)
    plt.xticks(rotation=45, ha="right")
    plt.title("Top 20 Hashtags (Twitter Political Dataset)")
    plt.ylabel("Frequency")
    plt.tight_layout()
    plt.show()

#tweet volume by month
if df["date"].notna().sum() > 0:
    ts = df.set_index("date").resample("MS").size()
    plt.figure(figsize=(10,5))
    plt.plot(ts.index, ts.values)
    plt.title("Tweet Volume by Month")
    plt.xlabel("Month"); plt.ylabel("Tweet Count")
    plt.tight_layout(); plt.show()

#followers distribution log
followers = df["user_followers"].dropna()
if len(followers) > 0:
    plt.figure(figsize=(10,5))
    plt.hist(followers, bins=50, log=True)
    plt.title("User Followers Distribution (log-scaled bin counts)")
    plt.xlabel("Followers"); plt.ylabel("Tweet Count (log scale)")
    plt.tight_layout(); plt.show()

#retweet vs original
rt_counts = df["is_retweet"].value_counts(dropna=False)
plt.figure(figsize=(6,4))
plt.bar(rt_counts.index.astype(str), rt_counts.values)
plt.title("Retweet vs Original Tweet Counts")
plt.xlabel("is_retweet"); plt.ylabel("Count")
plt.tight_layout(); plt.show()

#tweet length text
plt.figure(figsize=(10,5))
plt.hist(df["text_len_tokens"], bins=50)
plt.title("Tweet Text Length (tokens)")
plt.xlabel("Tokens"); plt.ylabel("Tweet Count")
plt.tight_layout(); plt.show()

#tf-idf 
vectorizer = TfidfVectorizer(
    lowercase=True, stop_words="english",
    min_df=50, max_df=0.9, ngram_range=(1,2)
)
X = vectorizer.fit_transform(df["text"])
mean_tfidf = np.asarray(X.mean(axis=0)).ravel()
vocab = np.array(vectorizer.get_feature_names_out())
top_idx = mean_tfidf.argsort()[::-1][:20]

plt.figure(figsize=(12,6))
plt.bar(vocab[top_idx], mean_tfidf[top_idx])
plt.xticks(rotation=45, ha="right")
plt.title("Top 20 Terms by Mean TF-IDF (Unigrams+Bigrams)")
plt.ylabel("Mean TF-IDF")
plt.tight_layout(); plt.show()

#weak ideology labels
LEFT_TAGS  = {"bluewave","resist","biden","democrats","womensmarch","blacklivesmatter"}
RIGHT_TAGS = {"maga","kag","trump","tcot","buildthewall","2a"}

def weak_label(tags):
    tags_l = set([t.lower() for t in tags if isinstance(t, str)])
    if tags_l & RIGHT_TAGS and not (tags_l & LEFT_TAGS): return "right"
    if tags_l & LEFT_TAGS and not (tags_l & RIGHT_TAGS): return "left"
    return "unknown"

df["weak_label"] = df["hashtags_parsed"].apply(weak_label)
wl = df["weak_label"].value_counts()

plt.figure(figsize=(6,4))
plt.bar(wl.index.astype(str), wl.values)
plt.title("Weak Ideology Labels from Hashtags (Seed Lexicon)")
plt.xlabel("Label"); plt.ylabel("Tweet Count")
plt.tight_layout(); plt.show()
