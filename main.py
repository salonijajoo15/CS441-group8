import pandas as pd

# Load the Twitter dataset
twitter_data = pd.read_csv("Political_tweets.csv", low_memory=False)

# Basic info
rows, cols = twitter_data.shape
columns = twitter_data.columns.tolist()

# Volumetric analysis
missing_values = twitter_data.isnull().sum()
missing_percentage = (missing_values / len(twitter_data)) * 100

# Print results
print("Number of rows:", rows)
print("Number of columns:", cols)
print("\nColumns:", columns)
print("\nMissing values (%):\n", missing_percentage)
print("\nSample rows:\n", twitter_data.head(3))


# Workaround: Skip wordcloud rendering due to font issues in environment.
# Instead, provide bar plots and counts of hashtags (more concrete for report).

import pandas as pd
import matplotlib.pyplot as plt

# Top 20 hashtags frequency
labels, values = zip(*top_hashtags)

plt.figure(figsize=(12,6))
plt.bar(labels, values, color="steelblue")
plt.xticks(rotation=45, ha='right')
plt.title("Top 20 Hashtags in Political Tweets", fontsize=14)
plt.ylabel("Frequency")
plt.show()

# Show top 10 hashtags list with counts for report insertion
top_hashtags[:10]
