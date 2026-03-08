import pandas as pd

# -----------------------------
# Load reduced transactions
# -----------------------------
transactions = pd.read_csv("data/transactions_small.csv")

# Get unique article IDs used in transactions
used_articles = transactions["article_id"].unique()

print("Unique articles used:", len(used_articles))

# -----------------------------
# Load original articles file
# -----------------------------
articles = pd.read_csv("data/articles.csv")

print("Original articles:", len(articles))

# -----------------------------
# Filter only needed articles
# -----------------------------
articles_filtered = articles[articles["article_id"].isin(used_articles)]

print("Filtered articles:", len(articles_filtered))

# -----------------------------
# Save filtered dataset
# -----------------------------
articles_filtered.to_csv("data/articles_filtered.csv", index=False)

print("articles_filtered.csv created successfully!")
