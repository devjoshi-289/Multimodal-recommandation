import pandas as pd

# Load only required columns
transactions = pd.read_csv(
    "data/transactions_train.csv",
    usecols=["customer_id", "article_id"],
    nrows=500000  # load only first 500k rows
)

articles = pd.read_csv("data/articles.csv", usecols=["article_id", "prod_name", "detail_desc"])

# Select top 10k most frequent articles
top_articles = (
    transactions["article_id"]
    .value_counts()
    .head(10000)
    .index
)

transactions = transactions[transactions["article_id"].isin(top_articles)]
articles = articles[articles["article_id"].isin(top_articles)]

# Save reduced dataset
transactions.to_csv("data/transactions_small.csv", index=False)
articles.to_csv("data/articles_small.csv", index=False)

print("Reduced dataset created.")
print("Transactions:", len(transactions))
print("Articles:", len(articles))