import pandas as pd

print("Loading transactions_small.csv...")

transactions = pd.read_csv("data/transactions_small.csv")

print("Creating ID mappings...")

user_ids = transactions["customer_id"].unique()
item_ids = transactions["article_id"].unique()

user_map = {u: i for i, u in enumerate(user_ids)}
item_map = {a: i for i, a in enumerate(item_ids)}

transactions["user_idx"] = transactions["customer_id"].map(user_map)
transactions["item_idx"] = transactions["article_id"].map(item_map)

print("Saving encoded interactions...")

transactions[["user_idx", "item_idx"]].to_csv("data/interactions_encoded.csv", index=False)

print("Encoding done.")
print("Users:", len(user_map))
print("Items:", len(item_map))
print("Interactions:", len(transactions))

# ------------------------------------------------
# Encode item attributes for hybrid recommender
# ------------------------------------------------

import pandas as pd

# Load datasets
articles = pd.read_csv("data/articles_filtered.csv")
transactions = pd.read_csv("data/transactions_small.csv")
interactions = pd.read_csv("data/interactions_encoded.csv")

# Build article_id → item_idx mapping
article_to_item = transactions.drop_duplicates("article_id").reset_index(drop=True)
article_to_item["item_idx"] = interactions["item_idx"]

mapping = dict(zip(article_to_item["article_id"], article_to_item["item_idx"]))

# Map item_idx into articles
articles["item_idx"] = articles["article_id"].map(mapping)

# Remove rows without mapping
articles = articles.dropna(subset=["item_idx"])

articles["item_idx"] = articles["item_idx"].astype(int)

# Encode attributes
articles["type_idx"] = articles["product_type_name"].astype("category").cat.codes
articles["color_idx"] = articles["colour_group_name"].astype("category").cat.codes
articles["section_idx"] = articles["section_name"].astype("category").cat.codes

# Save encoded features
item_features = articles[
    ["item_idx", "type_idx", "color_idx", "section_idx"]
]

item_features.to_csv("data/item_features_encoded.csv", index=False)

print("Item feature encoding complete.")
