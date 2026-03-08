import pandas as pd
from sentence_transformers import SentenceTransformer
import torch

print("Loading articles...")

articles = pd.read_csv("data/articles_filtered.csv")

# Create a rich text description combining all relevant metadata
articles["rich_text"] = (
    articles["prod_name"] + " " +
    articles["product_type_name"] + " " +
    articles["colour_group_name"] + ". " +
    articles["detail_desc"].fillna("")
)
texts = articles["rich_text"].tolist()

print("Loading BERT model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

print("Encoding descriptions...")
embeddings = model.encode(texts, show_progress_bar=True)

print("Saving embeddings...")

torch.save(
{
    "item_idx": list(range(len(embeddings))),  # create sequential index
    "embeddings": embeddings
},
"data/text_embeddings.pt"
)

print("Done.")
