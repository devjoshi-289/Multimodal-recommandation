import logging
import torch
import torch.nn.functional as F
import pandas as pd
import numpy as np
from model import text_data_db, image_data_db, item_features_db, articles_db, ncf_model, encode_text, encode_image

logger = logging.getLogger("search")

# Precompute article dict for O(1) lookups
article_dict = articles_db.set_index("article_id")["prod_name"].to_dict()

# Move database to CPU tensor for fast similarity computation
all_item_idx = torch.tensor(text_data_db["item_idx"])
all_text_embeddings = torch.from_numpy(np.array(text_data_db["embeddings"])).float()

# Normalize once to speed up cosine similarity
all_text_embeddings = F.normalize(all_text_embeddings, p=2, dim=1)

# Precompute image embeddings
if image_data_db is not None:
    all_image_idx = torch.tensor(image_data_db["item_idx"])
    all_image_embeddings = torch.from_numpy(np.array(image_data_db["embeddings"])).float()
    all_image_embeddings = F.normalize(all_image_embeddings, p=2, dim=1)
else:
    all_image_idx = None
    all_image_embeddings = None

def multimodal_search(image_path=None, text_query=None):
    """
    Computes similarity between the inputs and dataset items.
    Returns the top 5 most similar products.
    """
    logger.info(f"Starting search with text: '{text_query}', image: {image_path}")
    
    top_indices = []
    
    # 1. Text Search Using SentenceTransformers (all-MiniLM-L6-v2)
    if text_query:
        query_emb = encode_text(text_query).float().cpu()
        # Cosine similarity
        query_emb = F.normalize(query_emb, p=2, dim=0).unsqueeze(0)
        similarities = torch.mm(query_emb, all_text_embeddings.transpose(0, 1)).squeeze(0)
        
        # Get top 5 indices
        top_k = 5
        scores, top_idx_tensors = torch.topk(similarities, k=top_k)
        
        # We can also run the HybridNCF for these items given a mock user
        # to boost personalization, but for pure search, similarity is best.
        
        results = []
        for i in range(top_k):
            idx = int(top_idx_tensors[i])
            score = float(scores[i])
            
            # map index back to real article_id from articles_db
            # since text_embeddings were created in the order of articles_filtered.csv
            real_article_id = int(articles_db.iloc[idx]["article_id"])
            
            # fetch article info using the fast dictionary
            prod_name = article_dict.get(real_article_id)
            
            if prod_name:
                # Image URL served via FastAPI static mounting
                image_url = f"http://localhost:8000/images/0{real_article_id}.jpg"
                
                results.append({
                    "image": image_url,
                    "name": prod_name,
                    "similarity": round(score, 3)
                })
                
        logger.info(f"Text search found {len(results)} matches.")
        return results

    # 2. Image Search
    if image_path:
        if all_image_embeddings is None:
            logger.warning("Image search requested but image_embeddings.pt is not loaded.")
            return [{"image": "https://via.placeholder.com/400x500?text=No+Database", "name": "Error: Setup Required", "similarity": 0.0}]
        
        query_emb = encode_image(image_path)
        if query_emb is None:
            return [{"image": "https://via.placeholder.com/400x500?text=Encoding+Failed", "name": "Error Encoding Image", "similarity": 0.0}]
        
        query_emb = query_emb.float().cpu()
        query_emb = F.normalize(query_emb, p=2, dim=0).unsqueeze(0)
        
        # Cosine similarity
        similarities = torch.mm(query_emb, all_image_embeddings.transpose(0, 1)).squeeze(0)
        
        top_k = 5
        scores, top_idx_tensors = torch.topk(similarities, k=top_k)
        
        results = []
        for i in range(top_k):
            idx = int(top_idx_tensors[i])
            score = float(scores[i])
            
            # For image embeddings, the stored idx is actually the article_id
            article_id = all_image_idx[idx].item()
            
            prod_name = article_dict.get(article_id)
                
            if prod_name:
                image_url = f"http://localhost:8000/images/0{article_id}.jpg"
                
                results.append({
                    "image": image_url,
                    "name": prod_name,
                    "similarity": round(score, 3)
                })
        return results
