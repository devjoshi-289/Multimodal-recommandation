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

# Precompute categories mapping (we map idx -> index_group_name)
# We need an array aligned with all_item_idx so we can quickly filter
idx_categories = articles_db["index_group_name"].values

# Normalize once to speed up cosine similarity
all_text_embeddings = F.normalize(all_text_embeddings, p=2, dim=1)

# Precompute image embeddings
if image_data_db is not None:
    all_image_idx = torch.tensor(image_data_db["item_idx"])
    
    # Map article_id back to df index, then to category. Handle missing gracefully
    article_idx_map = articles_db.reset_index().set_index("article_id")["index"].to_dict()
    img_categories = []
    
    for aid in all_image_idx.tolist():
        df_idx = article_idx_map.get(aid)
        if df_idx is not None:
            img_categories.append(idx_categories[df_idx])
        else:
            img_categories.append("Unknown")
            
    img_categories = np.array(img_categories)

    all_image_embeddings = torch.from_numpy(np.array(image_data_db["embeddings"])).float()
    all_image_embeddings = F.normalize(all_image_embeddings, p=2, dim=1)
else:
    all_image_idx = None
    all_image_embeddings = None
    img_categories = None

def multimodal_search(image_path=None, text_query=None, category=None, limit=20, offset=0):
    """
    Computes similarity between the inputs and dataset items.
    Returns the most similar products with pagination and category filtering.
    """
    logger.info(f"Search params -> text: '{text_query}', image: {image_path}, category: {category}, limit: {limit}, offset: {offset}")
    
    # 0. Pure Category Browsing (No text, no image)
    if not text_query and not image_path and category:
        category_mask = (articles_db["index_group_name"] == category)
        filtered_df = articles_db[category_mask]
        
        # Determine slice safely
        total_items = len(filtered_df)
        if offset >= total_items:
            return []
            
        end_idx = min(offset + limit, total_items)
        page_df = filtered_df.iloc[offset:end_idx]
        
        results = []
        for _, row in page_df.iterrows():
            aid = row["article_id"]
            prod_name = row["prod_name"]
            results.append({
                "image": f"http://localhost:8000/images/0{aid}.jpg",
                "name": prod_name,
                "similarity": None  # No AI score for pure browsing
            })
            
        logger.info(f"Category '{category}' browse returned {len(results)} items.")
        return results
    
    # 1. Text Search Using SentenceTransformers
    if text_query:
        query_emb = encode_text(text_query).float().cpu()
        query_emb = F.normalize(query_emb, p=2, dim=0).unsqueeze(0)
        similarities = torch.mm(query_emb, all_text_embeddings.transpose(0, 1)).squeeze(0)
        
        # Apply category filter by setting masked similarities to -inf
        if category:
            mask = torch.tensor(idx_categories == category)
            similarities[~mask] = float('-inf')
        
        # Get top k = offset + limit items to allow pagination
        k = offset + limit
        if k > len(similarities):
            k = len(similarities)
            
        scores, top_idx_tensors = torch.topk(similarities, k=k)
        
        results = []
        # Slice the results according to offset
        for i in range(offset, len(scores)):
            idx = int(top_idx_tensors[i])
            score = float(scores[i])
            
            # Skip masked out results
            if score == float('-inf'):
                break
            
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
        
        similarities = torch.mm(query_emb, all_image_embeddings.transpose(0, 1)).squeeze(0)
        
        if category:
            mask = torch.tensor(img_categories == category)
            similarities[~mask] = float('-inf')
        
        k = offset + limit
        if k > len(similarities):
            k = len(similarities)
            
        scores, top_idx_tensors = torch.topk(similarities, k=k)
        
        results = []
        for i in range(offset, len(scores)):
            idx = int(top_idx_tensors[i])
            score = float(scores[i])
            
            if score == float('-inf'):
                break
            
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
