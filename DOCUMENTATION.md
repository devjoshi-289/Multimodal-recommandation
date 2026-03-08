# Fashion Recommender System - Documentation

This document explains the architecture of the Lumina Fashion Recommender, describes the purpose of each file, and provides instructions on data storage and model usage.

---

## 1. File Descriptions

### Core Backend & API
- **`main.py`**: The entry point for the FastAPI server. It defines the API endpoints for searching and serves the static frontend files.
- **`search.py`**: Contains the core multimodal search logic. It calculates cosine similarity between the query (text or image) and the precomputed embeddings in the database.
- **`model.py`**: Handles loading all AI models (SentenceTransformer for text, ResNet50 for images, and HybridNCF for recommendations). It also initializes the global instances of these models to ensure fast performance.

### Data Processing & Preparation
- **`preprocess.py`**: Takes raw H&M data (`transactions_train.csv` and `articles.csv`) and creates smaller, manageable versions (`transactions_small.csv` and `articles_small.csv`).
- **`filter_articles.py`**: Filters the product catalog to focus on specific categories or ensure data consistency.
- **`filter_images.py`**: Scans the image directory and ensures that only images with corresponding metadata in the filtered CSV are kept.
- **`encode.py`**: Maps raw Article IDs and Customer IDs into integer indices (`item_idx`, `user_idx`) used by the collaborative filtering model.

### Training & Embedding Generation
- **`train_image_encoder.py`**: Fine-tunes the ResNet50 model on your specific fashion dataset (predicting product types) to learn fashion-specific visual features.
- **`update_image_embeddings.py`**: Uses the fine-tuned ResNet50 to generate 2048-dimensional vectors for every product image and saves them to `data/image_embeddings.pt`.
- **`create_text_embeddings.py`**: Combines product names, types, and colors into a single string for every item and generates BERT embeddings saved in `data/text_embeddings.pt`.
- **`train_cf.py`**: Trains the Hybrid Neural Collaborative Filtering (NCF) model using user-item interactions and item features.

---

## 2. Data Storage Structure

The system expects files to be organized in the `data/` directory as follows:

- **`data/filtered_images/`**: Put all your `.jpg` product images here. Filenames should follow the H&M convention (e.g., `0108775015.jpg`).
- **`data/articles_filtered.csv`**: The primary product database containing names, colors, and descriptions.
- **`data/*.pt` & `data/*.pth`**: These are binary files containing the precomputed embeddings and model weights.
- **`data/transactions_small.csv`**: A subset of purchase history used for training the recommender.

---

## 3. How to Use the Model Pipeline

If you want to rebuild the system from scratch or update it with new data, follow these steps in order:

1. **Prepare Data**: Run `preprocess.py` and `filter_images.py`.
2. **Fine-tune Vision**: Run `train_image_encoder.py`. This creates the fashion-aware brain for the system.
3. **Generate Embeddings**:
   - Run `update_image_embeddings.py` to process the images.
   - Run `create_text_embeddings.py` to process the text descriptions.
4. **Train Recommender**: Run `train_cf.py` to learn user preferences.
5. **Start Server**: Run `python main.py`.

---

## 4. Web Interface
The frontend is located in the **`ux/`** folder. It is a premium, single-page application that communicates with the API at `localhost:8000`. To use it, simply open `ux/index.html` in your browser while the server is running.
