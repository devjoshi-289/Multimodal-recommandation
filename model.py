import torch
import torch.nn as nn
from sentence_transformers import SentenceTransformer
import pandas as pd
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

device = torch.device("cpu")

# 1. Define the model architecture exactly as in train_cf.py
class HybridNCF(nn.Module):
    def __init__(self, num_users, num_items, num_types, num_colors, num_sections):
        super().__init__()
        self.user_embedding = nn.Embedding(num_users, 128)
        self.item_embedding = nn.Embedding(num_items, 128)
        self.type_embedding = nn.Embedding(num_types, 16)
        self.color_embedding = nn.Embedding(num_colors, 16)
        self.section_embedding = nn.Embedding(num_sections, 16)
        self.mlp = nn.Sequential(
            nn.Linear(2736, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, 1)
        )

    def forward(self, user, item, type_idx, color_idx, section_idx, text_vec, image_vec):
        u = self.user_embedding(user)
        i = self.item_embedding(item)
        t = self.type_embedding(type_idx)
        c = self.color_embedding(color_idx)
        s = self.section_embedding(section_idx)
        x = torch.cat([u, i, t, c, s, text_vec, image_vec], dim=1)
        return self.mlp(x).squeeze()

def load_recsys_model():
    print("Loading HybridNCF model structure and weights...")
    
    # We need to know the dimensions to init. We can read them from the encoded data.
    interactions = pd.read_csv("data/interactions_encoded.csv")
    item_features = pd.read_csv("data/item_features_encoded.csv")
    
    num_users = interactions["user_idx"].nunique()
    num_items = interactions["item_idx"].nunique()
    num_types = item_features["type_idx"].nunique()
    num_colors = item_features["color_idx"].nunique()
    num_sections = item_features["section_idx"].nunique()
    
    model = HybridNCF(num_users, num_items, num_types, num_colors, num_sections)
    
    # Load weights
    try:
        model.load_state_dict(torch.load("best_model.pth", map_location=device))
        model.eval()
        print("Loaded best_model.pth successfully.")
    except Exception as e:
        print(f"Error loading model weights: {e}")
        
    return model

def load_text_encoder():
    print("Loading SentenceTransformer model...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2")
    return encoder

# Load databases
print("Loading item features and embeddings database for rapid search...")
text_data_db = torch.load("data/text_embeddings.pt", map_location=device, weights_only=False)
try:
    image_data_db = torch.load("data/image_embeddings.pt", map_location=device, weights_only=False)
except FileNotFoundError:
    image_data_db = None
    print("Warning: image_embeddings.pt not found.")
item_features_db = pd.read_csv("data/item_features_encoded.csv")

# We map item_idx back to real article_id to fetch images and names
articles_db = pd.read_csv("data/articles_filtered.csv")
    
def load_image_encoder():
    print("Loading ResNet50 model...")
    resnet = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    # Remove the final classification layer to get 2048-d features
    encoder = nn.Sequential(*list(resnet.children())[:-1])
    
    # Load finetuned weights if they exist
    import os
    if os.path.exists("data/resnet50_fashion_finetuned.pth"):
        print("Using fine-tuned Fashion ResNet50 weights!")
        encoder.load_state_dict(torch.load("data/resnet50_fashion_finetuned.pth", map_location=device))
        
    encoder.eval()
    encoder.to(device)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    return encoder, transform

# Global instances so they only load once at server startup
ncf_model = load_recsys_model()
text_encoder = load_text_encoder()
image_encoder, image_transform = load_image_encoder()

def encode_text(text_query):
    return text_encoder.encode(text_query, convert_to_tensor=True)

def encode_image(image_path):
    """
    Encode an image using the loaded ResNet50 model.
    """
    print(f"Processing image query: {image_path}")
    try:
        img = Image.open(image_path).convert('RGB')
        img_tensor = image_transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            emb = image_encoder(img_tensor)
        return emb.squeeze() # Returns a 1D tensor of size 2048
    except Exception as e:
        print(f"Error encoding image: {e}")
        return None
