import os
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm


def generate_embeddings():
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Load ResNet50 Architecture
    resnet = models.resnet50()
    # Replace the final layer to match the saved fine-tuned model (85 classes)
    num_ftrs = resnet.fc.in_features
    # We must first load the FULL model dict, so we recreate the classification layer.
    resnet.fc = nn.Linear(num_ftrs, 85)
    
    # We stripped the classification layer when saving! 
    # Wait, in the train script we saved `feature_extractor.state_dict()`.
    # `feature_extractor` is a `nn.Sequential` object.
    
    feature_extractor = nn.Sequential(*list(models.resnet50().children())[:-1])
    print("Loading fine-tuned weights...")
    feature_extractor.load_state_dict(torch.load("data/resnet50_fashion_finetuned.pth", map_location='cpu'))
    
    feature_extractor = feature_extractor.to(device)
    feature_extractor.eval()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    df = pd.read_csv("data/articles_filtered.csv")
    img_dir = "data/filtered_images"

    df = pd.read_csv("data/articles_filtered.csv")
    img_dir = "data/filtered_images"

    # Read valid images
    valid_data = []
    print("Verifying image files...")
    for _, row in df.iterrows():
        article_id = row['article_id']
        img_path = os.path.join(img_dir, f"0{article_id}.jpg")
        if os.path.exists(img_path):
            valid_data.append((article_id, img_path))
    print(f"Found {len(valid_data)} valid images out of {len(df)}")
    
    batch_size = 64
    valid_indices = []
    embeddings_list = []

    print("Extracting features (manual batching)...")
    
    with torch.no_grad():
        for i in tqdm(range(0, len(valid_data), batch_size)):
            batch = valid_data[i:i+batch_size]
            
            # Load images
            batch_tensors = []
            batch_ids = []
            
            for article_id, img_path in batch:
                image = Image.open(img_path).convert('RGB')
                tensor = transform(image)
                batch_tensors.append(tensor)
                batch_ids.append(article_id)
                
            img_tensor_stack = torch.stack(batch_tensors).to(device)
            
            # Forward Pass
            embs = feature_extractor(img_tensor_stack).view(img_tensor_stack.size(0), -1).cpu().numpy()
            
            valid_indices.extend(batch_ids)
            embeddings_list.extend(embs)

    print(f"Successfully encoded {len(valid_indices)} images.")
    
    # Save embeddings format identical to before
    print("Saving to data/image_embeddings.pt...")
    torch.save(
        {
            "item_idx": valid_indices,
            "embeddings": embeddings_list
        },
        "data/image_embeddings.pt"
    )
    print("Done!")

if __name__ == '__main__':
    generate_embeddings()
