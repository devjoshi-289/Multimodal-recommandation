import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
import pandas as pd
from tqdm import tqdm

class FashionImageDataset(Dataset):
    def __init__(self, csv_file, img_dir, transform=None, sample_size=1000):
        # Load a smaller subset for fast training
        self.df = pd.read_csv(csv_file).sample(n=sample_size, random_state=42)
        self.img_dir = img_dir
        self.transform = transform
        
        # We need to map product_type_name to an integer ID.
        # We will predict product_type_name as the primary task for visual similarity learning.
        self.types = self.df['product_type_name'].unique().tolist()
        self.type_to_idx = {t: i for i, t in enumerate(self.types)}
        self.num_classes = len(self.types)
        print(f"Dataset loaded with {len(self.df)} items and {self.num_classes} unique product types.")

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        article_id = str(row['article_id'])
        # The images are stored as "0" + article_id + ".jpg"
        img_name = f"0{article_id}.jpg"
        img_path = os.path.join(self.img_dir, img_name)
        
        try:
            image = Image.open(img_path).convert('RGB')
        except FileNotFoundError:
            # Fallback to black image if missing
            image = Image.new('RGB', (224, 224), (0, 0, 0))

        if self.transform:
            image = self.transform(image)
            
        label = self.type_to_idx[row['product_type_name']]
            
        return image, torch.tensor(label, dtype=torch.long)


def train_model():
    # Setup device
    device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Standard ResNet transformations with some data augmentation
    train_transforms = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    dataset = FashionImageDataset(
        csv_file="data/articles_filtered.csv",
        img_dir="data/filtered_images",
        transform=train_transforms
    )

    dataloader = DataLoader(dataset, batch_size=64, shuffle=True, num_workers=4)

    # Load ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    
    # Freeze bottom layers (optional, but speeds up training and prevents overfitting on small datasets)
    # Let's freeze the first 8 children out of 10 for super fast fine-tuning
    child_counter = 0
    for child in model.children():
        if child_counter < 8:
            for param in child.parameters():
                param.requires_grad = False
        child_counter += 1

    # Replace the final layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, dataset.num_classes)
    
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-4)

    num_epochs = 3 # 3 epochs should be enough for basic fine-tuning
    print("Starting training...")

    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(dataloader, total=len(dataloader), desc=f"Epoch {epoch+1}/{num_epochs}")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()

            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)

        epoch_loss = running_loss / len(dataset)
        epoch_acc = 100. * correct / total
        print(f"Epoch {epoch+1}/{num_epochs} - Loss: {epoch_loss:.4f} Acc: {epoch_acc:.2f}%")

    # Save ONLY the feature extractor part (remove the classification layer we just added)
    print("Saving fine-tuned backbone...")
    feature_extractor = nn.Sequential(*list(model.children())[:-1])
    torch.save(feature_extractor.state_dict(), "data/resnet50_fashion_finetuned.pth")
    print("Saved to data/resnet50_fashion_finetuned.pth")

if __name__ == '__main__':
    train_model()
