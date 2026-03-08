import torch
import torch.nn as nn

class HybridNCF(nn.Module):

    def __init__(self, num_users, num_items,
                 num_types, num_colors, num_sections):

        super().__init__()

        self.user_embedding = nn.Embedding(num_users,128)
        self.item_embedding = nn.Embedding(num_items,128)

        self.type_embedding = nn.Embedding(num_types,16)
        self.color_embedding = nn.Embedding(num_colors,16)
        self.section_embedding = nn.Embedding(num_sections,16)

        self.mlp = nn.Sequential(
            nn.Linear(2736,256),
            nn.ReLU(),
            nn.Linear(256,128),
            nn.ReLU(),
            nn.Linear(128,1)
        )

    def forward(self,user,item,type_idx,color_idx,section_idx,text_vec,image_vec):

        u = self.user_embedding(user)
        i = self.item_embedding(item)

        t = self.type_embedding(type_idx)
        c = self.color_embedding(color_idx)
        s = self.section_embedding(section_idx)

        x = torch.cat([u,i,t,c,s,text_vec,image_vec], dim=1)

        return self.mlp(x).squeeze()
if __name__ == "__main__":
    import pandas as pd
    from torch.utils.data import Dataset, DataLoader
    from sklearn.model_selection import train_test_split
    import numpy as np

    device = torch.device("cpu")
    print("Using device:", device)

    # -----------------------------
    # Load interaction data
    # -----------------------------
    data = pd.read_csv("data/interactions_encoded.csv")

    num_users = data["user_idx"].nunique()
    num_items = data["item_idx"].nunique()

    print("Users:", num_users)
    print("Items:", num_items)
    print("Interactions:", len(data))

    train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)

    # -----------------------------
    # Load item feature data
    # -----------------------------
    item_features = pd.read_csv("data/item_features_encoded.csv")
    valid_items = set(item_features["item_idx"].values)
    data = data[data["item_idx"].isin(valid_items)]
    print("Filtered interactions:", len(data))

    item_feature_dict = {
        row["item_idx"]: (
            row["type_idx"],
            row["color_idx"],
            row["section_idx"]
        )
        for _, row in item_features.iterrows()
    }

    num_types = item_features["type_idx"].nunique()
    num_colors = item_features["color_idx"].nunique()
    num_sections = item_features["section_idx"].nunique()

    # -----------------------------
    # Load text embeddings
    # -----------------------------
    text_data = torch.load("data/text_embeddings.pt", weights_only=False)

    text_embedding_dict = {
        int(idx): torch.tensor(vec, dtype=torch.float32)
        for idx, vec in zip(text_data["item_idx"], text_data["embeddings"])
    }

    # -----------------------------
    # Load image embeddings
    # -----------------------------
    image_data = torch.load("data/image_embeddings.pt", weights_only=False)

    image_embedding_dict = {
        int(idx): torch.tensor(vec, dtype=torch.float32)
        for idx, vec in zip(image_data["item_idx"], image_data["embeddings"])
    }

    # -----------------------------
    # Dataset with negative sampling
    # -----------------------------
    class InteractionDataset(Dataset):

        def __init__(self, df, num_items):
            self.users = df["user_idx"].values
            self.items = df["item_idx"].values
            self.num_items = num_items

            self.user_item_set = {}
            for u, i in zip(self.users, self.items):
                self.user_item_set.setdefault(u, set()).add(i)

        def __len__(self):
            return len(self.users)

        def __getitem__(self, idx):

            user = self.users[idx]
            pos_item = self.items[idx]
            text_vec = text_embedding_dict.get(pos_item, torch.zeros(384))
            image_vec = image_embedding_dict.get(pos_item, torch.zeros(2048))

            while True:
                neg_item = np.random.randint(0, self.num_items)
                if neg_item not in self.user_item_set[user]:
                    break

            type_idx, color_idx, section_idx = item_feature_dict.get(pos_item, (0,0,0))

            return (
                torch.tensor(user),
                torch.tensor(pos_item),
                torch.tensor(neg_item),
                torch.tensor(type_idx),
                torch.tensor(color_idx),
                torch.tensor(section_idx),
                text_vec,
                image_vec
            )

    train_dataset = InteractionDataset(train_data, num_items)
    train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
# -----------------------------
# Model HybridNCF
# -----------------------------

    # -----------------------------
    # Create model
    # -----------------------------
    model = HybridNCF(
        num_users,
        num_items,
        num_types,
        num_colors,
        num_sections
    ).to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

    # -----------------------------
    # Build test structure
    # -----------------------------
    test_user_item = {}

    for _, row in test_data.iterrows():
        test_user_item.setdefault(row["user_idx"], []).append(row["item_idx"])

    # -----------------------------
    # Training Script
    # -----------------------------

    if __name__ == "__main__":

        print("Starting training...")

        train_dataset = InteractionDataset(train_data, num_items)
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)

        model = HybridNCF(
            num_users,
            num_items,
            num_types,
            num_colors,
            num_sections
        ).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)

        # Build test structure
        test_user_item = {}

        for _, row in test_data.iterrows():
            test_user_item.setdefault(row["user_idx"], []).append(row["item_idx"])

        best_ndcg = 0

        for epoch in range(20):

            model.train()
            total_loss = 0

            for users, pos_items, neg_items, type_idx, color_idx, section_idx, text_vec, image_vec in train_loader:

                users = users.to(device)
                pos_items = pos_items.to(device)
                neg_items = neg_items.to(device)

                type_idx = type_idx.to(device)
                color_idx = color_idx.to(device)
                section_idx = section_idx.to(device)

                text_vec = text_vec.to(device)
                image_vec = image_vec.to(device)

                optimizer.zero_grad()

                pos_scores = model(users, pos_items, type_idx, color_idx, section_idx, text_vec, image_vec)

                text_vec_neg = torch.stack([
                    text_embedding_dict.get(int(i), torch.zeros(384))
                    for i in neg_items.cpu()
                ]).to(device)

                image_vec_neg = torch.stack([
                    image_embedding_dict.get(int(i), torch.zeros(2048))
                    for i in neg_items.cpu()
                ]).to(device)

                neg_scores = model(users, neg_items, type_idx, color_idx, section_idx, text_vec_neg, image_vec_neg)

                loss = -torch.mean(torch.nn.functional.logsigmoid(pos_scores - neg_scores))

                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            print(f"Epoch {epoch+1} Loss: {total_loss:.4f}")

        print("Training finished")

        # -----------------------------
        # Evaluation
        # -----------------------------
        model.eval()

        hits = 0
        ndcg_total = 0
        K = 10
        num_eval_users = 1000

        eval_users = list(test_user_item.keys())[:num_eval_users]

        with torch.no_grad():

            for user in eval_users:

                pos_item = test_user_item[user][0]

                neg_items = set()

                while len(neg_items) < 99:
                    neg = np.random.randint(0, num_items)
                    if neg != pos_item:
                        neg_items.add(neg)

                items_to_rank = list(neg_items)
                items_to_rank.append(pos_item)

                user_tensor = torch.tensor([user]*100).to(device)
                item_tensor = torch.tensor(items_to_rank).to(device)

                type_idx, color_idx, section_idx = item_feature_dict.get(pos_item,(0,0,0))

                type_tensor = torch.tensor([type_idx]*100).to(device)
                color_tensor = torch.tensor([color_idx]*100).to(device)
                section_tensor = torch.tensor([section_idx]*100).to(device)

                text_tensor = torch.stack(
                    [text_embedding_dict.get(int(i), torch.zeros(384)) for i in item_tensor.cpu()]
                ).to(device)

                image_tensor = torch.stack([
                    image_embedding_dict.get(int(i), torch.zeros(2048))
                    for i in items_to_rank
                ]).to(device)

                scores = model(user_tensor, item_tensor, type_tensor, color_tensor, section_tensor, text_tensor, image_tensor)

                _, sorted_indices = torch.sort(scores, descending=True)

                ranked_items = [items_to_rank[i] for i in sorted_indices]

                if pos_item in ranked_items[:K]:

                    hits += 1

                    rank = ranked_items.index(pos_item) + 1
                    ndcg_total += 1 / np.log2(rank + 1)

        hr = hits / len(eval_users)
        ndcg = ndcg_total / len(eval_users)

        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, HR@10: {hr:.4f}, NDCG@10: {ndcg:.4f}")

        if ndcg > best_ndcg:
            best_ndcg = ndcg
            torch.save(model.state_dict(), "best_model.pth")

