import pandas as pd
import os
import shutil
from tqdm import tqdm

# File containing the 10k items
articles_file = "data/articles_filtered.csv"

# External drive images location
images_root = "/Volumes/luffy/images"

# Output folder inside project
output_folder = "data/filtered_images"

os.makedirs(output_folder, exist_ok=True)

# Load filtered article list
articles = pd.read_csv(articles_file)

print("Total articles:", len(articles))

copied = 0
missing = 0

for article_id in tqdm(articles["article_id"]):

    image_name = "0" + str(article_id) + ".jpg"
    folder = image_name[:3]

    src = os.path.join(images_root, folder, image_name)
    dst = os.path.join(output_folder, image_name)

    if os.path.exists(src):
        shutil.copy(src, dst)
        copied += 1
    else:
        missing += 1

print("\nFiltering finished")
print("Images copied:", copied)
print("Images missing:", missing)
