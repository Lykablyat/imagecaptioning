import json
import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import AutoTokenizer

# Load tokenized captions
with open("tokenized_captions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

image_folder = "images"
tokenized_dataset = []

print("Starting dataset creation...")

for idx, item in enumerate(data):
    image_path = os.path.join(image_folder, item["image"])

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        continue

    try:
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image)
    except Exception as e:
        print(f"Failed to load image {image_path}: {e}")
        continue

    for caption_tokens in item["tokenized_captions"]:
        if isinstance(caption_tokens, list):
            caption_tensor = torch.tensor(caption_tokens[:50], dtype=torch.long)
            if caption_tensor.size(0) < 50:
                padding = torch.zeros(50 - caption_tensor.size(0), dtype=torch.long)
                caption_tensor = torch.cat((caption_tensor, padding))

            tokenized_dataset.append((image_tensor, caption_tensor))

    if idx % 50 == 0:
        print(f"Processed {idx} / {len(data)} images")

# Save the dataset
print(f"Saving {len(tokenized_dataset)} pairs...")
torch.save(tokenized_dataset, "image_caption_dataset.pt")
print("Done! Saved to image_caption_dataset.pt")
