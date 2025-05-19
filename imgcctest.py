import json
from PIL import Image
import matplotlib.pyplot as plt

with open("image_captions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

image_folder = "images/"

for item in data:
    image_path = image_folder + item["image"]
    captions = item["captions"]

    img = Image.open(image_path)

    plt.figure(figsize=(8, 10))  # Adjust size as needed
    plt.imshow(img)
    plt.axis('off')

    # Join captions into one string separated by new lines
    captions_text = "\n".join(captions)

    # Add captions below image, adjust position with 'y' param
    plt.gcf().text(0.5, 0.02, captions_text, fontsize=10, ha='center', wrap=True)

    plt.show()
