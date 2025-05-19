import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")

with open("image_captions.json", "r", encoding="utf-8") as f:
    data = json.load(f)

tokenized_data = []

for item in data:
    image_name = item["image"]
    captions = item["captions"]
    tokenized_captions = []

    for caption in captions:
        tokens = tokenizer.encode(caption, add_special_tokens=True)
        tokenized_captions.append(tokens)
        print(f"Caption: {caption}")
        print(f"Tokens: {tokens}\n")

    tokenized_data.append({
        "image": image_name,
        "tokenized_captions": tokenized_captions
    })

# Optionally save tokenized captions to a new JSON file
with open("tokenized_captions.json", "w", encoding="utf-8") as f:
    json.dump(tokenized_data, f, ensure_ascii=False, indent=2)
