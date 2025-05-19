import pandas as pd
import json
import os

excel_file = "imgdescdb.xlsx"  # Change to your Excel file name

if not os.path.exists(excel_file):
    print(f"HATA: Dosya bulunamadı: {excel_file}")
    exit()

df = pd.read_excel(excel_file, header=None)

# Fill down merged cells in column 0 (image names)
df[0] = df[0].fillna(method='ffill')

dataset = []

# Each image has exactly 3 descriptions, grouped by image filename
# Group by image filename and collect 3 captions from column 2
for image_name, group in df.groupby(0):
    captions = group[2].astype(str).tolist()
    dataset.append({
        "image": image_name,
        "captions": captions
    })

with open("image_captions.json", "w", encoding="utf-8") as f:
    json.dump(dataset, f, indent=2, ensure_ascii=False)

print("✅ Dönüştürme tamamlandı. 'image_captions.json' dosyası oluşturuldu.")
