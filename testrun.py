import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoTokenizer
from PIL import Image
import matplotlib.pyplot as plt
import torchvision.transforms as T

# --- Model definition (same as training) ---
class SimpleCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_dim)
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.unsqueeze(1)
        embeddings = self.embedding(captions)
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        outputs, _ = self.rnn(inputs)
        outputs = self.fc_out(outputs)
        return outputs

# --- Load tokenizer and model ---
tokenizer = AutoTokenizer.from_pretrained("dbmdz/bert-base-turkish-cased")
vocab_size = tokenizer.vocab_size
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = SimpleCaptionModel(vocab_size, embed_dim=256, hidden_dim=512).to(device)
model.load_state_dict(torch.load("model.pt", map_location=device))
model.eval()

# --- Image preprocessing ---
image_path = "images/185139.jpg"
image = Image.open(image_path).convert("RGB")

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

img_tensor = transform(image).unsqueeze(0).to(device)

# --- Caption generation ---
max_len = 30
input_ids = torch.tensor([tokenizer.cls_token_id], device=device).unsqueeze(0)  # batch=1

with torch.no_grad():
    for _ in range(max_len):
        outputs = model(img_tensor, input_ids)
        next_token_logits = outputs[0, -1, :]
        next_token_id = torch.argmax(next_token_logits).unsqueeze(0).unsqueeze(0)
        input_ids = torch.cat([input_ids, next_token_id], dim=1)

        if next_token_id.item() == tokenizer.sep_token_id:
            break

caption_ids = input_ids[0].tolist()
caption_text = tokenizer.decode(caption_ids, skip_special_tokens=True)

# --- Display image with caption using matplotlib ---
plt.figure(figsize=(8, 6))
plt.imshow(image)
plt.axis('off')
plt.title(caption_text, fontsize=12, wrap=True)
plt.tight_layout()
plt.show()
