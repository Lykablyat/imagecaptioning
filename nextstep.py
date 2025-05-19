import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import torchvision.models as models

# --- Load dataset ---
data = torch.load("image_caption_dataset.pt")
print(f"Type of data: {type(data)}")
print(f"Length of dataset: {len(data)}")
print(f"Sample item type: {type(data[0])}, sample item: {data[0]}")

# --- Dataset class ---
class ImageCaptionDataset(Dataset):
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        img, cap = self.data_list[idx]
        if not torch.is_tensor(img):
            img = torch.tensor(img, dtype=torch.float32)
        if not torch.is_tensor(cap):
            cap = torch.tensor(cap, dtype=torch.long)
        return img, cap

def collate_fn(batch):
    images, captions = zip(*batch)
    images = torch.stack(images)
    captions_padded = pad_sequence(captions, batch_first=True, padding_value=0)
    return images, captions_padded

dataset = ImageCaptionDataset(data)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

print(f"Dataset length: {len(dataset)}")
print(f"Number of batches: {len(dataloader)}")

# --- Model ---
class SimpleCaptionModel(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.encoder = models.resnet18(pretrained=True)
        self.encoder.fc = nn.Linear(self.encoder.fc.in_features, embed_dim)

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.rnn = nn.LSTM(embed_dim, hidden_dim, batch_first=True)
        self.fc_out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, images, captions):
        features = self.encoder(images)
        features = features.unsqueeze(1)  # (batch, 1, embed_dim)
        embeddings = self.embedding(captions)  # (batch, seq_len, embed_dim)
        inputs = torch.cat((features, embeddings[:, :-1, :]), dim=1)
        outputs, _ = self.rnn(inputs)
        outputs = self.fc_out(outputs)
        return outputs

# --- Training setup ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
vocab_size = 32000  # your tokenizer vocab size

model = SimpleCaptionModel(vocab_size, embed_dim=256, hidden_dim=512).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# --- Training loop (single epoch, single batch with debug) ---
model.train()
total_loss = 0
for batch_idx, (images, captions) in enumerate(dataloader):
    print(f"Batch {batch_idx+1}: images {images.shape}, captions {captions.shape}")
    images = images.to(device)
    captions = captions.to(device)

    optimizer.zero_grad()
    outputs = model(images, captions)
    loss = F.cross_entropy(outputs[:, 1:].reshape(-1, vocab_size), captions[:, 1:].reshape(-1), ignore_index=0)
    loss.backward()
    optimizer.step()

    total_loss += loss.item()
    print(f"Sample loss: {loss.item():.4f}")

print(f"Epoch average loss: {total_loss:.4f}")
