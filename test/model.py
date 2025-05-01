# unified_model_training.ipynb (Updated for Revised Preprocessing)

# # Cell 1: Mount Google Drive
# from google.colab import drive
# drive.mount('/content/drive')

# Cell 2: Import Libraries
import os
import time
import torch
import pickle
import random
import numpy as np
import torch.nn as nn
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torchvision import transforms, models
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

# Cell 3: Paths and Device
video_pairs_path = 'msrvtt_data/frame_caption_pairs.pkl'
image_pairs_path = 'flicker/image_caption_pairs.pkl'
vocab_path = 'vocab.pkl'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Cell 4: Dataset Class
class CaptionDataset(Dataset):
    def __init__(self, data_pairs, vocab, transform=None):
        self.data = data_pairs
        self.vocab = vocab
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image_path, caption = self.data[idx]
        image = Image.open(image_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        tokens = [self.vocab['<start>']] + [self.vocab.get(word, self.vocab['<unk>']) for word in caption.lower().split()] + [self.vocab['<end>']]
        return image, torch.tensor(tokens)

# Cell 5: Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
])


# Cell 6: Load Datasets and Vocabulary
with open(video_pairs_path, 'rb') as f:
    video_data = pickle.load(f)

with open(image_pairs_path, 'rb') as f:
    image_data = pickle.load(f)

with open(vocab_path, 'rb') as f:
    vocab = pickle.load(f)

inv_vocab = {idx: word for word, idx in vocab.items()}
data_pairs = video_data + image_data
dataset = CaptionDataset(data_pairs, vocab, transform)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))

# CELL:6.1: validation loader & train loader
train_pairs, val_pairs = train_test_split(data_pairs, test_size=0.1, random_state=42)

train_dataset = CaptionDataset(train_pairs, vocab, transform)
val_dataset = CaptionDataset(val_pairs, vocab, transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=lambda x: tuple(zip(*x)))
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=lambda x: tuple(zip(*x)))

# Cell 7: Encoder
# class EncoderCNN(nn.Module):
#     def __init__(self, embed_size):
#         super(EncoderCNN, self).__init__()
#         resnet = models.resnet50(pretrained=True)
#         resnet.fc = nn.Identity()  # remove final FC layer
#         modules = list(resnet.children())[:-2]
#         self.resnet = nn.Sequential(*modules) # ResNet50 without the last two layers
#         self.linear = nn.Linear(resnet.fc.in_features, embed_size)
#         self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

#     def forward(self, images):
#         with torch.no_grad():
#             features = self.resnet(images)  # shape: (batch_size, 2048, 1, 1)
#             features = features.squeeze(-1).squeeze(-1)  # shape: (batch_size, 2048)
#         features = self.bn(self.linear(features))  # shape: (batch_size, embed_size)
#         return features

import torch.nn.functional as F

class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        resnet = models.resnet50(pretrained=True)
        modules = list(resnet.children())[:-2]  # Remove last FC and avgpool
        self.resnet = nn.Sequential(*modules)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # to get 2048-dim
        self.linear = nn.Linear(2048, embed_size)
        self.bn = nn.BatchNorm1d(embed_size, momentum=0.01)

    def forward(self, inputs):
        """
        inputs: (B, C, H, W) for image or (B, T, C, H, W) for video
        """
        if inputs.dim() == 4:  # Single image
            with torch.no_grad():
                features = self.resnet(inputs)  # (B, 2048, H', W')
                features = self.pool(features).squeeze(-1).squeeze(-1)  # (B, 2048)
            features = self.linear(features)  # (B, embed_size)
            features = self.bn(features)
            return features
        
        elif inputs.dim() == 5:  # Video: (B, T, C, H, W)
            B, T, C, H, W = inputs.size()
            inputs = inputs.view(B * T, C, H, W)
            with torch.no_grad():
                features = self.resnet(inputs)  # (B*T, 2048, H', W')
                features = self.pool(features).squeeze(-1).squeeze(-1)  # (B*T, 2048)
            features = self.linear(features)  # (B*T, embed_size)
            features = self.bn(features)
            features = features.view(B, T, -1)  # (B, T, embed_size)
            features = features.mean(dim=1)  # aggregate frame features (B, embed_size)
            return features

        else:
            raise ValueError("Unsupported input shape for EncoderCNN")


# Cell 8: Decoder
class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, batch_first=True)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.init_h = nn.Linear(embed_size, hidden_size)
        self.init_c = nn.Linear(embed_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions[:, :-1])
        h0 = self.init_h(features).unsqueeze(0)  # [1, batch, hidden]
        c0 = self.init_c(features).unsqueeze(0)
        hiddens, _ = self.lstm(embeddings, (h0, c0))
        outputs = self.linear(hiddens)
        return outputs



# Cell 9: Initialize and Train Model
embed_size = 256
hidden_size = 512
vocab_size = len(vocab)

encoder = EncoderCNN(embed_size).to(device)
decoder = DecoderRNN(embed_size, hidden_size, vocab_size).to(device)

params = list(decoder.parameters()) + list(encoder.linear.parameters()) + list(encoder.bn.parameters())
criterion = nn.CrossEntropyLoss(ignore_index=vocab['<pad>'])
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()),lr=1e-4)

def compute_bleu(references, candidates):
    smoothie = SmoothingFunction().method4
    scores = [
        sentence_bleu([ref], cand, smoothing_function=smoothie)
        for ref, cand in zip(references, candidates)
    ]
    return np.mean(scores)

from tqdm.notebook import tqdm

num_epochs = 30
best_val_acc = 0.0
losses = []
accuracies = []
bleu_scores = []

for epoch in range(num_epochs):
    encoder.train()
    decoder.train()
    total_loss = 0
    total_correct = 0
    total_words = 0

    print(f"\n[Epoch {epoch + 1}/{num_epochs}]")
    train_loop = tqdm(enumerate(train_loader), total=len(train_loader), leave=False)

    for batch_idx, (images, captions) in train_loop:
        images = torch.stack(images).to(device)
        captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>']).to(device)

        features = encoder(images)
        outputs = decoder(features, captions)

        targets = captions[:, 1:]
        outputs = outputs[:, :targets.size(1), :]

        loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        predicted = outputs.argmax(2)
        mask = targets != vocab['<pad>']
        correct = (predicted == targets) & mask
        total_correct += correct.sum().item()
        total_words += mask.sum().item()

        acc = 100 * total_correct / total_words
        train_loop.set_description(f"[Batch {batch_idx + 1} / {len(train_loader)}] [Loss: {loss.item():.6f}, Acc: {acc:.2f}%]")

    # Epoch summary
    epoch_loss = total_loss / len(train_loader)
    epoch_acc = total_correct / total_words
    losses.append(epoch_loss)
    accuracies.append(epoch_acc)

    print("Testing")
    encoder.eval()
    decoder.eval()
    test_loss = 0
    test_correct = 0
    test_words = 0
    references = []
    candidates = []

    val_loop = tqdm(enumerate(val_loader), total=len(val_loader), leave=False)
    with torch.no_grad():
        for test_batch_idx, (images, captions) in val_loop:
            images = torch.stack(images).to(device)
            captions = nn.utils.rnn.pad_sequence(captions, batch_first=True, padding_value=vocab['<pad>']).to(device)

            features = encoder(images)
            outputs = decoder(features, captions)

            targets = captions[:, 1:]
            outputs = outputs[:, :targets.size(1), :]

            loss = criterion(outputs.reshape(-1, vocab_size), targets.reshape(-1))
            predicted = outputs.argmax(2)
            mask = targets != vocab['<pad>']
            correct = (predicted == targets) & mask
            test_correct += correct.sum().item()
            test_words += mask.sum().item()

            acc = 100 * test_correct / test_words
            val_loop.set_description(f"[Batch {test_batch_idx + 1} / {len(val_loader)}]  [Loss: {loss.item():.6f}, Acc: {acc:.2f}%]")

            for pred_seq, target_seq in zip(predicted, targets):
                pred_tokens = [inv_vocab[idx.item()] for idx in pred_seq if idx.item() not in {vocab['<pad>'], vocab['<end>']}]
                target_tokens = [inv_vocab[idx.item()] for idx in target_seq if idx.item() not in {vocab['<pad>'], vocab['<end>']}]
                candidates.append(pred_tokens)
                references.append(target_tokens)

    val_acc = 100 * test_correct / test_words
    bleu_score = compute_bleu(references, candidates)
    bleu_scores.append(bleu_score)

    print(f"Accuracy {val_acc:.8f}")

    print(f"[Epoch {epoch+1}] Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc*100:.2f}%")
    print(f"Validation Acc: {val_acc:.2f}%, BLEU Score: {bleu_score:.4f}")

    # Save checkpoint
    checkpoint = {
        'epoch': epoch + 1,
        'encoder_state_dict': encoder.state_dict(),
        'decoder_state_dict': decoder.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_accuracy': val_acc,
        'vocab': vocab
    }
    torch.save(checkpoint, f"checkpoint_epoch_{epoch+1}.pt")

    # Save best model
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(checkpoint, "best_model.pt")
        print("✅ Best model saved.\n")

# Final save
torch.save(encoder.state_dict(), 'unified_encoder.pth')
torch.save(decoder.state_dict(), 'unified_decoder.pth')
torch.save({
    'encoder_state_dict': encoder.state_dict(),
    'decoder_state_dict': decoder.state_dict(),
    'vocab': vocab
}, 'unified_model_epoch_30.pth')

print("\n📦 Unified model saved as 'unified_model_epoch_30.pth'")
print(f"🏆 Best Accuracy: {max(accuracies) * 100:.2f}%")
print(f"📈 Final Accuracy: {accuracies[-1] * 100:.2f}%")
print(f"📝 Final BLEU Score: {bleu_scores[-1]:.4f}")
print("Training complete.")
plt.figure(figsize=(12, 5))

# Loss Plot
plt.subplot(1, 2, 1)
plt.plot(range(1, num_epochs + 1), losses, marker='o', label='Loss')
plt.title('Training Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Accuracy Plot
plt.subplot(1, 2, 2)
plt.plot(range(1, num_epochs + 1), accuracies, marker='o', color='green', label='Accuracy')
plt.title('Training Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()

epochs = list(range(1, num_epochs + 1))

plt.plot(epochs, accuracies, label="Accuracy", color='blue', marker='o')
plt.plot(epochs, bleu_scores, label="BLEU Score", color='green', marker='x')
plt.xlabel("Epoch")
plt.ylabel("Score")
plt.title("Accuracy vs BLEU Score over Epochs")
plt.legend()
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.show()
