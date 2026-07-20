import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torchinfo import summary

# Read the dataset
with open('/home/tanmay.somkuwar/implementation/hindi.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess the dataset
chars = sorted(list(set(text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}

input_seq_len = 100  # Length of each input sequence
output_seq_len = 1   # Predict 1 character at a time


def char_tensor(text):
    """Convert a string to a tensor of indices"""
    tensor = torch.zeros(len(text)).long()
    for i, c in enumerate(text):
        tensor[i] = char_to_idx[c]
    return tensor


class TextDataset(Dataset):
    def __init__(self, text, seq_len):
        self.text = text
        self.seq_len = seq_len

    def __len__(self):
        return len(self.text) - self.seq_len

    def __getitem__(self, idx):
        chunk = self.text[idx:idx + self.seq_len + 1]
        input_seq = char_tensor(chunk[:-1])
        target_seq = char_tensor(chunk[1:])
        return input_seq, target_seq


# Split the data into train and validation sets using train_test_split to avoid data leakage
dataset = TextDataset(text, input_seq_len)
train_indices, val_indices = train_test_split(range(len(dataset)), test_size=0.2, random_state=42)

train_dataset = torch.utils.data.Subset(dataset, train_indices)
val_dataset = torch.utils.data.Subset(dataset, val_indices)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Define the model
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Multiply by 2 for bidirectional

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))  # Automatically initialize hidden state if not provided
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim * 2)  # Multiply by 2 for bidirectional
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(),  # Multiply by 2 for bidirectional
                  weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        return hidden

# Model instantiation and summary
vocab_size = len(chars)
hidden_dim = 256
n_layers = 2
batch_size = 64

model = BiLSTMModel(vocab_size, hidden_dim, n_layers)
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

n_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# Track the best training and validation accuracy
best_train_acc = 0
best_val_acc = 0

for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

    for inputs, targets in train_loader:
        inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float()
        inputs, targets = inputs.to(device), targets.to(device)

        hidden = model.init_hidden(inputs.size(0))  # Initialize hidden state with batch size
        hidden = tuple([each.data for each in hidden])
        model.zero_grad()
        output, hidden = model(inputs, hidden)

        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)
        _, pred = torch.max(output, dim=1)
        correct_train += (pred == targets.view(-1)).sum().item()
        total_train += targets.view(-1).size(0)

    train_loss /= len(train_loader.dataset)
    train_acc = 100 * correct_train / total_train
    best_train_acc = max(best_train_acc, train_acc)

    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float()
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0))  # Initialize hidden state with batch size
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(inputs, hidden)

            loss = criterion(output, targets.view(-1))

            val_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(output, dim=1)
            correct_val += (pred == targets.view(-1)).sum().item()
            total_val += targets.view(-1).size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * correct_val / total_val
    best_val_acc = max(best_val_acc, val_acc)

    print(f'Epoch {epoch}/{n_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save model after each epoch
    torch.save(model.state_dict(), "char_hindi_bilstm.pth")
    print("Model saved as char_hindi_bilstm.pth")

print(f'\nBest Training Accuracy: {best_train_acc:.2f}%')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

import torch
import torch.nn.functional as F
import torch.nn as nn

with open('/home/tanmay.somkuwar/implementation/hindi.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess the dataset
chars = sorted(list(set(text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}

vocab_size = len(chars)
hidden_dim = 256
n_layers = 2
batch_size = 64

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
input_seq_len = 100  # Length of each input sequence

class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers):
        super(BiLSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_dim * 2, vocab_size)  # Multiply by 2 for bidirectional

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))  # Automatically initialize hidden state if not provided
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim * 2)  # Multiply by 2 for bidirectional
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_(),  # Multiply by 2 for bidirectional
                  weight.new(self.n_layers * 2, batch_size, self.hidden_dim).zero_())
        return hidden

model = BiLSTMModel(vocab_size, hidden_dim, n_layers)


