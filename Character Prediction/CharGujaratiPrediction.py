#LSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Read the text dataset
with open('/home/tanmay.somkuwar/implementation/gujarati.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Character-level preprocessing
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

# Custom Dataset
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

dataset = TextDataset(text, input_seq_len)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = torch.utils.data.random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# Model definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        out, hidden = self.lstm(x, hidden)
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# Hyperparameters
vocab_size = len(chars)
hidden_dim = 512
n_layers = 2
n_epochs = 10

model = LSTMModel(vocab_size, hidden_dim, n_layers)
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float()
        inputs, targets = inputs.to(device), targets.to(device)

        hidden = model.init_hidden(inputs.size(0))
        hidden = tuple([each.data for each in hidden])
        model.zero_grad()
        output, hidden = model(inputs, hidden)

        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    # Validation loop
    model.eval()
    val_loss = 0.0
    correct_val = 0
    total_val = 0

    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float()
            inputs, targets = inputs.to(device), targets.to(device)

            hidden = model.init_hidden(inputs.size(0))
            hidden = tuple([each.data for each in hidden])
            output, hidden = model(inputs, hidden)

            loss = criterion(output, targets.view(-1))

            val_loss += loss.item() * inputs.size(0)
            _, pred = torch.max(output, dim=1)
            correct_val += (pred == targets.view(-1)).sum().item()
            total_val += targets.view(-1).size(0)

    val_loss /= len(val_loader.dataset)
    val_acc = 100 * correct_val / total_val

    print(f'Epoch {epoch}/{n_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

# Save the model weights
torch.save(model.state_dict(), '/home/tanmay.somkuwar/implementation/char_guj_lstm.pth')
print("Model weights saved as char_guj_lstm.pth")

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import math

# Read the dataset
with open('/home/tanmay.somkuwar/implementation/gujarati.txt', 'r', encoding='utf-8') as f:
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


# Split the data into train and validation sets
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


# Model instantiation
vocab_size = len(chars)
hidden_dim = 256
n_layers = 2
batch_size = 64

model = BiLSTMModel(vocab_size, hidden_dim, n_layers)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model.to(device)

# Training parameters
n_epochs = 10
optimizer = optim.Adam(model.parameters(), lr=0.0005)
criterion = nn.CrossEntropyLoss()

# Training loop
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0

    for inputs, targets in train_loader:
        inputs = nn.functional.one_hot(inputs, num_classes=vocab_size).float()
        inputs, targets = inputs.to(device), targets.to(device)

        hidden = model.init_hidden(inputs.size(0))
        hidden = tuple([each.data for each in hidden])
        model.zero_grad()
        output, hidden = model(inputs, hidden)

        loss = criterion(output, targets.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item() * inputs.size(0)

    train_loss /= len(train_loader.dataset)

    print(f'Epoch {epoch}/{n_epochs}, Train Loss: {train_loss:.4f}')

print("Training complete.")

# Save model weights
torch.save(model.state_dict(), 'char_guj_bilstm.pth')
print("Model weights saved to 'char_guj_bilstm.pth'.")

import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the LSTM Model class
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_dim, n_layers):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers
        self.lstm = nn.LSTM(vocab_size, hidden_dim, n_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x, hidden=None):
        if hidden is None:
            hidden = self.init_hidden(x.size(0))
        out, hidden = self.lstm(x, hidden)
        out = self.fc(out)  # No need to reshape here
        return out, hidden


    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# Load the Gujarati dataset and generate char_to_idx and idx_to_char
with open('/home/tanmay.somkuwar/implementation/gujarati.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess the dataset
chars = sorted(list(set(text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
vocab_size = len(char_to_idx)

# Initialize the model
hidden_dim = 512
n_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = LSTMModel(vocab_size, hidden_dim, n_layers)
model.to(device)

# Load the model state dict into the model
model_path = "/home/tanmay.somkuwar/implementation/Character Prediction Models/char_guj_lstm.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the BiLSTM Model class
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


# Load the Gujarati dataset and generate char_to_idx and idx_to_char
with open('/home/tanmay.somkuwar/implementation/gujarati.txt', 'r', encoding='utf-8') as f:
    text = f.read()

# Preprocess the dataset
chars = sorted(list(set(text)))
char_to_idx = {ch: idx for idx, ch in enumerate(chars)}
idx_to_char = {idx: ch for idx, ch in enumerate(chars)}
vocab_size = len(char_to_idx)

# Initialize the model
hidden_dim = 256
n_layers = 2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = BiLSTMModel(vocab_size, hidden_dim, n_layers)
model.to(device)

# Load the model state dict into the model
model_path = "/home/tanmay.somkuwar/implementation/Character Prediction Models/char_guj_bilstm.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
