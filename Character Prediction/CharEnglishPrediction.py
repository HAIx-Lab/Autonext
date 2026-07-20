import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from torchinfo import summary

# Read the text dataset
with open('/home/tanmay.somkuwar/implementation/sherlock_holmes_stories.txt', 'r', encoding='utf-8') as f:
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
optimizer = optim.Adam(model.parameters(), lr=0.002)
criterion = nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

best_train_acc = 0
best_val_acc = 0
best_train_perplexity = float('inf')
best_val_perplexity = float('inf')

# Training loop
for epoch in range(1, n_epochs + 1):
    model.train()
    train_loss = 0.0
    correct_train = 0
    total_train = 0

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
        _, pred = torch.max(output, dim=1)
        correct_train += (pred == targets.view(-1)).sum().item()
        total_train += targets.view(-1).size(0)

    train_loss /= len(train_loader.dataset)
    train_acc = 100 * correct_train / total_train
    train_perplexity = np.exp(train_loss)
    if train_acc > best_train_acc:
        best_train_acc = train_acc
    if train_perplexity < best_train_perplexity:
        best_train_perplexity = train_perplexity

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
    val_perplexity = np.exp(val_loss)
    if val_acc > best_val_acc:
        best_val_acc = val_acc
    if val_perplexity < best_val_perplexity:
        best_val_perplexity = val_perplexity

    print(f'Epoch {epoch}/{n_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Perplexity: {train_perplexity:.2f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Perplexity: {val_perplexity:.2f}')

# Print best metrics
print(f'Best Training Accuracy: {best_train_acc:.2f}%')
print(f'Best Training Perplexity: {best_train_perplexity:.2f}')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'Best Validation Perplexity: {best_val_perplexity:.2f}')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from torchinfo import summary

# Read the dataset
with open('/home/tanmay.somkuwar/implementation/sherlock_holmes_stories.txt', 'r', encoding='utf-8') as f:
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

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

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
batch_size = 32

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

print(f'\nBest Training Accuracy: {best_train_acc:.2f}%')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import matplotlib.pyplot as plt
import math

# Check if CUDA is available
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
seq_length = 100
batch_size = 32
embedding_dim = 100
hidden_dim = 512
num_layers = 2
learning_rate = 0.0005
num_epochs = 10

# Load and preprocess data
with open("/home/tanmay.somkuwar/implementation/sherlock_holmes_stories.txt", 'r') as f:
    text = f.read()

chars = sorted(set(text))
vocab_size = len(chars)
char_to_idx = {char: idx for idx, char in enumerate(chars)}
idx_to_char = {idx: char for idx, char in enumerate(chars)}

# Encode the text into integers
encoded_text = np.array([char_to_idx[c] for c in text])

# Create dataset
class CharDataset(Dataset):
    def __init__(self, data, seq_length):
        self.data = data
        self.seq_length = seq_length

    def __len__(self):
        return len(self.data) - self.seq_length

    def __getitem__(self, idx):
        seq = self.data[idx:idx+self.seq_length]
        target = self.data[idx+1:idx+self.seq_length+1]
        return torch.tensor(seq, dtype=torch.long), torch.tensor(target, dtype=torch.long)

dataset = CharDataset(encoded_text, seq_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the GRU model with dropout and layer normalization
class GRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_layers):
        super(GRUModel, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_dim, num_layers,
                          batch_first=True, dropout=0.2)
        self.layer_norm = nn.LayerNorm(hidden_dim)
        self.fc = nn.Linear(hidden_dim, vocab_size)

    def forward(self, x):
        x = self.embedding(x)
        out, _ = self.gru(x)
        out = self.layer_norm(out)
        out = self.fc(out)
        return out

model = GRUModel(vocab_size, embedding_dim, hidden_dim, num_layers).to(device)

# Loss and optimizer with L2 regularization
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)

# Learning rate scheduler
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5)

# Metrics tracking
val_acc_list, val_loss_list, val_perplexity_list = [], [], []

# Training loop
best_val_accuracy = 0
for epoch in range(num_epochs):
    model.train()
    train_loss = 0
    train_correct = 0
    train_total = 0

    for inputs, targets in dataloader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)

        loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()

        train_loss += loss.item()
        _, predicted = torch.max(outputs, dim=2)
        train_total += targets.numel()
        train_correct += (predicted == targets).sum().item()

    train_accuracy = 100 * train_correct / train_total
    train_loss /= len(dataloader)

    # Validation
    model.eval()
    val_loss = 0
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            loss = criterion(outputs.view(-1, vocab_size), targets.view(-1))
            val_loss += loss.item()

            _, predicted = torch.max(outputs, dim=2)
            val_total += targets.numel()
            val_correct += (predicted == targets).sum().item()

    val_accuracy = 100 * val_correct / val_total
    val_loss /= len(dataloader)
    val_perplexity = math.exp(val_loss)

    # Save metrics
    val_acc_list.append(val_accuracy)
    val_loss_list.append(val_loss)
    val_perplexity_list.append(val_perplexity)

    scheduler.step(val_loss)

    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.2f}%, "
          f"Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_accuracy:.2f}%, "
          f"Validation Perplexity: {val_perplexity:.4f}")

    # Save the model with the best validation accuracy
    if val_accuracy > best_val_accuracy:
        best_val_accuracy = val_accuracy
        torch.save(model.state_dict(), 'best_model.pth')


