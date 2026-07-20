import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Read the text dataset
with open('/home/tanmay.somkuwar/implementation/punjabi.txt', 'r', encoding='utf-8') as f:
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

# Variables to store metrics
val_losses, val_accuracies = [], []

# Training loop
best_val_accuracy = 0
for epoch in range(1, n_epochs + 1):
    model.train()
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
    val_losses.append(val_loss)
    val_acc = 100 * correct_val / total_val
    val_accuracies.append(val_acc)

    print(f'Epoch {epoch}/{n_epochs}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%')

    # Save the model with the best validation accuracy
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'char_punjabi_lstm.pth')


# Compute perplexity from val_loss
val_perplexity = [np.exp(loss) for loss in val_loss]

# Print the result
print("Validation Perplexity:", val_perplexity)


# Calculate perplexity
val_perplexity = [np.exp(loss) for loss in val_loss]

# Print the val_perplexity values
print("Validation Perplexity:", val_perplexity)

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
        out = out.contiguous().view(-1, self.hidden_dim)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# Load the Punjabi dataset and generate char_to_idx and idx_to_char
with open('/home/tanmay.somkuwar/implementation/punjabi.txt', 'r', encoding='utf-8') as f:
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
model_path = "/home/tanmay.somkuwar/implementation/Character Prediction Models/char_punjabi_lstm.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

