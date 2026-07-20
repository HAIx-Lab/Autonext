import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np

# Read the text dataset
with open('/home/tanmay.somkuwar/implementation/mr.txt', 'r', encoding='utf-8') as f:
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

train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=128, shuffle=False)

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
optimizer = optim.Adam(model.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# Move model to GPU if available
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
model.to(device)

# Variables to store best accuracies and perplexities
best_train_acc = 0
best_val_acc = 0
best_train_perplexity = float('inf')
best_val_perplexity = float('inf')

# Lists to store metrics
train_accuracies = []
val_accuracies = []
train_losses = []
val_losses = []
train_perplexities = []
val_perplexities = []

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

    train_accuracies.append(train_acc)
    train_losses.append(train_loss)
    train_perplexities.append(train_perplexity)

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

    val_accuracies.append(val_acc)
    val_losses.append(val_loss)
    val_perplexities.append(val_perplexity)

    print(f'Epoch {epoch}/{n_epochs}')
    print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%, Train Perplexity: {train_perplexity:.2f}')
    print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%, Val Perplexity: {val_perplexity:.2f}')

# Print best metrics
print(f'Best Training Accuracy: {best_train_acc:.2f}%')
print(f'Best Training Perplexity: {best_train_perplexity:.2f}')
print(f'Best Validation Accuracy: {best_val_acc:.2f}%')
print(f'Best Validation Perplexity: {best_val_perplexity:.2f}')

# Save the best model
torch.save(model.state_dict(), 'char_marathi_lstm.pth')

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

# Load the Marathi dataset and generate char_to_idx and idx_to_char
with open('/home/tanmay.somkuwar/implementation/mr.txt', 'r', encoding='utf-8') as f:
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
model_path = "/home/tanmay.somkuwar/implementation/Character Prediction Models/char_marathi_lstm.pth"
model.load_state_dict(torch.load(model_path))
model.eval()

# Define the prediction function to predict top 4 characters
def predict_top_4_chars(model, char_input, char_to_idx, idx_to_char, vocab_size, device):
    model.eval()
    input_indices = [char_to_idx[char] for char in char_input]
    input_tensor = F.one_hot(torch.tensor(input_indices), num_classes=vocab_size).float().unsqueeze(0).to(device)

    output, _ = model(input_tensor)
    output_dist = F.softmax(output[:, -1, :], dim=1)  # Apply softmax on the last output timestep

    # Get the top 4 predictions
    top_4_chars = torch.topk(output_dist, 4).indices.squeeze(0).tolist()
    top_4_chars = [idx_to_char[idx] for idx in top_4_chars]
    return top_4_chars

# Evaluation method to predict top 4 characters for the entire sentence
def evaluate(model, start_sequence, actual_sequence, char_to_idx, idx_to_char, vocab_size, device):
    model.eval()
    assert len(start_sequence) >= 3, "Start sequence must be at least 3 characters long"
    correct_predictions = 0
    total_predictions = 0
    generated_text = start_sequence
    current_sequence = start_sequence
    print(f"Input Sequence: {start_sequence}")

    for i in range(len(start_sequence), len(actual_sequence)):
        next_char = actual_sequence[i]
        top_4_chars = predict_top_4_chars(model, current_sequence, char_to_idx, idx_to_char, vocab_size, device)
        print(f"Next characters after '{current_sequence}': {top_4_chars}")

        if next_char in top_4_chars:
            correct_predictions += 1
        total_predictions += 1
        generated_text += next_char
        current_sequence = current_sequence[1:] + next_char

    accuracy = (correct_predictions / total_predictions) * 100 if total_predictions > 0 else 0
    print(f"Correct Predictions: {correct_predictions}/{total_predictions}")
    print(f"Accuracy: {accuracy:.2f}%")
    return generated_text

# Compute perplexity using the formula
val_perplexity = [np.exp(loss) for loss in val_loss]

# Print the perplexity list
print("Validation Perplexity:", val_perplexity)

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
        out = out.contiguous().view(-1, self.hidden_dim)  # Flatten for the fully connected layer
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden


# Load the Marathi dataset and generate char_to_idx and idx_to_char
with open('/home/tanmay.somkuwar/implementation/mr.txt', 'r', encoding='utf-8') as f:
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
model_path = "/home/tanmay.somkuwar/implementation/Character Prediction Models/char_marathi_lstm.pth"
model.load_state_dict(torch.load(model_path))
model.eval()


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
        out = out.contiguous().view(-1, self.hidden_dim)  # Flattening the output
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                  weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
        return hidden

# Load the Marathi dataset and generate char_to_idx and idx_to_char
with open('/home/tanmay.somkuwar/implementation/mr.txt', 'r', encoding='utf-8') as f:
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
model_path = "/home/tanmay.somkuwar/implementation/Character Prediction Models/char_marathi_lstm.pth"
model.load_state_dict(torch.load(model_path))
model.eval()
