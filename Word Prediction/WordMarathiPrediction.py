import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import collections
import matplotlib.pyplot as plt

# Hyperparameters
hidden_size = 512
num_layers = 2
batch_size = 128
num_epochs = 10
learning_rate = 0.001
sequence_length = 30
dropout_prob = 0.5

# Data Preprocessing
def read_data(fname):
    with open(fname, 'r') as f:
        content = f.read().replace('\n', '<eos>').split()
    return content

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = {word: idx for idx, (word, _) in enumerate(count)}
    reverse_dictionary = {idx: word for word, idx in dictionary.items()}
    return dictionary, reverse_dictionary

class TextDataset(Dataset):
    def __init__(self, data, dictionary, sequence_length):
        self.data = data
        self.dictionary = dictionary
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        inputs = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor([self.dictionary[word] for word in inputs], dtype=torch.long), \
               torch.tensor([self.dictionary[word] for word in target], dtype=torch.long)

# Model Definition
class LSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout_prob):
        super(LSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(out.size(0) * out.size(1), self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = (weight.new(self.num_layers, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers, batch_size, self.hidden_size).zero_())
        return hidden

# Accuracy Calculation
def calculate_accuracy(output, targets):
    _, predicted = torch.max(output, 1)
    correct = (predicted == targets).float().sum()
    return correct / targets.numel()

# Training and Evaluation
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device):
    train_acc_list = []
    valid_acc_list = []

    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        hidden = model.init_hidden(batch_size)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([h.data for h in hidden])

            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(output, targets.view(-1)).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)
        train_acc_list.append(train_acc)

        valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)
        valid_acc_list.append(valid_acc)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}\n')

    plot_accuracy(train_acc_list, valid_acc_list)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([h.data for h in hidden])

            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            total_loss += loss.item()
            total_acc += calculate_accuracy(output, targets.view(-1)).item()

    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    return total_loss, total_acc

# Text Generation
def generate_text(model, start_word, dictionary, reverse_dictionary, num_words, device):
    model.eval()
    words = [start_word]
    input = torch.tensor([dictionary[start_word]], dtype=torch.long).unsqueeze(0).to(device)
    hidden = model.init_hidden(1)

    with torch.no_grad():
        for _ in range(num_words):
            output, hidden = model(input, hidden)
            output_word = reverse_dictionary[output.argmax(dim=1).item()]
            words.append(output_word)
            input = torch.tensor([dictionary[output_word]], dtype=torch.long).unsqueeze(0).to(device)
    return ' '.join(words)


def main():
    # Prepare Data
    file_path = '/home/tanmay.somkuwar/implementation/mr.txt'
    raw_data = read_data(file_path)
    dictionary, reverse_dictionary = build_dataset(raw_data)

    dataset = TextDataset(raw_data, dictionary, sequence_length)

    # Split dataset into training and validation (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, optimizer
    vocab_size = len(dictionary)
    model = LSTMModel(vocab_size, hidden_size, num_layers, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Train and evaluate
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device)

    # Save model weights and dictionaries
    torch.save(model.state_dict(), 'lstm_model_weights.pth')
    torch.save(dictionary, 'dictionary.pth')
    torch.save(reverse_dictionary, 'reverse_dictionary.pth')

if __name__ == '__main__':
    main()

#BiLSTM
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import collections

# Hyperparameters
hidden_size = 512
num_layers = 2
batch_size = 32
num_epochs = 10
learning_rate = 0.0005
sequence_length = 30
dropout_prob = 0.5

# Data Preprocessing
def read_data(fname):
    with open(fname, 'r') as f:
        content = f.read().replace('\n', '<eos>').split()
    return content

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = {word: idx for idx, (word, _) in enumerate(count)}
    reverse_dictionary = {idx: word for word, idx in dictionary.items()}
    return dictionary, reverse_dictionary

class TextDataset(Dataset):
    def __init__(self, data, dictionary, sequence_length):
        self.data = data
        self.dictionary = dictionary
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        inputs = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor([self.dictionary[word] for word in inputs], dtype=torch.long), \
               torch.tensor([self.dictionary[word] for word in target], dtype=torch.long)

# Model Definition
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout_prob):
        super(BiLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers, dropout=dropout_prob, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, vocab_size)  # Multiply by 2 because it's bidirectional

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.lstm(x, hidden)
        out = out.reshape(out.size(0) * out.size(1), self.hidden_size * 2)  # Adjust for bidirectional
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        # Since it's bidirectional, the number of directions is 2
        hidden = (weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_(),
                  weight.new(self.num_layers * 2, batch_size, self.hidden_size).zero_())
        return hidden

# Accuracy Calculation
def calculate_accuracy(output, targets):
    _, predicted = torch.max(output, 1)
    correct = (predicted == targets).float().sum()
    return correct / targets.numel()

# Function to Save Model Weights
def save_model_weights(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")

# Training and Evaluation
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, save_path):
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        hidden = model.init_hidden(batch_size)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([h.data for h in hidden])

            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(output, targets.view(-1)).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}\n')

    # Save model weights after training
    save_model_weights(model, save_path)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = tuple([h.data for h in hidden])

            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            total_loss += loss.item()
            total_acc += calculate_accuracy(output, targets.view(-1)).item()

    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    return total_loss, total_acc

def main():
    # Prepare Data
    file_path = '/home/tanmay.somkuwar/implementation/mr.txt'
    raw_data = read_data(file_path)
    dictionary, reverse_dictionary = build_dataset(raw_data)

    dataset = TextDataset(raw_data, dictionary, sequence_length)

    # Split dataset into training and validation (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')
    print(device)

    # Initialize model, loss, optimizer
    vocab_size = len(dictionary)
    model = BiLSTMModel(vocab_size, hidden_size, num_layers, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # File path to save model weights
    save_path = "bilstm_model_weights.pth"

    # Train and evaluate
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, save_path)


if __name__ == '__main__':
    main()

# GRU
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import collections

# Hyperparameters
hidden_size = 512
num_layers = 2
batch_size = 32
num_epochs = 10
learning_rate = 0.0005
sequence_length = 30
dropout_prob = 0.5

# Data Preprocessing
def read_data(fname):
    with open(fname, 'r') as f:
        content = f.read().replace('\n', '<eos>').split()
    return content

def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = {word: idx for idx, (word, _) in enumerate(count)}
    reverse_dictionary = {idx: word for word, idx in dictionary.items()}
    return dictionary, reverse_dictionary

class TextDataset(Dataset):
    def __init__(self, data, dictionary, sequence_length):
        self.data = data
        self.dictionary = dictionary
        self.sequence_length = sequence_length

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        inputs = self.data[idx:idx + self.sequence_length]
        target = self.data[idx + 1:idx + self.sequence_length + 1]
        return torch.tensor([self.dictionary[word] for word in inputs], dtype=torch.long), \
               torch.tensor([self.dictionary[word] for word in target], dtype=torch.long)

# Model Definition
class GRUModel(nn.Module):
    def __init__(self, vocab_size, hidden_size, num_layers, dropout_prob):
        super(GRUModel, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, num_layers, dropout=dropout_prob, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        x = self.embedding(x)
        out, hidden = self.gru(x, hidden)
        out = out.reshape(out.size(0) * out.size(1), self.hidden_size)
        out = self.fc(out)
        return out, hidden

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        hidden = weight.new(self.num_layers, batch_size, self.hidden_size).zero_()
        return hidden

# Accuracy Calculation
def calculate_accuracy(output, targets):
    _, predicted = torch.max(output, 1)
    correct = (predicted == targets).float().sum()
    return correct / targets.numel()

# Function to Save Model Weights
def save_model_weights(model, file_path):
    torch.save(model.state_dict(), file_path)
    print(f"Model weights saved to {file_path}")

# Training and Evaluation
def train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, save_path):
    for epoch in range(num_epochs):
        model.train()
        train_loss, train_acc = 0, 0
        hidden = model.init_hidden(batch_size)

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = hidden.data  # Ensure hidden state is detached from the previous graph

            model.zero_grad()
            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
            train_acc += calculate_accuracy(output, targets.view(-1)).item()

        train_loss /= len(train_loader)
        train_acc /= len(train_loader)

        valid_loss, valid_acc = evaluate_model(model, valid_loader, criterion, device)

        print(f'Epoch [{epoch + 1}/{num_epochs}]')
        print(f'Train Loss: {train_loss:.4f}, Train Accuracy: {train_acc:.4f}')
        print(f'Validation Loss: {valid_loss:.4f}, Validation Accuracy: {valid_acc:.4f}\n')

    # Save model weights after training
    save_model_weights(model, save_path)

def evaluate_model(model, dataloader, criterion, device):
    model.eval()
    total_loss, total_acc = 0, 0
    hidden = model.init_hidden(batch_size)

    with torch.no_grad():
        for inputs, targets in dataloader:
            inputs, targets = inputs.to(device), targets.to(device)
            hidden = hidden.data  # Ensure hidden state is detached from the previous graph

            output, hidden = model(inputs, hidden)
            loss = criterion(output, targets.view(-1))
            total_loss += loss.item()
            total_acc += calculate_accuracy(output, targets.view(-1)).item()

    total_loss /= len(dataloader)
    total_acc /= len(dataloader)
    return total_loss, total_acc

def main():
    # Prepare Data
    file_path = '/home/tanmay.somkuwar/implementation/mr.txt'
    raw_data = read_data(file_path)
    dictionary, reverse_dictionary = build_dataset(raw_data)

    dataset = TextDataset(raw_data, dictionary, sequence_length)

    # Split dataset into training and validation (80% train, 20% validation)
    train_size = int(0.8 * len(dataset))
    valid_size = len(dataset) - train_size
    train_dataset, valid_dataset = torch.utils.data.random_split(dataset, [train_size, valid_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, drop_last=True)

    # Device configuration
    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    # Initialize model, loss, optimizer
    vocab_size = len(dictionary)
    model = GRUModel(vocab_size, hidden_size, num_layers, dropout_prob).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # File path to save model weights
    save_path = "gru_model_weights.pth"

    # Train and evaluate
    train_model(model, train_loader, valid_loader, criterion, optimizer, num_epochs, device, save_path)

if __name__ == '__main__':
    main()

