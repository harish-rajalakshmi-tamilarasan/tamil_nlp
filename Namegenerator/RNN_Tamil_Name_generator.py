#!/usr/bin/env python
# coding: utf-8

# In[1]:


import torch
import torch.nn as nn
import torch.nn.init as init
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence,pack_padded_sequence, pad_packed_sequence


# In[2]:


class NamesDataset(Dataset):
    def __init__(self, names_list):
        self.names = ['.'+name+'.' for name in names]
        self.characters = sorted(set("".join(self.names)))
        self.char_to_idx = {character:idx for idx,character in enumerate(self.characters,1)}
        self.idx_to_char = {idx:character for idx,character in enumerate(self.characters,1)}
        self.char_to_idx['0']=0
        self.idx_to_char[0]='0'

    def __len__(self):
        return len(self.names)

    def __getitem__(self, idx):
        name = self.names[idx]
        sequence = [self.char_to_idx[char] for char in name]
        X = sequence[:-1]
        y = sequence[1:]
        return torch.tensor(X, dtype=torch.long),torch.tensor(y, dtype=torch.long)


# In[3]:


def collate_fn(batch):
    X,y = zip(*batch)
    length = [len(x) for x in X]
    X = pad_sequence(X,batch_first=True, padding_value=0)
    y = pad_sequence(y,batch_first=True, padding_value=0)
    return X, y, torch.tensor(length)


# In[4]:


from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# In[5]:


male_file_path = r'D:\Custom ML Implementation\ML_Projects\NameGenerator\dataset\tamil_males.txt'
female_file_path = r'D:\Custom ML Implementation\ML_Projects\NameGenerator\dataset\female_names.txt'
wordset = set()
for file_path in (male_file_path,female_file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            wordset.add(line.strip())
names = list(wordset)


# In[6]:


class NameGeneratorRNNModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size,output_size,num_layers=1):
        super(NameGeneratorRNNModel,self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size,embedding_dim)
        self.rnn = nn.RNN(embedding_dim,hidden_size, num_layers, batch_first=True, nonlinearity='tanh',dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, X, lengths, hidden):
        embedded = self.embedding(X)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.rnn(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# In[7]:


class NameGeneratorLSTMModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(NameGeneratorLSTMModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers

        self.embedding = nn.Embedding(vocab_size, embedding_dim)

        # LSTM layer (replacing RNN with LSTM)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5)

        # Fully connected layer for final output
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, X, lengths, hidden):
        # Embedding input
        embedded = self.embedding(X)

        # Pack the sequence to ignore padding in the LSTM
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        
        # LSTM forward pass
        packed_output, hidden = self.lstm(packed_input, hidden)
        
        # Unpack the output to restore padding
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        
        # Pass the output through the fully connected layer
        out = self.fc(output)
        
        return out, hidden

    def init_hidden(self, batch_size):
        # LSTM has both hidden state and cell state
        return (torch.zeros(self.num_layers, batch_size, self.hidden_size),
                torch.zeros(self.num_layers, batch_size, self.hidden_size))


# In[8]:


class NameGeneratorGRUModel(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1):
        super(NameGeneratorGRUModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.gru = nn.GRU(embedding_dim, hidden_size, num_layers, batch_first=True, dropout=0.5)
        self.fc = nn.Linear(hidden_size, output_size)
        
    def forward(self, X, lengths, hidden):
        embedded = self.embedding(X)
        packed_input = pack_padded_sequence(embedded, lengths, batch_first=True, enforce_sorted=False)
        packed_output, hidden = self.gru(packed_input, hidden)
        output, _ = pad_packed_sequence(packed_output, batch_first=True)
        out = self.fc(output)
        return out, hidden

    def init_hidden(self, batch_size):
        return torch.zeros(self.num_layers, batch_size, self.hidden_size)


# In[9]:


dataset = NamesDataset(names)
BATCH_SIZES = 256

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZES, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZES, shuffle=False,collate_fn=collate_fn)


# In[10]:


vocab_size = len(dataset.characters)+1
output_size = vocab_size


# In[11]:


hidden_size_rnn = 200
num_layers_rnn = 2
hidden_size_lstm = 200
num_layers_lstm = 2
hidden_size_gru = 200
num_layers_gru = 2
embedding_dim = 15


# In[12]:


rnn_model = NameGeneratorRNNModel(vocab_size, embedding_dim, hidden_size_rnn, vocab_size, num_layers_rnn)
lstm_model = NameGeneratorLSTMModel(vocab_size, embedding_dim, hidden_size_lstm, vocab_size, num_layers_lstm)
gru_model = NameGeneratorGRUModel(vocab_size, embedding_dim, hidden_size_gru, vocab_size, num_layers_gru)
criterion = nn.CrossEntropyLoss()
rnn_optimizer = optim.Adam(rnn_model.parameters(), lr=0.005)
lstm_optimizer = optim.Adam(lstm_model.parameters(), lr=0.005)
gru_optimizer = optim.Adam(gru_model.parameters(), lr=0.005)


# In[13]:


import torch

def train_mode(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model.train()
    train_loss = 0
    for X, y, lengths  in data_loader:
        hidden = model.init_hidden(X.size(0))
        output, hidden = model(X,lengths,hidden)
        loss = loss_fn(output.view(-1,vocab_size), y.view(-1))
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    return train_loss

def test_mode(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
              loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer):
    model.eval()
    test_loss = 0
    with torch.inference_mode():
        for X, y, lengths in data_loader:
            hidden = model.init_hidden(X.size(0))
            output, hidden = model(X, lengths, hidden)
            loss = loss_fn(output.view(-1,vocab_size), y.view(-1))
            test_loss += loss.item()
            
        test_loss /= len(data_loader)
    return test_loss


# In[14]:


def run_model(model, optimizer, epochs):
    print(f"Model: {type(model).__name__}")
    from tqdm.auto import tqdm
    torch.manual_seed(42)
    train_time_start_on_cpu = timer()
    for epoch in tqdm(range(epochs)):
        train_loss = train_mode(model,train_loader,criterion,optimizer)
        test_loss = test_mode(model,val_loader,criterion,optimizer)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {test_loss:.5f}")
    
    train_time_end_on_cpu = timer()
    total_train_time_model = print_train_time(start=train_time_start_on_cpu, 
                                               end=train_time_end_on_cpu,
                                               device=str(next(model.parameters()).device))


# In[15]:


if __name__ == "__main__":
    run_model(rnn_model, rnn_optimizer, 25)
    run_model(lstm_model, lstm_optimizer, 30)
    run_model(gru_model, gru_optimizer, 30)


# In[16]:


def generate(model,start_str='.', iterations=20):
    new_names = []
    for _ in range(iterations):
        model.eval()
        inputs = torch.tensor([dataset.char_to_idx[start_str]], dtype=torch.long).unsqueeze(0)
        hidden = model.init_hidden(1)
        output_name = ''
    
        while(True):
            output, hidden = model(inputs, torch.tensor([1]),hidden)
            probabilities = torch.softmax(output[0, -1], dim=0)
            char_idx = torch.multinomial(probabilities, 1).item()
            if dataset.idx_to_char[char_idx] == '.':
                break
            output_name += dataset.idx_to_char[char_idx]
            inputs = torch.tensor([[char_idx]], dtype=torch.long)
        if output_name not in names:
            new_names.append(output_name)   
    return new_names


# In[17]:


generate(rnn_model)


# In[18]:


generate(lstm_model)


# In[19]:


generate(gru_model)


# In[22]:


def save_model(model, model_name, save_path=r'D:\\Custom ML Implementation\\ML_Projects\\NameGenerator\\models\\'):
    model_save_path = save_path + model_name + ".pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model {model_name} saved to {model_save_path}")


# In[23]:


save_model(rnn_model, "RNN_model")
save_model(lstm_model, "LSTM_model")
save_model(gru_model, "GRU_model")


# In[25]:


def load_model(model_class, model_name, vocab_size, embedding_dim, hidden_size, output_size, num_layers=1, save_path=r"D:\\Custom ML Implementation\\ML_Projects\\NameGenerator\\models\\"):
    model = model_class(vocab_size, embedding_dim, hidden_size, output_size, num_layers)
    model_save_path = save_path + model_name + ".pth"
    model.load_state_dict(torch.load(model_save_path))
    model.eval()  # Set to evaluation mode
    print(f"Model {model_name} loaded from {model_save_path}")
    return model


# In[26]:


rnn_model_loaded = load_model(NameGeneratorRNNModel, "RNN_model", vocab_size, embedding_dim, hidden_size_rnn, vocab_size, num_layers_rnn)
lstm_model_loaded = load_model(NameGeneratorLSTMModel, "LSTM_model", vocab_size, embedding_dim, hidden_size_lstm, vocab_size, num_layers_lstm)
gru_model_loaded = load_model(NameGeneratorGRUModel, "GRU_model", vocab_size, embedding_dim, hidden_size_gru, vocab_size, num_layers_gru)


# In[ ]:




