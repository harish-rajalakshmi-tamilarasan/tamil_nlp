import torch
from rnn_model import NameGeneratorRNNModel, vocab_size, hidden_size, num_layers, embedding_dim, output_size, dataset
from lstm_model import NameGeneratorLSTMModel
from gru_model import NameGeneratorGRUModel
from dataset_utils import tamil_names
import os

current_dir = os.path.dirname(__file__)
base_dir = os.path.join(current_dir, 'models')

def generate(model,start_str='.',iterations=20,names = tamil_names):
    new_names = []
    for _ in range(iterations):
        hidden = model.init_hidden(1)
        if start_str != '.':
            inputs = torch.tensor([dataset.char_to_idx['.']], dtype=torch.long).unsqueeze(0)
            output, hidden = model(inputs, torch.tensor([1]), hidden)
            output_name = start_str
        else:
            output_name = ''
        inputs = torch.tensor([dataset.char_to_idx[start_str]], dtype=torch.long).unsqueeze(0)
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

def generate_rnn(start_str='.', iterations=20):
    print("start_str=",start_str)
    model = NameGeneratorRNNModel(vocab_size, embedding_dim, hidden_size, output_size, num_layers)
    model_path = os.path.join(base_dir, 'rnn_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return generate(model,start_str,iterations)

def generate_lstm(start_str='.', iterations=20):
    model = NameGeneratorLSTMModel(vocab_size, embedding_dim, hidden_size, output_size, num_layers)
    model_path = os.path.join(base_dir, 'lstm_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return generate(model,start_str,iterations)


def generate_gru(start_str='.', iterations=20):
    model = NameGeneratorGRUModel(vocab_size, embedding_dim, hidden_size, output_size, num_layers)
    model_path = os.path.join(base_dir, 'gru_model.pth')
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return generate(model,start_str,iterations)

if __name__ == "__main__":
    print(generate_rnn(start_str='அ', iterations=20))
    print(generate_lstm(start_str='அ', iterations=20))
    print(generate_gru(start_str='அ', iterations=20))


# def generate(model,start_str='.',iterations=20,dataset=None, names=None):
#     new_names = []
#     for _ in range(iterations):
       
#         inputs = torch.tensor([dataset.char_to_idx[start_str]], dtype=torch.long).unsqueeze(0)
#         hidden = model.init_hidden(1)
#         output_name = start_str
    
#         while(True):
#             output, hidden = model(inputs, torch.tensor([1]),hidden)
#             probabilities = torch.softmax(output[0, -1], dim=0)
#             char_idx = torch.multinomial(probabilities, 1).item()
#             if dataset.idx_to_char[char_idx] == '.':
#                 break
#             output_name += dataset.idx_to_char[char_idx]
#             inputs = torch.tensor([[char_idx]], dtype=torch.long)
#         if output_name not in names:
#             new_names.append(output_name)   
#     return new_names