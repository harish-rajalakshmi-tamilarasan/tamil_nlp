
import torch
from timeit import default_timer as timer 

def print_train_time(start: float, end: float, device: torch.device = None):
    
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time

def train_mode(model: torch.nn.Module, data_loader: torch.utils.data.DataLoader, 
               loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer, vocab_size: int):
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
              loss_fn: torch.nn.Module, optimizer: torch.optim.Optimizer,  vocab_size: int):
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

def run_model(model, optimizer, epochs,train_loader, val_loader, criterion, vocab_size):
    print(f"Model: {type(model).__name__}")
    from tqdm.auto import tqdm
    torch.manual_seed(42)
    train_time_start_on_cpu = timer()
    for epoch in tqdm(range(epochs)):
        train_loss = train_mode(model,train_loader,criterion,optimizer,vocab_size)
        test_loss = test_mode(model,val_loader,criterion,optimizer,vocab_size)
        print(f"Epoch: {epoch}, Train loss: {train_loss:.5f}, Test loss: {test_loss:.5f}")
    
    train_time_end_on_cpu = timer()
    print_train_time(start=train_time_start_on_cpu, 
                                               end=train_time_end_on_cpu,
                                               device=str(next(model.parameters()).device))


def generate(model,start_str='.', iterations=20,dataset=None,names=None):
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

def save_model(model, model_name, save_path=r'models\\'):
    model_save_path = save_path + model_name + ".pth"
    torch.save(model.state_dict(), model_save_path)
    print(f"Model {model_name} saved to {model_save_path}")