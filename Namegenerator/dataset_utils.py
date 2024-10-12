import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset, random_split
import matplotlib.pyplot as plt
from torch.nn.utils.rnn import pad_sequence
import os

class NamesDataset(Dataset):
    def __init__(self, names_list):
        self.names = ['.'+name+'.' for name in names_list]
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


def collate_fn(batch: list) -> tuple:
    X,y = zip(*batch)
    length = [len(x) for x in X]
    X = pad_sequence(X,batch_first=True, padding_value=0)
    y = pad_sequence(y,batch_first=True, padding_value=0)
    return X, y, torch.tensor(length)

def parse_names(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        wordset = {line.strip() for line in file}
    return wordset

def get_train_test_dataset(batch_sizes, dataset):
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_sizes, shuffle=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=batch_sizes, shuffle=False,collate_fn=collate_fn)
    return train_loader, val_loader

def get_dataset(batch_sizes,data="tamil"):
    dataset = tamil_dataset if data == "tamil" else english_dataset
    train_loader, val_loader = get_train_test_dataset(batch_sizes, dataset)
    return dataset, train_loader, val_loader

current_dir = os.path.dirname(__file__)
base_dir = os.path.join(current_dir, 'dataset')
male_file_path = os.path.join(base_dir, 'tamil_males.txt')
female_file_path = os.path.join(base_dir, 'female_names.txt')
english_file_path = os.path.join(base_dir, 'names.txt')

tamil_male_names = parse_names(male_file_path)
tamil_female_names = parse_names(female_file_path)
tamil_names = tamil_male_names.union(tamil_female_names)
english_names = parse_names(english_file_path)

tamil_dataset = NamesDataset(tamil_names)
english_dataset = NamesDataset(english_names)
