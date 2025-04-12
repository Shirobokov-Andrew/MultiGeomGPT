import numpy as np
import pickle
from torch.utils.data import Dataset
import torch
import os
from config.train_config import TrainConfig


class CharDataset(Dataset):
    def __init__(self, dataset_path: str, block_size: int, split: str = "train"):
        self.data = np.memmap(os.path.join(dataset_path, f'{split}.bin'), 
                            dtype=np.uint16, mode='r')
        self.block_size = block_size
        with open(os.path.join(dataset_path, 'meta.pkl'), 'rb') as f:
            meta = pickle.load(f)
            self.stoi, self.itos = meta['stoi'], meta['itos']
            self.vocab_size = meta['vocab_size']
            
    def __len__(self):
        return len(self.data) - self.block_size
    
    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx:idx + self.block_size].astype(np.int64))
        y = torch.from_numpy(self.data[idx + 1:idx + 1 + self.block_size].astype(np.int64))
        return x, y

config = TrainConfig()
train_dataset = CharDataset(config.dataset_path, config.block_size, 'train')
val_dataset = CharDataset(config.dataset_path, config.block_size, 'val')
print("kek")