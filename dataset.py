import os
from typing import Optional, List
import torch
from torch.utils.data import Dataset

import nltk
nltk.download('brown')
from nltk.corpus import brown
corpus_words = brown.words()


class RawTextDataset(Dataset):
    def __init__(self, words, context_size=1024):
        self.words = words
        self.context_size = context_size

    def __len__(self):
        return len(self.words) - self.context_size

    def __getitem__(self, idx):
        x = self.words[idx:idx+self.context_size]      # raw text
        y = self.words[idx+1:idx+self.context_size+1]  # next word prediction
        return x, y
    
def get_dataloader(batch_size=8, context_size=1024):
    dataset = RawTextDataset(corpus_words, context_size=context_size)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
    return dataloader