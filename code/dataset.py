import torch
from torch.utils.data import Dataset
import numpy as np
import os
import json

class LMDataset(Dataset):
    def __init__(self, data_dir, split):
        super().__init__()
        # load the data
        with open(os.path.join(data_dir, '%s.json'%split), 'r', encoding='utf-8') as f:
            meta = json.load(f)
        
        self.data = meta['data'] # list of samples
        self.stoi = meta['stoi'] # a dict that maps character to integer
        self.itos = meta['itos'] # a dict that maps string of integer to character
        self.vocab_size = meta['vocab_size'] # vocab size
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index]

class Converter:
    '''
    This class helps us convert strings to integers and back
    We use "0" to denote the padding character '<pad>', and "1" to denote the end of the sequence '<eos>'
    '''
    def __init__(self, stoi, itos):
        self.stoi = stoi # a dict that maps character to integer
        self.itos = itos # a dict that maps string of integer to character
    
    def single_encode(self, s):
        l = [] # initialize an empty list
        for i in s:
            l.append(self.stoi[i])
        # transform the list into a numpy array
        l = np.array(l, dtype=np.int64)
        return l 
        
    def single_decode(self, l):
        s = '' # initialize an empty string
        for i in l:
            # if we meet the end of the sequence (the value of integer is equal to 1), break
            if i == 1:
                break
            # convert string of the integer into a character
            s += self.itos[str(i)]
        return s 


    def encode(self, data):
        '''
        encode a list of strings into integers
        '''
        lens = [len(s) for s in data]
        max_len = max(lens)
        out = np.zeros((len(data), max_len+1), dtype=np.int64)
        for i,s in enumerate(data):
            out[i,:len(s)] = self.single_encode(s)
            out[i,len(s)] = 1
        x = torch.from_numpy(out[:,:-1])
        y = torch.from_numpy(out[:,1:])
        return x, y 

    def decode(self, data):
        '''
        decode a list of integers into strings
        '''
        data = data.cpu().numpy().astype(np.int64)
        out = []
        for i in range(len(data)):
            out.append(self.single_decode(data[i]))
        return out