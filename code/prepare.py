"""
Prepare the dataset for character-level language modeling.
So instead of encoding with GPT-2 BPE tokens, we just map characters to ints.
"""
import os
import numpy as np
import argparse
import json

parser = argparse.ArgumentParser()
parser.add_argument('--data_root', type=str, default='data/quansongci', help='data directory')
args = parser.parse_args()

# set the input file path
input_file_path = os.path.join(args.data_root, 'data.json')

with open(input_file_path, 'r', encoding='utf-8') as f:
    data = json.load(f)['data']
print(f"length of dataset: {len(data):,}")

# get all the unique characters that occur in this text
chars = sorted(list(set(''.join(data))))
vocab_size = len(chars) + 2 # for <pad> and <eos>
print("all the unique characters:", ''.join(chars))
print(f"vocab size: {vocab_size:,}")

# create a mapping from characters to integers
stoi = { ch:i+2 for i,ch in enumerate(chars) }
itos = { i+2:ch for i,ch in enumerate(chars) }
stoi['<pad>'] = 0
itos[0] = '<pad>'
stoi['<eos>'] = 1
itos[1] = '<eos>'


# create the train and test splits
n = len(data)
train_data = data[:int(n*0.9)]
val_data = data[int(n*0.9):]
print(f"train has {len(train_data):,} samples")
print(f"val has {len(val_data):,} samples")

# save the meta information as well, to help us encode/decode later
train_meta = {
    'data': train_data,
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(args.data_root, 'train.json'), 'w', encoding='utf-8') as f:
    json.dump(train_meta, f, ensure_ascii=False, indent=4)

val_meta = {
    'data': val_data,
    'vocab_size': vocab_size,
    'itos': itos,
    'stoi': stoi,
}
with open(os.path.join(args.data_root, 'val.json'), 'w', encoding='utf-8') as f:
    json.dump(val_meta, f, ensure_ascii=False, indent=4)

