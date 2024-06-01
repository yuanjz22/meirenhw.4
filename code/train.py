import os
import time
import math
import pickle
from contextlib import nullcontext
import argparse

import numpy as np
import torch
from torch.utils.data import DataLoader

from model import GPT, GPTConfig
from dataset import LMDataset, Converter
import matplotlib.pyplot as plt

# learning rate decay scheduler (cosine with warmup)
def get_lr(it, learning_rate, min_lr=1e-4, warmup_iters=100, lr_decay_iters=6000):
    # 1) linear warmup for warmup_iters steps
    if it < warmup_iters:
        return learning_rate * it / warmup_iters
    # 2) if it > lr_decay_iters, return min learning rate
    if it > lr_decay_iters:
        return min_lr
    # 3) in between, use cosine decay down to min learning rate
    decay_ratio = (it - warmup_iters) / (lr_decay_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio)) # coeff ranges 0..1
    return min_lr + coeff * (learning_rate - min_lr)

def train(data_root, model_name, batch_size, n_iters, ckpt_path, val_interval, device='cpu', no_res=False, no_pos=False):
    train_dataset = LMDataset(data_root, 'train')
    val_dataset = LMDataset(data_root, 'val')
    train_loader = DataLoader(train_dataset, batch_size=int(batch_size), shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    converter = Converter(train_dataset.stoi, train_dataset.itos)

    # adamw optimizer
    learning_rate = 5e-3 # max learning rate
    weight_decay = 1e-1
    beta1 = 0.9
    beta2 = 0.99
    grad_clip = 1.0 # clip gradients at this value, or disable if == 0.0
    
    # system
    
    dtype = 'bfloat16' if device == 'cpu' else 'float16' # 'float32', 'bfloat16', or 'float16', the latter will auto implement a GradScaler
    ptdtype = {'float32': torch.float32, 'bfloat16': torch.bfloat16, 'float16': torch.float16}[dtype]
    ctx = nullcontext() if device == 'cpu' else torch.autocast(device_type=device, dtype=ptdtype)
    #ctx = torch.autocast(device_type=device, dtype=ptdtype)
    best_val_loss = 1e9
    iter_num = 0 # number of iterations in the lifetime of this process

    # model init
    model_args = GPTConfig[model_name]
    model_args['vocab_size'] = train_dataset.vocab_size
    model_args['max_seq_len'] = 128
    model_args['no_res'] = no_res
    model_args['no_pos'] = no_pos

    # init a new model from scratch
    print("Initializing a new model from scratch")
    model = GPT(**model_args)

    model.to(device)

    # initialize a GradScaler. If enabled=False scaler is a no-op
    scaler = torch.cuda.amp.GradScaler(enabled=(dtype == 'float16'))

    # optimizer
    optim_groups = model.configure_optimizers(weight_decay)
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=(beta1, beta2))
    checkpoint = None # free up memory

    print('training...')
    # training loop    
    epoch_num = np.ceil(n_iters * int(batch_size) / float(len(train_dataset))).astype(np.int32)
    t0 = time.time()
    model.train()
    train_losses = []
    val_losses = []
    for epoch in range(epoch_num):
        for step, inputs in enumerate(train_loader):
            if iter_num >= n_iters:
                break
            X, Y = converter.encode(inputs)
            X, Y = X.to(device), Y.to(device)
            lr = get_lr(iter_num, learning_rate, lr_decay_iters=n_iters)
            for param_group in optimizer.param_groups:
                param_group['lr'] = lr

            # forward backward update, with optional gradient accumulation to simulate larger batch size
            # and using the GradScaler if data type is float16
            with ctx:
                logits, loss = model(X, Y)
                loss = loss  # scale the loss to account for gradient accumulation
            
            # backward pass, with gradient scaling if training in fp16
            scaler.scale(loss).backward()
            # clip the gradient
            if grad_clip != 0.0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            # step the optimizer and scaler if training in fp16
            scaler.step(optimizer)
            scaler.update()
            # flush the gradients as soon as we can, no need for this memory anymore
            optimizer.zero_grad(set_to_none=True)

            iter_num += 1
            train_losses.append(loss.item())
            # evaluate the loss on train/val sets and write checkpoints
            if iter_num % val_interval == 0:
                # timing and logging
                t1 = time.time()
                dt = t1 - t0
                t0 = t1
                lossf = loss.item()
                print(f"iter {iter_num}: loss {lossf:.4f}, time {dt*1000:.2f}ms")
                losses = estimate_loss(model, val_loader, converter, ctx, device)
                val_losses.append(losses['val'])
                print(f"iter {iter_num}: val loss {losses['val']:.4f}")
                print(f"saving latest checkpoint to {ckpt_path}")
                checkpoint = {
                        'state_dict': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'model_args': model_args,
                        'iter_num': iter_num,
                        'best_val_loss': best_val_loss,
                    }
                torch.save(checkpoint, os.path.join(ckpt_path, 'latest.pth'))

                if losses['val'] < best_val_loss:
                    best_val_loss = losses['val']
                    if iter_num > 0:
                        print(f"saving best checkpoint to {ckpt_path}")
                        torch.save(checkpoint, os.path.join(ckpt_path, 'best.pth'))

    plot(n_iters, train_losses, val_losses, val_interval, ckpt_path)

def plot(n_iters, train_losses, val_losses, val_interval, ckpt_path):
    # create a plot
    f, ax = plt.subplots(1,2,figsize=(18,6))
    val_iters = np.arange(1, n_iters+1, val_interval)

    # draw loss
    ax[0].plot(train_losses)
    ax[0].plot(val_iters, val_losses, 'r')

    # set labels
    ax[0].set_xlabel('training iters')
    ax[0].legend(['training loss', 'validation loss'])

    train_perplexity = [np.exp(x) for x in train_losses]
    val_perplexity = [np.exp(x) for x in val_losses]
    # draw perplexity
    ax[1].plot(train_perplexity)
    ax[1].plot(val_iters, val_perplexity, 'r')

    # set labels
    ax[1].set_xlabel('training iters')
    ax[1].legend(['training perplexity', 'validation perplexity'])
    plt.tight_layout()

    # show the image
    plt.savefig(os.path.join(ckpt_path, 'loss&perplexity.jpg'), dpi=300)
    plt.show()

# helps estimate an arbitrarily accurate loss over either split using many batches
@torch.no_grad()
def estimate_loss(model, val_loader, converter, ctx, device):
    out = {}
    model.eval()
    losses = 0
    max_iters = 100
    iter_num = 0
    for inputs in val_loader:
        if iter_num >= max_iters:
            break
        iter_num += 1
        X, Y = converter.encode(inputs)
        X, Y = X.to(device), Y.to(device)
        with ctx:
            logits, loss = model(X, Y)
            #loss = model.loss(logits, Y)
        losses += loss.item()
    out['val'] = losses / max_iters
    model.train()
    return out

if __name__ == '__main__':
    # set random seed for reproducibility
    seed = 2024
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = True # allow tf32 on matmul
    torch.backends.cudnn.allow_tf32 = True # allow tf32 on cudnn

    # set configurations of the model and training process
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, default='data/quansongci', help='file of training and validation data')
    parser.add_argument('--model_name', type=str, default='mygpt', help='name of the pretrained model')
    parser.add_argument('--iters', type=int, default=1000, help='number of training epochs')
    parser.add_argument('--batchsize', type=int, default=16, help='training batch size')
    parser.add_argument('--ckpt_path', type=str, default='workdirs/quansongci', help='path to save checkpoints')
    parser.add_argument('--val_interval', type=int, default=20, help='iter intervals of validation')
    parser.add_argument('--no_res', action='store_true', help='whether to use residual connection')
    parser.add_argument('--no_pos', action='store_true', help='whether to use positional encoding')
    parser.add_argument('--device', type=str, help='cpu or cuda')

    opt = parser.parse_args()
    if opt.device is None:
        opt.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    os.makedirs(opt.ckpt_path, exist_ok=True)
    train(opt.data_root, opt.model_name, opt.batchsize, opt.iters, opt.ckpt_path, opt.val_interval, opt.device, opt.no_res, opt.no_pos)


