# ========================================================
#             Media and Cognition
#             Homework 4  Sequence Modeling
#             model.py - Model definition
#             Student ID:2022010657
#             Name:元敬哲
#             Tsinghua University
#             (C) Copyright 2024
# ========================================================


# Import required libraries
############################################################
import math
import torch
import torch.nn as nn 
from torch.nn import functional as F
import numpy as np

############################################################

# Define the GELU activation function used in OpenAI GPT
############################################################
def gelu(z):
    """
    Reference: Gaussian Error Linear Units (GELU) paper: https://arxiv.org/abs/1606.08415
    0.5z(1 + tanh[(2/π)^(1/2) * (z + 0.044715 z^3)])
    """
    return 0.5 * z * (1.0 + torch.tanh(math.sqrt(2.0 / math.pi) * (z + 0.044715 * torch.pow(z, 3.0))))

############################################################

# Define the Multi-Head SelfAttention module
############################################################
class SelfAttention(nn.Module):

    def __init__(self, embed_dim, num_head, dropout):
        super().__init__()

        # define there linear layers for q, k, v generation separately
        self.q_layer = nn.Linear(embed_dim, embed_dim)
        self.k_layer = nn.Linear(embed_dim, embed_dim)
        self.v_layer = nn.Linear(embed_dim, embed_dim)

        # define the projection layer for output
        self.proj_layer = nn.Linear(embed_dim, embed_dim)

        # define the dropout layer for attention and output calculation
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)

        self.num_head = num_head
        self.head_dim = embed_dim // num_head
    
    def forward(self, x):

        batch_size, seq_len, dim = x.shape

        # >>> TODO 1: complete the forward process of the Multi-Head SelfAttention module.
        # Step 1.1: obtain q, k, v via self.q_layer(), self.k_layer(), self.v_layer() respectively.
        # the shape of q, k, v: (batch_size, seq_len, num_heads * head_dim)
        # num_heads denotes the number of heads in multi-head attention, head_dim denotes the dimension of each head
        q = self.q_layer(x)
        k = self.k_layer(x)
        v = self.v_layer(x)

        # Step 1.2: in order to calculate multi-head attention in parallel, reshape q, k, v first.
        # first use `Tensor.reshape()` to reshape q, k, v to: (batch_size, seq_len, num_heads, head_dim)
        q = q.reshape(batch_size,seq_len,self.num_head,self.head_dim)
        k = k.reshape(batch_size,seq_len,self.num_head,self.head_dim)
        v = v.reshape(batch_size,seq_len,self.num_head,self.head_dim)

        # then use `Tensor.transpose()` or `Tensor.permute()` to exchange the dim of q, k, v for matrix multiplication
        # the shape of q, k, v from (batch_size, seq_len, num_heads, head_dim) to (batch_size, num_heads, seq_len, head_dim)
        q = q.transpose(1,2)
        k = k.transpose(1,2)
        v = v.transpose(1,2)

        # Step 1.3: calculate multi-head attention in parallel: Attention(q, k, v) = softmax(qk^T / sqrt(head_dim)) v
        # Step 1.3.1: do matrix multiplication via `torch.matmul()`: attn = qk^T / sqrt(head_dim)
        # the shape of `attn`: (batch_size, num_heads, seq_len, seq_len)
        attn = torch.matmul(q,k.transpose(-2,-1))/np.sqrt(self.head_dim)
        
        # Step 1.3.2: for Auto-Regressive Language Model, the predictions for position i can depend only on the input at positions less than i.
        # Therefore, a mask is used to prevent positions from attending to subsequent positions
        # attn_mask = (0,1,...,1; 0,0,1,...,0; ...; 0,...,0) is a upper triangular matrix with shape (seq_len, seq_len)
        # Hint:
        # use torch.ones(?, device=x.device) to generate a matrix filled with value 1 with shape (seq_len, seq_len)
        attn_mask = torch.ones(seq_len,seq_len,device=x.device)
        # use torch.triu(?, diagonal=1) to obtain a upper triangular matrix where elements on the main diagonal are 0
        attn_mask = attn_mask.triu(diagonal=1)
        # use Tensor.bool() to convert the matrix to a boolean matrix
        attn_mask = attn_mask.bool()
        # fill the position of `attn` where `attn_mask==True` with value `float('-inf')` via `Tensor.masked_fill()`
        attn = torch.masked_fill(attn,attn_mask,float('-inf'))

        # Step 1.3.3: normalize `attn` via softmax funtion: attn = Softmax(attn) = Softmax(qk^T / sqrt(head_dim))
        attn = torch.softmax(attn,dim=3)
        # Step 1.3.4: apply dropout to `attn` via self.attn_drop()
        attn = self.attn_drop(attn)
        # Step 1.3.5: multiply v by `attn` via torch.matmul(): out = Attention(q, k, v) = attn v
        # the shape of `out`: (batch_size, num_heads, seq_len, head_dim)
        out = torch.matmul(attn,v)

        # Step 1.4: use `Tensor.transpose()` and `Tensor.reshape()' to concatenate output of different heads
        # the shape of `out` from (batch_size, num_heads, seq_len, head_dim) to (batch_size, seq_len, num_heads*head_dim)
        out = out.transpose(1,2).reshape(batch_size,seq_len,self.num_head*self.head_dim)

        # Step 1.5: obtain the final results via self.proj_layer() and self.proj_drop(): result = Dropout(MultiHead(Q, K, V)) = Dropout(out W^O)
        result = self.proj_drop(self.proj_layer(out))
        # <<< TODO 1

        # return the final results `result` and attention weights `attn`
        return result, attn
    
############################################################
    
# Define the feed forward network (FFN)
############################################################
class FFN(nn.Module):
    def __init__(self, embed_dim, feedforward_dim, dropout):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, feedforward_dim)
        self.fc2 = nn.Linear(feedforward_dim, embed_dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = gelu(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x
############################################################

# Define the TransformerLayer
############################################################
class TransformerLayer(nn.Module):
    def __init__(self, embed_dim, num_head, feedforward_dim, dropout, no_res):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = SelfAttention(embed_dim, num_head, dropout)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.ffn = FFN(embed_dim, feedforward_dim, dropout)
        self.no_res = no_res # whether to use residual connection

    def forward(self, x):
        # >>> TODO 2: complete the forward process of the TransformerLayer module.
        # Step 2.1: calculate the output of multi-head self-attention
        # normalize the input via `self.norm1()`: x_norm = LayerNorm(x)
        x_norm = self.norm1(x)

        # calculate the output of multi-head self-attention via `self.attn()`: x_attn, attn = SelfAttention(x_norm)
        x_attn, attn = self.attn(x_norm)

        # add the input 'x' to the output of attention (x_attn) if self.no_res is False: x_attn = x + x_attn if no_res is False else x_attn
        if self.no_res == False:
            x_attn = x_attn + x

        # Step 2.2: calculate the output of feed forward network
        # calculate the output of feed forward network via `self.ffn()` and `self.norm2()`: x_ffn = FFN(LayerNorm(x_attn))
        x_ffn = self.ffn(self.norm2(x_attn))

        # add the output of attention (x_attn) to the output of feed forward network (x_ffn) if self.no_res is False: out = x_attn + x_ffn if no_res is False else x_ffn
        if self.no_res == False:
            out = x_ffn + x_attn
        else:
            out = x_ffn
        # <<< TODO 2
        
        return out, attn
############################################################

# Define the GPT module
############################################################
class GPT(nn.Module):
    def __init__(self, vocab_size, max_seq_len, num_layer, embed_dim, num_head, feedforward_dim, dropout, no_res=False, no_pos=False):
        '''
            vocab_size: the size of vocabulary
            max_seq_len: the maximum length of input texts
            num_layer: the number of transformer layers
            embed_dim: the embedding dimension
            num_head: the number of heads in Multi-Head Self Attention
            feedforward_dim: the dimension in the feed forward network
            dropout: dropout ratio
            no_res: whether to use residual connection in transformer layers
            no_pos: whether to use position embeddings
        '''
        super().__init__()
        self.num_layer = num_layer
        self.max_seq_len = max_seq_len
        self.no_pos = no_pos

        # Define Embedding Layer to transfer input text tokens and positions to embeddings
        self.word_token_embedding = nn.Embedding(vocab_size, embed_dim)
        self.word_pos_embedding = nn.Embedding(max_seq_len, embed_dim)

        self.drop = nn.Dropout(dropout)
        # Define the transformer layers
        self.transformer = nn.ModuleList([TransformerLayer(embed_dim, num_head, feedforward_dim, dropout, no_res) for _ in range(num_layer)])
        
        # Define the head layer to predict output
        self.norm = nn.LayerNorm(embed_dim)
        self.language_model_head = nn.Linear(embed_dim, vocab_size, bias=False)

        """
        Weight tying improves the performance of language models by tying (sharing) the weights of the embedding and softmax layers.
        Reference: https://paperswithcode.com/method/weight-tying
        """
        self.word_token_embedding.weight = self.language_model_head.weight

        self.init_weights()

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                torch.nn.init.normal_(m.weight, mean=0.0, std=0.02)
        
        # apply special scaled init to the residual projections, per GPT-2 paper
        for pn, p in self.named_parameters():
            if pn.endswith('proj_layer.weight'):
                torch.nn.init.normal_(p, mean=0.0, std=0.02/math.sqrt(2 * self.num_layer))

    
    def forward(self, word_idx, targets=None):
        batch_size, seq_len = word_idx.shape

        # >>> TODO 3: complete the forward process of GPT
        # Step 3.1: use torch.arange(?, dtype=torch.long, device=word_idx.device) to generate the position sequence `pos` [0, 1, ..., seq_len-1] 
        pos = torch.arange(0,seq_len,dtype=torch.long,device=word_idx.device)

        # Step 3.2: use self.word_token_embedding() and self.word_pos_embedding() to transfer `word_idx` and `pos` to embeddings ('token_embed` and `pos_embed`)
        token_embed = self.word_token_embedding(word_idx)
        pos_embed = self.word_pos_embedding(pos)

        # Step 3.3: initialize the input embeddings `x` of transformer layers
        # add the token embeddings and position embeddings to obtain the input embeddings `x` if self.no_pos is False
        if self.no_pos == False:
            x = token_embed  + pos_embed
        else:
            x = token_embed

        # apply dropout to the input embeddings via `self.drop()`
        x = self.drop(x)
        # print(x)

        # Step 3.4: use for loop to obtain the output and attention weights of multiple transformer layers
        # define a list `attention_weights` and append the attention weights of each transformer layer into the list
        attention_weights = [] 
        for layer in self.transformer:
            # Step 4.1: obtain the output and attention weights of transformer layers
            x, attn = layer(x)
            # Step 4.2: append the attention weights of transformer layers into the list `attention_weights`
            attention_weights.append(attn)
     
        # Step 3.5: use self.norm() to normalize the output of transformer layers and then use self.language_model_head() to obtain the `logits` for prediction
        # self.language_model_head() is a linear layer defined in __init__() function
        # Note: do not add softmax here since it is included in the cross entropy loss function
        x = self.norm(x)
        logits = self.language_model_head(x)
        # <<< TODO 3python prepare.py --data_root data/quansongci

        # return logits and loss or attention weights
        if targets is not None:
            loss = torch.nn.functional.cross_entropy(logits.reshape(-1, logits.size(-1)), targets.reshape(-1), ignore_index=0)
            return logits, loss
        assert isinstance(attention_weights, list), "attention_weights must be a list, please check whether to append the attention weights of all transformer layers into it!"
        return logits, attention_weights

    def configure_optimizers(self, weight_decay):
        """
        This long function is unfortunately doing something very simple and is being very defensive:
        We are separating out all parameters of the model into two buckets: those that will experience
        weight decay for regularization and those that won't (biases, and layernorm/embedding weights).
        We are then returning the PyTorch optimizer object.
        """

        # separate out all parameters to those that will and won't experience regularizing weight decay
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (nn.Linear, )
        blacklist_weight_modules = (nn.LayerNorm, torch.nn.Embedding)
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn # full param name
                # random note: because named_modules and named_parameters are recursive
                # we will see the same tensors p many many times. but doing it this way
                # allows us to know which parent module any tensor p belongs to...
                if pn.endswith('bias'):
                    # all biases will not be decayed
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    # weights of whitelist modules will be weight decayed
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    # weights of blacklist modules will NOT be weight decayed
                    no_decay.add(fpn)

        # subtle: 'transformer.wte.weight' and 'lm_head.weight' are tied, so they
        # will appear in the no_decay and decay sets respectively after the above.
        # In addition, because named_parameters() doesn't return duplicates, it
        # will only return the first occurence, key'd by 'transformer.wte.weight', below.
        # so let's manually remove 'lm_head.weight' from decay set. This will include
        # this tensor into optimization via transformer.wte.weight only, and not decayed.
        decay.remove('language_model_head.weight')

        # validate that we considered every parameter
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        assert len(inter_params) == 0, "parameters %s made it into both decay/no_decay sets!" % (str(inter_params), )
        assert len(param_dict.keys() - union_params) == 0, "parameters %s were not separated into either decay/no_decay set!" \
                                                    % (str(param_dict.keys() - union_params), )

        # create the pytorch optimizer object
        optim_groups = [
            {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
            {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
        ]
        return optim_groups
    
    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=1.0, top_k=None):
        """
        Take a conditioning sequence of indices idx (LongTensor of shape (b,t)) and complete
        the sequence max_new_tokens times, feeding the predictions back into the model each time.
        Most likely you'll want to make sure to be in model.eval() mode of operation for this.
        """
        for _ in range(max_new_tokens):
            # if the sequence context is growing too long we must crop it at block_size
            idx_cond = idx
            # forward the model to get the logits for the index in the sequence
            logits, _ = self(idx_cond)
            # pluck the logits at the final step and scale by desired temperature
            logits = logits[:, -1, :] / temperature
            # optionally crop the logits to only the top k options
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = -float('Inf')
            # apply softmax to convert logits to (normalized) probabilities
            probs = F.softmax(logits, dim=-1)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1)
            # append sampled index to the running sequence and continue
            idx = torch.cat((idx, idx_next), dim=1)

        return idx.squeeze().cpu().numpy()
############################################################

GPTConfig = {
    'mygpt': dict(num_layer=4, embed_dim=128, num_head=4, feedforward_dim=128*4, dropout=0.0),
    'gpt2-mini': dict(num_layer=6, embed_dim=384, num_head=6, feedforward_dim=384*4, dropout=0.2),
    'gpt2': dict(num_layer=12, embed_dim=768, num_head=12, feedforward_dim=768*4, dropout=0.2),
}