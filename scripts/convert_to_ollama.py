import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import struct
import os

# Model parameters (same as your training)
block_size = 128
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2

# Define model classes (copy all your model classes here)
class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        k = self.key(x)
        q = self.query(x)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        wei = self.dropout(wei)
        v = self.value(x)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(head_size * num_heads, n_embd)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)
        out = self.dropout(self.proj(out))
        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = MultiHeadAttention(n_head, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        y = self.sa(x)
        x = self.ln1(x + y)
        y = self.ffwd(x)
        x = self.ln2(x + y)
        return x

class GPTLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, index, targets=None):
        B, T = index.shape
        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=index.device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)
        return logits

def convert_to_gguf(input_path, output_path):
    print("Loading model...")
    with open(input_path, 'rb') as f:
        model = pickle.load(f)
    
    print("Converting weights...")
    # Create header
    header = {
        "version": 1,
        "n_vocab": model.token_embedding_table.weight.shape[0],
        "n_embd": n_embd,
        "n_head": n_head,
        "n_layer": n_layer,
        "block_size": block_size,
    }
    
    # Prepare weights dictionary
    weights = {}
    
    # Add embeddings - now with detach()
    weights["token_embeddings.weight"] = model.token_embedding_table.weight.detach().cpu().numpy()
    weights["position_embeddings.weight"] = model.position_embedding_table.weight.detach().cpu().numpy()
    
    # Add transformer blocks - now with detach()
    for i, block in enumerate(model.blocks):
        prefix = f"layers.{i}."
        weights[prefix + "attention.wq.weight"] = block.sa.proj.weight.detach().cpu().numpy()
        weights[prefix + "attention.wk.weight"] = block.sa.proj.weight.detach().cpu().numpy()
        weights[prefix + "attention.wv.weight"] = block.sa.proj.weight.detach().cpu().numpy()
        weights[prefix + "attention.wo.weight"] = block.sa.proj.weight.detach().cpu().numpy()
        weights[prefix + "ffn.w1.weight"] = block.ffwd.net[0].weight.detach().cpu().numpy()
        weights[prefix + "ffn.w2.weight"] = block.ffwd.net[2].weight.detach().cpu().numpy()
        weights[prefix + "attention_norm.weight"] = block.ln1.weight.detach().cpu().numpy()
        weights[prefix + "ffn_norm.weight"] = block.ln2.weight.detach().cpu().numpy()
    
    # Add final norm and output - now with detach()
    weights["output_norm.weight"] = model.ln_f.weight.detach().cpu().numpy()
    weights["output.weight"] = model.lm_head.weight.detach().cpu().numpy()
    
    print(f"Writing GGUF file to {output_path}...")
    with open(output_path, 'wb') as f:
        # Write header bytes
        header_bytes = struct.pack('6I', 
            0x67677566,  # GGUF magic
            1,           # GGUF version
            header["n_vocab"],
            header["n_embd"],
            header["n_head"],
            header["n_layer"]
        )
        f.write(header_bytes)
        
        # Write weights
        for name, array in weights.items():
            # Write tensor name
            name_bytes = name.encode('utf-8')
            f.write(struct.pack('I', len(name_bytes)))
            f.write(name_bytes)
            
            # Write tensor shape
            f.write(struct.pack('I', len(array.shape)))
            for dim in array.shape:
                f.write(struct.pack('I', dim))
            
            # Write tensor data
            array.tofile(f)
    
    print("Conversion complete!")

if __name__ == "__main__":
    input_path = "data/models/model-04-reddit_text.pkl"
    output_path = "model-converted.gguf"
    convert_to_gguf(input_path, output_path)