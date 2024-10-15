import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import pickle
import numpy as np

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print("Device:", device)

block_size = 128
batch_size = 64
max_iters = 1000
learning_rate = 3e-4
eval_iters = 100
n_embd = 384
n_layer = 8
n_head = 8
dropout = 0.2


df = pd.read_csv("data/raw/SMSSpamCollection.tsv", sep="\t", header=None, names=["label", "text"])
print(df.info())
print(df["label"].value_counts())


def create_balanced_dataset(df):
    num_spam = df[df["label"] == "spam"].shape[0]
    # print(num_spam)
    ham_subset = df[df["label"] == "ham"].sample(num_spam, random_state=42)
    balanced_df = pd.concat([ham_subset, df[df["label"] == "spam"]])
    return balanced_df


balanced_df = create_balanced_dataset(df)
print(balanced_df["label"].value_counts())


balanced_df["label"] = balanced_df["label"].map({"ham": 0, "spam": 1})
balanced_df.head()


def random_split(df, split=0.8):
    df = df.sample(frac=1, random_state=123).reset_index(drop=True)
    df_train = df[:int(split*len(df))]
    df_val = df[int(split*len(df)):]
    return df_train, df_val


train_df, validation_df = random_split(balanced_df)
print(train_df.head())
print(validation_df.head())
train_df.to_csv("data/processed/fine_tune_train.csv", index=None)
validation_df.to_csv("data/processed/fine_tune_validation.csv", index=None)


chars = ""

with open('data/raw/OpenWebText/vocab.txt', 'r', encoding='utf-8') as f:
    text = f.read()
    chars = sorted(list(set(text)))

vocab_size = len(chars)


string_to_int = { ch:i for i, ch in enumerate(chars) }
int_to_string = { i:ch for i, ch in enumerate(chars) }

encode = lambda s: [string_to_int[c] for c in s]
decode = lambda l: ''.join([int_to_string[i] for i in l])

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
        k = self.key(x)   # (B, T, hs)
        q = self.query(x) # (B, T, hs)
        wei = q @ k.transpose(-2, -1) * k.shape[-1]**-0.5 # (B, T, T)
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf')) # (B, T, T)
        wei = F.softmax(wei, dim=-1) # (B, T, T)
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
        self.classification_head = nn.Linear(n_embd, 1)  # New classification head
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
        elif isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)

    def forward(self, index, targets=None, classify=False):
        B, T = index.shape

        tok_emb = self.token_embedding_table(index)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        
        if classify:
            # For classification, we'll use the mean of all token representations
            x_mean = x.mean(dim=1)
            logits = self.classification_head(x_mean).squeeze(-1)
            if targets is not None:
                loss = F.binary_cross_entropy_with_logits(logits, targets.float())
            else:
                loss = None
        else:
            logits = self.lm_head(x)
            if targets is not None:
                B, T, C = logits.shape
                logits = logits.view(B*T, C)
                targets = targets.view(B*T)
                loss = F.cross_entropy(logits, targets)
            else:
                loss = None

        return logits, loss

    def classify(self, index):
        # Method for getting classification probabilities
        logits, _ = self.forward(index, classify=True)
        return torch.sigmoid(logits)
    
    def generate(self, index, max_new_tokens):
        # index is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            # crop idx to the last block_size tokens
            index_cond = index[:, -block_size:]
            # get the predictions
            logits, loss = self.forward(index_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            index_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            index = torch.cat((index, index_next), dim=1) # (B, T+1)
        return index

model = GPTLanguageModel(vocab_size)
print('loading model parameters...')
with open('data/models/model-04-reddit_text.pkl', 'rb') as f:
    model = pickle.load(f)
print('loaded successfully!')
model.classification_head = nn.Linear(n_embd, 1)

model = model.to(device)
# print(model)


from torch.utils.data import Dataset, DataLoader

class SMSDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_len):
        self.data = dataframe
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        text = self.data.iloc[index]['text']
        label = self.data.iloc[index]['label']
        
        encoded = self.tokenizer(text)
        if len(encoded) > self.max_len:
            encoded = encoded[:self.max_len]
        else:
            encoded = encoded + [0] * (self.max_len - len(encoded))  # Padding
        
        return {
            'input_ids': torch.tensor(encoded, dtype=torch.long),
            'label': torch.tensor(label, dtype=torch.float)
        }

# Assuming you have train_df and validation_df from your previous data preparation
train_dataset = SMSDataset(train_df, encode, block_size)
val_dataset = SMSDataset(validation_df, encode, block_size)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size)


optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)


def train_epoch(model, dataloader, optimizer, device):
    model.train()
    total_loss = 0
    for batch in dataloader:
        input_ids = batch['input_ids'].to(device)
        labels = batch['label'].to(device)
        
        optimizer.zero_grad()
        logits, loss = model(input_ids, labels, classify=True)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    return total_loss / len(dataloader)


def evaluate(model, dataloader, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in dataloader:
            input_ids = batch['input_ids'].to(device)
            labels = batch['label'].to(device)
            
            logits, loss = model(input_ids, labels, classify=True)
            preds = torch.sigmoid(logits) > 0.5
            
            total_loss += loss.item()
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    accuracy = (np.array(all_preds) == np.array(all_labels)).mean()
    return total_loss / len(dataloader), accuracy


num_epochs = 5
best_val_loss = float('inf')

for epoch in range(num_epochs):
    train_loss = train_epoch(model, train_loader, optimizer, device)
    val_loss, val_accuracy = evaluate(model, val_loader, device)
    
    print(f"Epoch {epoch+1}/{num_epochs}")
    print(f"Train Loss: {train_loss:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Accuracy: {val_accuracy:.4f}")
    
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        # torch.save(model.state_dict(), 'best_model_spam.pth')
        with open('data/models/model-05-spam_finetuned.pkl', 'wb') as f:
            pickle.dump(model, f)
        print('model saved')
    print()


