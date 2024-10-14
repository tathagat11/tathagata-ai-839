import os
from tqdm.notebook import tqdm
from datasets import load_dataset
import concurrent.futures

def process_text(text):
    return set(text), text + "\n"

def process_chunk(chunk):
    chunk_vocab = set()
    chunk_texts = []
    for text in chunk['text']:
        chars, processed_text = process_text(text)
        chunk_vocab.update(chars)
        chunk_texts.append(processed_text)
    return chunk_vocab, chunk_texts

def process_dataset_in_chunks(dataset, output_file, chunk_size=1000, num_workers=4):
    vocab = set()
    total_chunks = len(dataset) // chunk_size + (1 if len(dataset) % chunk_size != 0 else 0)
    
    with open(output_file, 'w', encoding='utf-8') as outfile:
        with concurrent.futures.ProcessPoolExecutor(max_workers=num_workers) as executor:
            for i in tqdm(range(0, len(dataset), chunk_size), total=total_chunks, desc="Processing chunks"):
                chunk = dataset.select(range(i, min(i + chunk_size, len(dataset))))
                chunk_vocab, chunk_texts = process_chunk(chunk)
                vocab.update(chunk_vocab)
                outfile.writelines(chunk_texts)
    
    return vocab

# Load the dataset
dataset = load_dataset("openwebtext", trust_remote_code=True)

# Calculate split sizes
total_samples = len(dataset['train'])
split_index = int(total_samples * 0.9)  # 90% for training

# Create train and validation splits
train_dataset = dataset['train'].select(range(split_index))
val_dataset = dataset['train'].select(range(split_index, total_samples))

output_file_train = "data/raw/OpenWebText/train.txt"
output_file_val = "data/raw/OpenWebText/val.txt"
vocab_file = "data/raw/OpenWebText/vocab.txt"

print("Processing training data...")
vocab_train = process_dataset_in_chunks(train_dataset, output_file_train, chunk_size=1000, num_workers=os.cpu_count())

print("Processing validation data...")
vocab_val = process_dataset_in_chunks(val_dataset, output_file_val, chunk_size=1000, num_workers=os.cpu_count())

# Combine vocabularies and write to vocab.txt
vocab = vocab_train.union(vocab_val)
with open(vocab_file, "w", encoding="utf-8") as vfile:
    for char in sorted(vocab):
        vfile.write(char + '\n')

print("Processing complete. Files created.")