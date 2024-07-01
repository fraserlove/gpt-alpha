"""
Downloads and tokenises the fineweb-edu dataset, saving the data shards to disk.
"""

import os
import tqdm
import tiktoken
import datasets
import multiprocessing
import numpy as np

local_dir = 'cache/fineweb_edu_10B' # Local directory to save the dataset
remote_name = 'sample-10BT' # Dataset name on Hugging Face
shard_size = int(1e8) # 100M tokens per shard, total of 100 shards

# Create the local directory if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(__file__), local_dir)
os.makedirs(CACHE_DIR, exist_ok=True)

tokeniser = tiktoken.get_encoding('gpt2') # or GPTTokeniser('gpt.tkn')

# Download the dataset from Hugging Face
dataset = datasets.load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train')

def tokenise(doc: dict) -> np.ndarray:
    tokens = [tokeniser._special_tokens['<|endoftext|>']]
    tokens.extend(tokeniser.encode_ordinary(doc['text']))
    tokens = np.array(tokens)
    assert (0 <= tokens).all() and (tokens < 2**16).all(), 'Token dictionary exceeds bounds for uint16'
    return tokens.astype(np.uint16)

# Tokenise all documents and write output shards, each of shard_size tokens (last shard has remainder)
nprocs = max(1, os.cpu_count() // 2) # Use half the CPU cores
with multiprocessing.Pool(nprocs) as pool:
    shard_idx = 0
    all_tokens = np.empty((shard_size,), dtype=np.uint16) # Pre-allocate the maximum shard size
    token_count = 0
    prog_bar = None
    for tokens in pool.imap(tokenise, dataset, chunksize=16):
        if token_count + len(tokens) < shard_size: # Shard not full yet
            # Append tokens to the current shard
            all_tokens[token_count : token_count + len(tokens)] = tokens
            token_count += len(tokens)
            # Update progress bar
            if prog_bar is None:
                prog_bar = tqdm.tqdm(total=shard_size, unit='tok', desc=f'Shard {shard_idx:04d}')
            prog_bar.update(len(tokens))
        else:
            # Write the current shard and reset for the next one
            split = 'val' if shard_idx == 0 else 'train'
            filename = os.path.join(CACHE_DIR, f'edufineweb_{split}_{shard_idx:04d}')
            # Pack document into this shard, remaining tokens go to next one
            remainder = shard_size - token_count
            prog_bar.update(remainder)
            all_tokens[token_count : token_count + remainder] = tokens[:remainder]
            np.save(filename, all_tokens)
            shard_idx += 1
            prog_bar = None
            # Start a new shard with the remaining tokens
            all_tokens[0 : len(tokens) - remainder] = tokens[remainder:]
            token_count = len(tokens) - remainder

    # Write remaining tokens to the last shard
    if token_count != 0:
        split = 'val' if shard_idx == 0 else 'train'
        filename = os.path.join(CACHE_DIR, f'edufineweb_{split}_{shard_idx:04d}')
        np.save(filename, all_tokens[:token_count])