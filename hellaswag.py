"""
Downloads and evaluates HellaSwag, a commonsense reasoning dataset.

Dataset consists of a context and four possible completions, one of which is correct.

Run this script to download the HellaSwag dataset and evaluate a HuggingFace GPT-2 model
on the validation set. The model is evaluated by predicting the most likely completion given
the context.
"""

import os
import tqdm
import json
import argparse
import tiktoken
import requests
import torch
import torch.nn.functional as F

# Create the cache directory for hellaswag if it doesn't exist
CACHE_DIR = os.path.join(os.path.dirname(__file__), 'cache/hellaswag')
os.makedirs(CACHE_DIR, exist_ok=True)

tokeniser = tiktoken.get_encoding('gpt2') # or GPTTokeniser('gpt.tkn')

def download_file(url: str, file_name: str, chunk_size: int  = 1024) -> None:
    """Download a file from a given url."""
    resp = requests.get(url, stream=True)
    file_size = int(resp.headers.get('content-length', 0))
    with open(file_name, 'wb') as file:
        prog_bar = tqdm.tqdm(total=file_size, unit='iB', desc=file_name, unit_scale=True, unit_divisor=1024)
        for data in resp.iter_content(chunk_size=chunk_size):
            size = file.write(data)
            prog_bar.update(size)

def download(split: str) -> None:
    """Download HellaSwag."""
    data_url = f'https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_{split}.jsonl'
    data_filename = os.path.join(CACHE_DIR, f'hellaswag_{split}.jsonl')
    if not os.path.exists(data_filename):
        download_file(data_url, data_filename)

def prepare_example(example: dict) -> tuple[dict, torch.Tensor, torch.Tensor, int]:
    """
    Given an example, return the data in a format suitable for training.
    - tokens (context + completion), (4, N) where N is the length of the longest row, 4 possible completions
    - mask (1 in the region of the candidate completion, 0 elsewhere)
    - label (the index of the correct completion)
    """
    ctx = example['ctx']
    label = example['label']
    endings = example['endings']

    data = {
        'label': label,
        'ctx_tokens': None,
        'ending_tokens': [],
    }

    # Encode the context and each ending
    ctx_tokens = tokeniser.encode(ctx)
    data['ctx_tokens'] = ctx_tokens
    tok_rows = []
    mask_rows = []
    for end in endings:
        end_tokens = tokeniser.encode(' ' + end) # Prepend a space to match the tokeniser
        tok_rows.append(ctx_tokens + end_tokens)
        mask_rows.append([0] * len(ctx_tokens) + [1] * len(end_tokens))
        data['ending_tokens'].append(end_tokens)

    # Pad to the same length
    max_len = max(len(row) for row in tok_rows)
    tokens = torch.zeros((4, max_len), dtype=torch.long)
    mask = torch.zeros((4, max_len), dtype=torch.long)
    for k, (tok_row, mask_row) in enumerate(zip(tok_rows, mask_rows)):
        tokens[k, :len(tok_row)] = torch.tensor(tok_row)
        mask[k, :len(mask_row)] = torch.tensor(mask_row)

    return data, tokens, mask, label

def most_likely_row(tokens: torch.Tensor, mask: torch.Tensor, logits: torch.Tensor) -> int: 
    """Predict the most likely row given the tokens, mask and logits."""

    # Evaluate the autoregressive loss at all positions
    shift_logits = logits[..., :-1, :].contiguous()
    shift_tokens = tokens[..., 1:].contiguous()
    flat_shift_logits = shift_logits.view(-1, shift_logits.size(-1))
    flat_shift_tokens = shift_tokens.view(-1)
    shift_losses = F.cross_entropy(flat_shift_logits, flat_shift_tokens, reduction='none')
    shift_losses = shift_losses.view(tokens.size(0), -1)

    # Calculate the average loss for the completion region (mask == 1) for each row
    shift_mask = (mask[..., 1:]).contiguous() # Skip the context
    masked_shift_losses = shift_losses * shift_mask

    # Sum and divide by the number of 1s in the mask to get the average loss
    sum_loss = masked_shift_losses.sum(dim=1)
    avg_loss = sum_loss / shift_mask.sum(dim=1)

    # Completion with the minimum loss is the prediction
    pred = avg_loss.argmin().item()
    return pred

def iterate_examples(split):
    """Iterate over the HellaSwag examples."""
    download(split)
    with open(os.path.join(CACHE_DIR, f'hellaswag_{split}.jsonl'), 'r') as f:
        for line in f:
            example = json.loads(line)
            yield example

@torch.no_grad()
def evaluate(model_type, device):
    """Evaluate a HuggingFace model on the HellaSwag validation set."""

    # Load the model
    model = GPT2LMHeadModel.from_pretrained(model_type)
    model.to(device)
    # Compile the model for faster execution
    # model = torch.compile(model)

    n_total = 0
    n_correct = 0
    examples = iterate_examples('val')
    for example in examples:
        _, tokens, mask, label = prepare_example(example)
        tokens, mask = tokens.to(device), mask.to(device)
        logits = model(tokens).logits
        # Predict the most likely row
        pred = most_likely_row(tokens, mask, logits)
        # Update accuracy
        n_total += 1
        n_correct += int(pred == label)

        # Debug: Print the context, completions and the predicted and actual labels
        if n_total % 1000 == 0:
            print(f'--- Example {n_total} ---')
            context = example['ctx']
            print(f'Context:\n {context}')
            print(f'Endings:')
            for i, end in enumerate(example['endings']):
                print(f' ({i}) {end}' + (' (P)' if i == label else '') + (' (A)' if i == pred else ''))

    print(f'{model_type} acc: {(n_correct / n_total):.4f}')

if __name__ == '__main__':
    # For HuggingFace GPT-2 model evaluation - pip install transformers
    from transformers import GPT2LMHeadModel

    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model_type', type=str, default='gpt2', help='HuggingFace GPT-2 model')
    args = parser.parse_args()

    torch.set_float32_matmul_precision('high') # Use tensor cores for matmul
    device = f'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    evaluate(args.model_type, device)