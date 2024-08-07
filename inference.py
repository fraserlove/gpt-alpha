"""
Generate from a trained GPT model.

Tiktoken is the default tokeniser, however, a custom tokeniser can be used by replacing
tiktoken.get_encoding('gpt2') with GPTTokeniser('gpt.tkn'). The tokeniser must be
the same as the one used during training.
"""

import tiktoken
import torch

from gpt.gpt import GPT

# ------------------------------- Generation parameters --------------------------------------
# The model configuration is loaded from the checkpoint file in 'cache/logs' with the .pt
# extension. The number of tokens to generate per sample, the number of samples to generate
# per process, the temperature for sampling, and the top-k sampling parameter are set below.
# --------------------------------------------------------------------------------------------
max_tokens = 64
n_samples = 2
temp = 1.0
top_k = 50

# Use GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f'{device}')

# Load the GPT-2 tokeniser
tokeniser = tiktoken.get_encoding('gpt2') # or GPTTokeniser('gpt.tkn')

# Load the checkpoint
ckpt = torch.load('cache/124M.pt', map_location=device)
# Load the model and move it to the device
model = GPT(ckpt['config']).to(device)
# Load the saved model state
model.load_state_dict(ckpt['model'])
model.eval()
print(f'checkpoint loaded')

# Generate samples from the model
context = 'Once upon a time,'
context = torch.tensor(tokeniser.encode(context), dtype=torch.long).to(device)
samples = model.generate(context, n_samples=n_samples, max_tokens=max_tokens, temp=temp, top_k=top_k)
samples = [samples[j, :].tolist() for j in range(n_samples)]
print('\n'.join(tokeniser.decode(sample).split('<|endoftext|>')[0] for sample in samples))