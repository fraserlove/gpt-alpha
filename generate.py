"""
Generate from a trained GPT model.

Generation can be run both on a single gpu and also over multiple GPUs with
distributed data parallel (DDP).

Tiktoken is the default tokeniser, however, a custom tokeniser can be used by replacing
tiktoken.get_encoding('gpt2') with GPTTokeniser('gpt.tkn'). The tokeniser must be
the same as the one used during training.

To run on a single GPU, use:
$ python generate.py

To run with DDP on multiple GPUs on a single node, use:
$ torchrun --standalone --nproc_per_node={n_gpus} generate.py
"""

import os
import tiktoken
import torch
import torch.distributed as dist

from gpt.gpt import GPT

# ----------------------- Generation parameters -------------------------
# The model configuration is loaded from the checkpoint file in 'cache/logs'
# with the .pt extension. The number of tokens to generate per sample and
# the number of samples to generate per process can be set below.
# -----------------------------------------------------------------------
max_tokens = 64
n_samples = 2
ckpt_file = '124M.pt'
log_dir = 'cache/logs'

# ------------------------- Distributed Data Parallel (DDP) ----------------------------------
# If the RANK environment variable is set, via 'torchrun', then DDP is used for multi-GPU
# training. RANK is the global rank across all nodes, LOCAL_RANK is the local rank within
# the node, and WORLD_SIZE is the total number of GPUs. The master process is the process
# with rank 0, which outputs the training logs and validation metrics. DDP is initialised
# with the NCCL backend for communication between GPUs.
# --------------------------------------------------------------------------------------------
ddp = int(os.environ.get('RANK', -1)) != -1
ddp_rank = int(os.environ.get('RANK', 0))
ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
master_process = ddp_rank == 0 # Output from the master process only
dist.init_process_group(backend='nccl') if ddp else None

device = f'cuda:{ddp_local_rank}' if ddp else 'cuda' if torch.cuda.is_available() else 'cpu'
device_name = f'({torch.cuda.get_device_name(ddp_local_rank)})' if 'cuda' in device else ''
torch.cuda.set_device(device) if ddp else None
print(f'{device} {device_name}')

# Load the GPT-2 tokeniser
tokeniser = tiktoken.get_encoding('gpt2') # or GPTTokeniser('gpt.tkn')

# Load the checkpoint
ckpt_path = os.path.join(log_dir, ckpt_file)
ckpt = torch.load(ckpt_path, map_location=device)
model = GPT(ckpt['config'])
model.load_state_dict(ckpt['model'])
model.to(device)
model = torch.compile(model)
print(f'loaded checkpoint {ckpt_file} with val loss {ckpt["val_loss"]:.2f}')

# Generate samples from the model
context = 'Once upon a time,'
context = torch.tensor(tokeniser.encode(context), dtype=torch.long).to(device)
samples = model.generate(context, n_samples=n_samples, max_tokens=max_tokens)
samples = [samples[j, :max_tokens].tolist() for j in range(n_samples)]
print('\n'.join(tokeniser.decode(sample) for sample in samples))