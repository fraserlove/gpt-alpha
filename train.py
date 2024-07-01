"""
Training script for GPT model on the fineweb_edu_10B dataset.

This training script can be run both on a single gpu and also in a larger multi-GPU
training run with distributed data parallel (DDP).

Tiktoken is the default tokeniser, however, a custom tokeniser can be used by replacing
tiktoken.get_encoding('gpt2') with GPTTokeniser('gpt.tkn'). The fineweb-edu dataset
must be tokenised with the same tokeniser used during training.

To run on a single GPU, use:
$ python train.py

To run with DDP on multiple GPUs on 1 node, use:
$ torchrun --standalone --nproc_per_node={n_gpus} train.py
"""

import os
import math
import time
import tiktoken
import torch
import torch.nn as nn
import torch.distributed as dist

import hellaswag
from gpt.gpt import GPT, GPTConfig
from gpt.dataloader import GPTDataLoader

torch.manual_seed(0)
# Improved performance by using TensorFloat32 or two bfloat16 for matmuls
torch.set_float32_matmul_precision('high')

# Load the GPT-2 tokeniser
tokeniser = tiktoken.get_encoding('gpt2') # or GPTTokeniser('gpt.tkn')

# ----------------------- Default training parameters for GPT-3 124M -------------------------
# The total batch size is the number of tokens to process in a single iteration before a gradient
# update, for GPT-2 / GPT-3 this is 2^19 = ~0.5M tokens. A cosine learning rate schedule is used,
# with a warmup period of 375M tokens as in GPT-3. This corresponds to 375e6 / 2^19 = 715 warmup
# iterations. A max learning rate of 18e-4 (3x that of GPT-3) is used with a linear decay over
# the training period. The model is trained for 1 epoch on the fineweb_edu_10B dataset, which has
# 10B tokens. This corresponds to 10^10 / 2^19 = 19073 iterations. The vocabulary size is rounded
# up to the nearest multiple of 128, which is 50304, for efficiency.
# --------------------------------------------------------------------------------------------
batch_size = 64
total_batch_size = 524288
vocab_size = 50304
max_lr = 18e-4
warmup_iters = 715
max_iters = 19073

# -------------------- Logging, evaluation and checkpoint parameters -------------------------
# Evaluation is performed every 250 iterations on 20 batches of the validation set and the 
# HellaSwag dataset. The model generates 2 text samples of length 64 tokens per GPU every
# 250 iterations. Model checkpoints are saved every 5000 iterations to the 'cache/logs' directory
# alongside the training logs. The checkpoint file can be set to resume training from a previous
# checkpoint. This checkpoint must be present in the 'cache/logs' directory with the .pt extension.
# --------------------------------------------------------------------------------------------
eval_delta = 100
n_samples = 2
max_tokens = 64
val_iters = 20
log_dir = 'cache/logs'
ckpt_file = None

# ------------------------- Distributed Data Parallel (DDP) ----------------------------------
# If the RANK environment variable is set, via 'torchrun', then DDP is used for multi-GPU
# training. RANK is the global rank across all nodes, LOCAL_RANK is the local rank within
# the node, and WORLD_SIZE is the total number of GPUs. The master process is the process
# with rank 0, which outputs the training logs and validation metrics. DDP is initialised
# with the NCCL backend for communication between GPUs.
# --------------------------------------------------------------------------------------------
assert torch.cuda.is_available(), 'CUDA required for training'
ddp = int(os.environ.get('RANK', -1)) != -1
ddp_rank = int(os.environ.get('RANK', 0))
ddp_local_rank = int(os.environ.get('LOCAL_RANK', 0))
ddp_world_size = int(os.environ.get('WORLD_SIZE', 1))
master_process = ddp_rank == 0 # Output from the master process only
dist.init_process_group(backend='nccl') if ddp else None

device = f'cuda:{ddp_local_rank}' if ddp else 'cuda'
device_name = f'({torch.cuda.get_device_name(ddp_local_rank)})'
torch.cuda.set_device(device) if ddp else None
print(f'{device} {device_name}')

def lr_schedule(i: int) -> float:
    """Cosine decay learning rate schedule with a linear warmup."""
    min_lr = max_lr * 0.1

    if i < warmup_iters: # Linear warmup for warmup_iters
        return max_lr * (i + 1) / warmup_iters

    if i > max_iters: # Minimum learning rate after max_iters
        return min_lr
    
    # Otherwise, cosine decay down to minimum learning rate
    decay_ratio = (i - warmup_iters) / (max_iters - warmup_iters)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

def abbr_size(i: int) -> str:
    """Abbreviate numbers for logging, showing the three most significant digits."""
    for unit in ['', 'K', 'M', 'B', 'T', 'P']:
        if i < 1000:
            return f'{i:.3g}{unit}'
        i /= 1000

def create_logs(log_dir: str, n_params: int) -> str:
    """Create a log directory for the model."""
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, f'{abbr_size(n_params)}.txt')
    return log_file

def checkpoint(log_dir: str, model: GPT, n_params: int, i: int, loss_acc: torch.Tensor) -> None:
    """Save the model checkpoint."""
    ckpt_path = os.path.join(log_dir, f'{abbr_size(n_params)}.pt')
    ckpt = {
        'model': model.state_dict(),
        'optimiser': optimiser.state_dict(),
        'config': model.config,
        'iter': i,
        'val_loss': loss_acc.item()
    }
    torch.save(ckpt, ckpt_path)

if ckpt_file:
    ckpt_path = os.path.join(log_dir, ckpt_file)
    ckpt = torch.load(ckpt_path, map_location=device)
    model = GPT(ckpt['config'])
    model.load_state_dict(ckpt['model'])
    i = ckpt['iter']
    val_loss = ckpt['val_loss']
    if master_process:
        print(f'loaded checkpoint {ckpt_file} at iteration {i}')
else:
    model = GPT(GPTConfig(vocab_size=vocab_size))
    i = 0

model.to(device)
# Compile the model for faster execution
# model = torch.compile(model)

# Wrap the model in DDP if using multiple GPUs
model = nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank]) if ddp else model
raw_model = model.module if ddp else model

# Configure the AdamW optimiser with weight decay, per GPT-3
optimiser = raw_model.configure_optimisers(weight_decay=0.1, lr=max_lr)
if ckpt_file:
    optimiser.load_state_dict(ckpt['optimiser'])
ckpt = None # Free up memory

n_params = sum(param.numel() for param in model.parameters())
if master_process:
    print(f'model size: {abbr_size(n_params)}')
# Create a log for the model
log_file = create_logs(log_dir, n_params)

# Gradient accumulation is used to increase the effective batch size for training. The gradient
# accumulation iterations is the number of itetations to process before a gradient update. The
# total batch size must be divisible by the product of the batch size, block size, and number of
# GPUs (ddp_world_size) for gradient accumulation to work correctly.
block_size = raw_model.config.block_size
assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0
acc_iters = total_batch_size // (batch_size * block_size * ddp_world_size)

# Load the fineweb_edu_10B dataset
train_loader = GPTDataLoader(B=batch_size, T=block_size, proc_rank=ddp_rank, n_proc=ddp_world_size, split='train')
val_loader = GPTDataLoader(B=batch_size, T=block_size, proc_rank=ddp_rank, n_proc=ddp_world_size, split='val')

best_val_loss = float('inf')

# Training loop
while i < max_iters:
    t0 = time.time()

    # Evaluate the model
    if i % eval_delta == 0:
        # Validation
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            loss_acc = 0.0
            for _ in range(val_iters):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss /= val_iters
                loss_acc += loss.detach()
            if ddp:
                dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

            # Log the validation loss
            if master_process:
                print(f'val {loss_acc.item():.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'{i:2d} ({(i + 1) * total_batch_size}) val {loss_acc.item():.4f}\n')   

        # Save the model checkpoint if the validation loss is the best so far
        if loss_acc < best_val_loss and master_process:
            best_val_loss = loss_acc
            checkpoint(log_dir, raw_model, n_params, i, loss_acc)
            if master_process:
                print('saved checkpoint')

        # HellaSwag
        n_correct = 0
        n_total = 0
        for j, example in enumerate(hellaswag.iterate_examples('val')):
            # Only process examples where j % ddp_world_size == ddp_rank for distributed computation
            if j % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = hellaswag.prepare_example(example)
            tokens, mask = tokens.to(device), mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                pred = hellaswag.most_likely_row(tokens, mask, logits)
            n_total += 1
            n_correct += int(pred == label)

        # Average the stats across all processes
        if ddp:
            stats = torch.tensor([n_correct, n_total], dtype=torch.long, device=device)
            dist.all_reduce(stats, op=dist.ReduceOp.SUM)
            n_correct, n_total = stats.tolist()
        acc = n_correct / n_total
        if master_process:
            print(f'hellaswag {acc:.4f}')
            with open(log_file, 'a') as f:
                f.write(f'{i:2d} ({(i + 1) * total_batch_size}) hellaswag {acc:.4f}\n')

        # Generate samples
        context = 'Once upon a time,'
        context = torch.tensor(tokeniser.encode(context), dtype=torch.long).to(device)
        # Generate samples from the model with a separate seed for each process
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            samples = raw_model.generate(context, n_samples=n_samples, seed=ddp_rank)

        # Decode the generated tokens
        for j in range(n_samples):
            tokens = samples[j, :max_tokens].tolist()
            decoded = tokeniser.decode(tokens)
            print(f'{decoded}\n')

    # Training
    model.train()
    optimiser.zero_grad()
    loss_acc = 0

    # Accumulate gradients
    for j in range(acc_iters):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (j == acc_iters - 1)
        # Mixed precision training with bfloat16 for faster training
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            _, loss = model(x, y)
        # Scale the loss to be the average over the accumulation iterations
        loss /= acc_iters
        loss_acc += loss.detach()
        loss.backward()
    # Average the loss across all processes
    if ddp:
        dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)

    # Clip the gradients to prevent exploding gradients, per GPT-3
    nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    # Update the learning rate with the cosine decay schedule
    lr = lr_schedule(i)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr
    # Accumulated gradient update
    optimiser.step()

    # Wait for all kernels to finish
    torch.cuda.synchronize()
    dt = time.time() - t0
    # Log the training loss
    if master_process:
        tokens_per_sec = (batch_size * block_size * acc_iters * ddp_world_size) / dt
        print(f'{i:2d} | toks: {abbr_size(i * total_batch_size)} | loss: {loss_acc.item():.4f} | lr: {lr:.4e} | dt: {dt*1000:.2f}ms | tok/s: {abbr_size(tokens_per_sec)}')
        with open(log_file, 'a') as f:
            f.write(f'{i:2d} ({(i + 1) * total_batch_size}) train {loss_acc.item():.4f}\n')
    i += 1

if ddp:
    dist.destroy_process_group()