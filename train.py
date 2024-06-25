import os
import math
import time
import tiktoken
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from datetime import datetime

import hellaswag
from gpt.gpt import GPT, GPTConfig
from gpt.dataloader import GPTDataLoader

# Model and training parameters
block_size = 2048 # Maximum context length
batch_size = 32 # Sequences to process in parallel
total_batch_size = 524288 # Sequences to process in a singular gradient update (2^19, ~0.5M tokens in total)
max_lr = 6e-4 # Maximum learning rate
min_lr = max_lr * 0.1 # Minimum learning rate
warmup_iters = 715 # Warmup iterations for linear learning rate schedule. Warm up over 375M tokens. 375e6 / 2^19 = 715
max_iters = 19073 # Iterations to train the model. Dataset size is 1B Tokens so 10^9 / 2^19 = 19073 is ~1 epoch

# Logging and evaluation parameters
val_delta = 100 # Run validation every val_delta iterations
hs_delta = 100 # Evaluate HellaSwag every hs_delta iterations
gen_delta = 250 # Generate text samples every gen_delta iterations
ckpt_delta = 5000 # Save model checkpoints every ckpt_delta iterations
gen_sequences = 2 # Number of sequences to generate per gpu
gen_length = 32 # Maximum length of generated sequences
val_iters = 20 # Number of validation iterations

torch.set_float32_matmul_precision('high') # Use tensor cores for matmul

torch.manual_seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed(0)

def get_lr(i: int) -> float:
    """Cosine learning rate schedule with linear warmup."""
    if i < warmup_iters: # Linear warmup for warmup_iters
        return max_lr * (i + 1) / warmup_iters
    if i > max_iters:
        return min_lr # Minimum learning rate after max_iters
    # Otherwise, use cosine decay down to min_lr
    decay_ratio = (i - warmup_iters) / (max_iters - warmup_iters)
    assert 0 <= decay_ratio <= 1
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)

# Set up Distributed Data Parallel (DDP) for multi-GPU training
# 'torchrun' sets up the environment variables for DDP (RANK - Global rank across
# all nodes, LOCAL_RANK - Local rank within the node, WORLD_SIZE - Number of GPUs)
# e.g. torchrun --standalone --nproc_per_node=8 train.py will train on 8 GPUs
# Otherwise, run the script with 'python train.py' for single-GPU training
ddp = int(os.environ.get('RANK', -1)) != -1
if ddp:
    assert torch.cuda.is_available(), 'DDP requires CUDA'
    dist.init_process_group(backend='nccl')
    ddp_rank = int(os.environ['RANK'])
    ddp_local_rank = int(os.environ['LOCAL_RANK'])
    ddp_world_size = int(os.environ['WORLD_SIZE'])
    device = f'cuda:{ddp_local_rank}'
    device_name = f'({torch.cuda.get_device_name(ddp_local_rank)})'
    torch.cuda.set_device(device)
    master_process = ddp_rank == 0 # Output only from the master process
else:
    ddp_rank = 0
    ddp_local_rank = 0
    ddp_world_size = 1
    master_process = True
    device = f'cuda' if torch.cuda.is_available() else 'cpu' # Use GPU if available
    device_name = f'({torch.cuda.get_device_name(ddp_local_rank)})' if device == 'cuda' else ''

print(f'Device: {device} {device_name}')

assert total_batch_size % (batch_size * block_size * ddp_world_size) == 0, \
    'Total batch size must be divisible by batch size * block size * number of GPUs'
# Number of gradient accumulation iterations
acc_iters = total_batch_size // (batch_size * block_size * ddp_world_size)

train_loader = GPTDataLoader(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='train')
val_loader = GPTDataLoader(B=batch_size, T=block_size, process_rank=ddp_rank, num_processes=ddp_world_size, split='val')

# Set vocab_size = k2^n for optimal performance
model = GPT(GPTConfig(vocab_size=50304))
model.to(device)
model = torch.compile(model) # Compile the model for faster execution
if ddp:
    model = nn.parallel.DistributedDataParallel(model, device_ids=[ddp_local_rank])
raw_model = model.module if ddp else model

if master_process:
    total_params = sum(param.numel() for param in model.parameters())
    print(f'Model parameters: {total_params}')

# Training the model
optimiser = raw_model.configure_optimisers(weight_decay=0.1, lr=max_lr, device=device)

# Create a log directory for the model
log_dir = 'cache/logs'
os.makedirs(log_dir, exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
log_file = os.path.join(log_dir, f'{timestamp}.txt')
with open(log_file, 'w') as f: # Clear the log file
    pass

for i in range(max_iters):
    t0 = time.time()

    # Validation
    if i % val_delta == 0 or i == max_iters - 1:
        model.eval()
        val_loader.reset()
        with torch.no_grad():
            val_loss_acc = 0
            for j in range(val_iters):
                x, y = val_loader.next_batch()
                x, y = x.to(device), y.to(device)
                with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                    _, loss = model(x, y)
                loss = loss / val_iters
                val_loss_acc += loss.detach()
            if ddp:
                dist.all_reduce(val_loss_acc, op=dist.ReduceOp.AVG)

            # Log the validation loss
            if master_process:
                print(f'val loss: {val_loss_acc.item():.4f}')
                with open(log_file, 'a') as f:
                    f.write(f'val loss: {val_loss_acc.item():.4f}\n')

                # Write model checkpoints every ckpt_delta iterations
                if i > 0 and (i % ckpt_delta == 0 or i == max_iters - 1):
                    ckpt_path = os.path.join(log_dir, f'model_{i:05d}.pt')
                    ckpt = {
                        'model': raw_model.state_dict(),
                        'config': raw_model.config,
                        'step': i,
                        'val_loss': val_loss_acc.item()
                    }
                    torch.save(ckpt, ckpt_path)

    # HellaSwag evaluation
    if i % hs_delta == 0 or i == max_iters - 1:
        n_correct = 0
        n_total = 0
        for i, example in enumerate(hellaswag.iterate_examples('val')):
            # Only process examples where i % ddp_world_size == ddp_rank for distributed computation
            if i % ddp_world_size != ddp_rank:
                continue
            _, tokens, mask, label = hellaswag.prepare_example(example)
            tokens, mask = tokens.to(device), mask.to(device)

            with torch.no_grad():
                with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                    logits, _ = model(tokens)
                pred = hellaswag.most_likely_row(tokens, mask, logits)
            n_total += 1
            n_correct += int(pred == label)

        # Reduce the stats across all processes
        if ddp:
            n_total = torch.tensor(n_total, dtype=torch.long, device=device)
            n_correct = torch.tensor(n_correct, dtype=torch.long, device=device)
            dist.all_reduce(n_total, op=dist.ReduceOp.SUM)
            dist.all_reduce(n_correct, op=dist.ReduceOp.SUM)
            n_total = n_total.item()
            n_correct = n_correct.item()
        acc = n_correct / n_total
        if master_process:
            print(f'hellaswag acc: {acc:.4f}')
            with open(log_file, 'a') as f:
                f.write(f'{i} hellaswag acc: {acc:.4f}\n')

    # Generate samples
    if i % gen_delta == 0 or i == max_iters - 1:
        model.eval()
        tokeniser = tiktoken.get_encoding('gpt2')
        # Alternatively, use the custom tokeniser from gpt.tokeniser
        # tokeniser = GPTTokeniser('gpt.tkn')
        tokens = torch.tensor(tokeniser.encode('Hello,'), dtype=torch.long)
        x = tokens.unsqueeze(0).repeat(gen_sequences, 1).to(device)
        # Separate random number generator for sampling
        rng = torch.Generator(device=device).manual_seed(0 + ddp_rank)

        # Generate text until maximum length is reached
        while x.size(1) < gen_length:
            with torch.no_grad():
                with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
                    logits, _ = model(x)
                logits = logits[:, -1, :] # Last token logits
                probs = F.softmax(logits, dim=1) # Softmax for probabilities

                # Top-k sampling where k=50
                topk_probs, topk_indices = torch.topk(probs, 50, dim=-1)
                # Sample from the top-k probabilities
                ix = torch.multinomial(topk_probs, 1, generator=rng)
                # Gather sampled token indices
                xcol = torch.gather(topk_indices, -1, ix)
                # Concatenate sampled token to the sequence
                x = torch.cat((x, xcol), dim=1)

        # Decode the generated tokens
        for j in range(gen_sequences):
            tokens = x[j, :gen_length].tolist()
            decoded = tokeniser.decode(tokens)
            print(f'{i:2d} sample: {decoded}')
            with open(log_file, 'a') as f:
                    f.write(f'{i:2d} sample: {decoded}\n')

    # Training
    model.train()
    optimiser.zero_grad()
    loss_acc = 0

    for j in range(acc_iters):
        x, y = train_loader.next_batch()
        x, y = x.to(device), y.to(device)
        if ddp:
            model.require_backward_grad_sync = (j == acc_iters - 1)
        with torch.autocast(device_type=device.split(':')[0], dtype=torch.bfloat16):
            _, loss = model(x, y)
        # Scale the loss to be the average loss over the accumulation iterations to account
        # for gradient accumulation as gradients are summed on each backwards pass
        loss = loss / acc_iters
        loss_acc += loss.detach()
        loss.backward()

    if ddp:
        dist.all_reduce(loss_acc, op=dist.ReduceOp.AVG)
    nn.utils.clip_grad_norm_(model.parameters(), 1.0) # Clip gradients
    lr = get_lr(i)
    for param_group in optimiser.param_groups:
        param_group['lr'] = lr
    optimiser.step()
    if 'cuda' in device:
        torch.cuda.synchronize() # Wait for the GPU to finish
    dt = time.time() - t0
    tokens_per_sec = (batch_size * block_size * acc_iters * ddp_world_size) / dt

    # Log the training loss
    if master_process:
        print(f'{i:2d} | loss: {loss_acc.item():.4f} | lr: {lr:.4e} | dt: {dt*1000:.2f}ms | tok/s: {tokens_per_sec:.0f}')
        with open(log_file, 'a') as f:
            f.write(f'{i} train {loss_acc.item():.4f}\n')

if ddp:
    dist.destroy_process_group()