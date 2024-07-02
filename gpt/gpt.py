"""
Full implementation of a Generative Pre-trained Transformer (GPT) model.

References
1) GPT-2 Paper:
https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf
2) GPT-3 Paper:
https://arxiv.org/abs/2005.14165
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass

@dataclass
class GPTConfig:
    block_size: int = 2048 # Maximum context length
    vocab_size: int = 50257 # Number of unique tokens
    n_layer: int = 12 # Number of transformer blocks
    n_head: int = 12 # Number of self-attention heads
    n_embd: int = 768 # Embedding dimensionality

class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        assert config.n_embd % config.n_head == 0, 'Embedding dimensionality must be divisible by number of heads'
        # Transformations for queries, keys, and values for all heads
        self.c_attn = nn.Linear(config.n_embd, 3 * config.n_embd)
        # Output projection
        self.c_proj = nn.Linear(config.n_embd, config.n_embd)
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        # Autoregressive mask - not needed due as using PyTorch's flash-attention implementation
        # self.register_buffer('mask', torch.tril(torch.ones(config.block_size, config.block_size))
        #     .view(1, 1, config.block_size, config.block_size))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape # batch_size, block_size, n_embd
        # Calculate queries, keys, and values for all heads in a single pass
        # H is the number of heads and C/H is the head size, C = H * C/H
        qkv = self.c_attn(x)
        q, k, v = qkv.split(self.n_embd, dim=2)
        q = q.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C/H)
        k = k.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C/H)
        v = v.view(B, T, self.n_head, C // self.n_head).transpose(1, 2) # (B, H, T, C/H)
        # Compute attention scores ('affinities')
        # W = q @ k.transpose(-2, -1) * (k.shape[-1] ** -0.5) # (B, H, T, C/H) @ (B, H, C/H, T) -> (B, H, T, T)
        # W = W.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf')) # Autoregressive mask
        # W = F.softmax(W, dim=-1)
        # Perform the attention-weighted sum
        # y = W @ v # (B, H, T, T) @ (B, H, T, C/H) -> (B, H, T, C/H)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True) # Flash-attention - https://arxiv.org/abs/2205.14135
        y = y.transpose(1, 2).contiguous().view(B, T, C) # Re-assemble all head outputs side by side
        y = self.c_proj(y)
        return y    

class MLP(nn.Module):
    """Single non-linear feed-forward layer."""

    def __init__(self, config: GPTConfig):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd)
        self.gelu = nn.GELU(approximate='tanh')
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.c_fc(x)
        x = self.gelu(x)
        x = self.c_proj(x)
        return x

class Block(nn.Module):
    """Transformer block with a causal self-attention layer and a feed-forward layer."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.ln_1 = nn.LayerNorm(config.n_embd)
        self.attn = CausalSelfAttention(config)
        self.ln_2 = nn.LayerNorm(config.n_embd)
        self.mlp = MLP(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln_1(x))
        x = x + self.mlp(self.ln_2(x))
        return x

class GPT(nn.Module):
    """A GPT model."""
    
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.transformer = nn.ModuleDict(dict(
            wte = nn.Embedding(config.vocab_size, config.n_embd), # Token embeddings
            wpe = nn.Embedding(config.block_size, config.n_embd), # Positional embeddings
            h = nn.ModuleList([Block(config) for _ in range(config.n_layer)]), # Transformer blocks
            ln_f = nn.LayerNorm(config.n_embd), # Final layer norm
        ))
        self.lm_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Weight sharing between embedding and output layers - https://arxiv.org/abs/1608.05859
        self.transformer.wte.weight = self.lm_head.weight

        # Initialise weights as per GPT-2
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
        # Scale init of residual layers as std grows with depth in residual streams
        for name, param in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(param, mean=0.0, std=0.02 * (2 * self.config.n_layer) ** -0.5)

    def forward(self, x: torch.Tensor, y: torch.Tensor = None) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = x.shape # batch_size, block_size
        assert T <= self.config.block_size, f'Sequence of length {T} exceeds block size {self.config.block_size}'
        pos = torch.arange(T, dtype=torch.long, device=x.device)
        pos_embd = self.transformer.wpe(pos) # (T) -> (T, C)
        tok_embd = self.transformer.wte(x) # (B, T) -> (B, T, C)
        z = tok_embd + pos_embd
        for block in self.transformer.h:
            z = block(z)
        z = self.transformer.ln_f(z)
        logits = self.lm_head(z) # (B, T, C) -> (B, T, V) where V is vocab_size
        loss = None
        if y is not None:
            # Flatten batch and sequence dimensions to (B*T, C) and (B*T) respectively, for cross-entropy loss
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        return logits, loss
    
    def configure_optimisers(self, weight_decay: float, lr: float) -> torch.optim.Optimizer:
        """Configure AdamW optimiser with weight decay and learning rate."""
        params = {name: param for name, param in self.named_parameters() if param.requires_grad}
        # Any parameter that is at least 2D has weight decay applied - i.e. all weight tensors
        # in matmuls + embeddings decay, all bias tensors don't.
        decay_params = [param for _, param in params.items() if param.dim() >= 2]
        no_decay_params = [param for _, param in params.items() if param.dim() < 2]
        optim_groups = [
            {'params': decay_params, 'weight_decay': weight_decay},
            {'params': no_decay_params, 'weight_decay': 0.0}
        ]
        # Use fused optimiser for faster training on GPU
        optimiser = torch.optim.AdamW(optim_groups, lr=lr, betas=(0.9, 0.95), eps=1e-8, fused=True)
        return optimiser
    
    @torch.no_grad()
    def generate(self, x: torch.Tensor, max_tokens: int = 64, n_samples: int = 1, temp: float = 1.0, top_k: int = 50, seed: int = None) -> torch.Tensor:
        rng = torch.Generator(device=x.device).manual_seed(seed)
        # Repeat the input context for each sample
        x = x.unsqueeze(0).repeat(n_samples, 1)
        """Generate a sequence of tokens given an initial context."""
        for _ in range(max_tokens):
            # Crop the sequence context to the last block_size tokens
            x = x if x.size(1) <= self.config.block_size else x[:, -self.config.block_size:]
            # Get the previous predictions
            logits, _ = self(x)
            # Scale the logits by the temperature and keep only the last prediction
            logits = logits[:, -1, :] / temp
            # Softmax for probabilities
            probs = F.softmax(logits, dim=1)
            # Top-k sampling
            topk_probs, topk_indices = torch.topk(probs, top_k, dim=-1)
            # Sample from the top-k probabilities
            ix = torch.multinomial(topk_probs, 1, generator=rng)
            # Gather sampled token indices
            x_next = torch.gather(topk_indices, -1, ix)
            # Concatenate sampled token to the sequence
            x = torch.cat((x, x_next), dim=1)
        return x