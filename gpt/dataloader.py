"""
Data loader for GPT training.

Loads tokenised data shards from disk (cache/fineweb_edu_10B) and returns batches of training data.
"""

import os
import torch
import numpy as np

def load_tokens(filename: str) -> torch.Tensor:
    return torch.tensor(np.load(filename).astype(np.int32), dtype=torch.long)

class GPTDataLoader:
    """Data loader for GPT training."""

    def __init__(self, B: int, T: int, proc_rank: int, n_proc: int, split: str):
        self.B = B
        self.T = T
        self.proc_rank = proc_rank
        self.n_proc = n_proc
        assert split in {'train', 'val'}

        data_root = 'cache/fineweb_edu_10B'
        shards = sorted([shard for shard in os.listdir(data_root) if split in shard])
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(shards) > 0, f'No shards found for split {split}'
        self.reset()

    def reset(self) -> None:
        # Load the first shard and reset the current position
        self.current_shard = 0
        self.tokens = load_tokens(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.proc_rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T # batch_size, block_size
        buffer = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.current_pos += B * T * self.n_proc
        if self.current_pos + (B * T * self.n_proc + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B * T * self.proc_rank
        return x, y