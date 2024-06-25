"""
A data loader for GPT training.
"""

import os
import torch
import numpy as np

def load_tokens(filename: str) -> torch.Tensor:
    return torch.tensor(np.load(filename).astype(np.int32), dtype=torch.long)

class GPTDataLoader:
    """Data loader for GPT training."""

    def __init__(self, B: int, T: int, process_rank: int, num_processes: int, split: str):
        self.B = B
        self.T = T
        self.process_rank = process_rank
        self.num_processes = num_processes
        self.current_pos = B * T * process_rank
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
        self.current_position = self.B * self.T * self.process_rank

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        B, T = self.B, self.T # batch_size, block_size
        buffer = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.current_pos += B * T * self.num_processes
        if self.current_pos + (B * T * self.num_processes + 1) > len(self.tokens):
            self.current_shard = (self.current_shard + 1) % len(self.shards)
            self.tokens = load_tokens(self.shards[self.current_shard])
            self.current_pos = B * T * self.process_rank
        return x, y