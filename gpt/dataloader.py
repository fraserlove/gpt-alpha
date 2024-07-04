"""
Data loader for GPT training.

Loads tokenised data shards from disk (cache/fineweb_edu_10B) and returns batches of training data.
The data loader shuffles the shards and documents within each shard to ensure that documents are not
seen in the same order during training.
"""

import os
import torch
import tiktoken
import numpy as np

class GPTDataLoader:
    """Data loader for GPT training."""

    def __init__(self, B: int, T: int, proc_rank: int, n_proc: int, split: str):
        self.B = B
        self.T = T
        self.proc_rank = proc_rank
        self.n_proc = n_proc
        self.split = split
        assert split in {'train', 'val'}
        self.rng = np.random.default_rng(42)

        data_root = 'cache/fineweb_edu_10B'
        shards = sorted([shard for shard in os.listdir(data_root) if split in shard])
        shards = [os.path.join(data_root, shard) for shard in shards]
        self.shards = shards
        assert len(shards) > 0, f'No shards found for split {split}'
        self.reset()

    def reset(self) -> None:
        """Shuffle the shards and reset the data loader."""
        self.current_shard = 0
        if self.split == 'train':
            self.rng.shuffle(self.shards)
        # Load the first shard and reset the current position
        self.tokens = self.load_shard(self.shards[self.current_shard])
        self.current_pos = self.B * self.T * self.proc_rank

    def load_tokens(self, filename: str) -> torch.Tensor:
        """Load tokenised data from a file."""
        return torch.tensor(np.load(filename).astype(np.int32), dtype=torch.long)
    
    def shuffle_shard(self, shard: torch.Tensor) -> torch.Tensor:
        """Shuffle the documents within a shard."""
        # Load the GPT-2 tokeniser
        tokeniser = tiktoken.get_encoding('gpt2') # or GPTTokeniser('gpt.tkn')
        # Get the token ID for the end of text token
        eot_token = tokeniser._special_tokens['<|endoftext|>']
        # Split shard into documents using the end of text token
        eot_positions = (torch.where(shard == eot_token)[0] + 1).tolist()
        documents = [shard[start:end] for start, end in zip([0] + eot_positions[:-1], eot_positions)]
        # Shuffle the documents
        self.rng.shuffle(documents)
        return torch.cat(documents)

    def load_shard(self, shard_path: str) -> torch.Tensor:
        """Load a shard of tokenised data."""
        shard = self.load_tokens(shard_path)
        if self.split == 'train':
            shard = self.shuffle_shard(shard)
        return shard

    def next_batch(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the next batch of training data."""
        B, T = self.B, self.T # batch_size, block_size
        buffer = self.tokens[self.current_pos : self.current_pos + B * T + 1]
        x = buffer[:-1].view(B, T)
        y = buffer[1:].view(B, T)
        self.current_pos += B * T * self.n_proc
        # If end of the shard is reached, move to the next shard
        if self.current_pos + (B * T * self.n_proc + 1) > len(self.tokens):
            self.current_shard += 1
            # If all shards have been seen (epoch completed), reset the data loader and reshuffle the shards
            if self.current_shard == len(self.shards):
                self.reset()
            else:
                self.tokens = self.load_shard(self.shards[self.current_shard])
                self.current_pos = B * T * self.proc_rank
        return x, y