"""
GPT (byte-level) Byte Pair Encoding (BPE) Tokeniser.

Based off the gpt2 and cl100k_base encodings from TikToken. Handles special tokens and
the regular expression splitting pattern for GPT-2 and GPT-4. The API is designed to be
identical to the TikToken tokeniser.

Run this script to train a new tokeniser on the first 10,000 texts in fineweb-edu to
generate a vocabulary of 50,257 tokens (50,000 BPE + 256 Byte tokens + <|endoftext|>
token) similar to GPT-2. The tokeniser can be saved to a file and loaded later to
encode and decode text.

References
1) GPT-2 Tokeniser:
https://github.com/openai/gpt-2/blob/master/src/encoder.py
2) TikToken:
https://github.com/openai/tiktoken
"""

import datasets
import regex as re

# Regular expression splitting patterns for GPT-2 and GPT-4.
# https://github.com/openai/tiktoken/blob/main/tiktoken_ext/openai_public.py
GPT2_SPLIT_PATTERN = r"'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"
GPT4_SPLIT_PATTERN = r"'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"

def consecutive_pairs(ints: list[int], freq: dict[tuple[int, int], int] = None) -> dict[tuple[int, int], int]:
    """
    Generate a dictionary of the frequencies of consecutive integers in the list.
    Example: [1, 2, 3, 1, 2] -> {(1, 2): 2, (2, 3): 1, (3, 1): 1}
    Optionally update an existing frequency dictionary.
    """
    freq = {} if freq is None else freq
    for pair in zip(ints, ints[1:]):
        freq[pair] = freq.get(pair, 0) + 1
    return freq

def replace_pair(ints: list[int], pair: tuple[int, int], new_int: int) -> list[int]:
    """
    Replace all consecutive occurrences of a pair of integers in the list with a new integer.
    Example: ints=[1, 2, 3, 1, 2], pair=(1, 2), new_int=4 -> [4, 3, 4]
    """
    new_ints = []
    i = 0
    while i < len(ints):
        # If not at the last position AND the pair matches, replace it
        if (i < len(ints) - 1) and ints[i:i+2] == list(pair):
            new_ints.append(new_int)
            i += 2
        else:
            new_ints.append(ints[i])
            i += 1
    return new_ints

class Tokeniser:
    """Base Tokeniser."""

    def __init__(self):
        self.merges = {} # Merges of tokens (tuple[int, int] -> int)
        self.pattern = "" # Pattern for tokenisation
        self._special_tokens = {} # Special tokens to be added to the vocab (str -> int)
        self.vocab = self._build_vocab() # Vocabulary of tokens (str -> int)

    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train on a text and build a vocabulary of size vocab_size."""
        raise NotImplementedError

    def encode(self, text: str) -> list[int]:
        """Encode a string into a sequence of tokens."""
        raise NotImplementedError

    def decode(self, tokens: list[int]) -> str:
        """Decode a sequence of tokens into a string."""
        raise NotImplementedError

    def _build_vocab(self) -> dict[str, int]:
        """Build the vocabulary from the merges and special tokens."""
        vocab = {i: bytes([i]) for i in range(256)} # Initialise with single bytes as tokens
        for (token_0, token_1), new_token in self.merges.items():
            vocab[new_token] = vocab[token_0] + vocab[token_1]
        for special, idx in self._special_tokens.items():
            vocab[idx] = special.encode("utf-8")
        return vocab
    
    def save(self, file_prefix: str) -> None:
        """Save the tokeniser to a file."""
        model_file = file_prefix + '.tkn'
        # Write the pattern, special tokens, and merges to the file
        with open(model_file, 'w') as f:
            f.write(f'{self.pattern}\n')
            f.write(f'{len(self._special_tokens)}\n')
            for special, token in self._special_tokens.items():
                f.write(f'{special} {token}\n')
            for token_1, token_2 in self.merges:
                f.write(f'{token_1} {token_2}\n')

    def load(self, tokeniser_file: str) -> None:
        """Load the tokeniser from a file."""
        assert tokeniser_file.endswith('.tkn')
        merges = {}
        special_tokens = {}
        new_token = 256
        # Read the pattern, special tokens, and merges from the file
        with open(tokeniser_file, 'r', encoding='utf-8') as f:
            self.pattern = f.readline().strip()
            n_special = int(f.readline().strip())
            for _ in range(n_special):
                special, token = f.readline().strip().split()
                special_tokens[special] = int(token)
            for line in f:
                token_1, token_2 = map(int, line.split())
                merges[(token_1, token_2)] = new_token
                new_token += 1
        self.merges = merges
        self._special_tokens = special_tokens
        self.vocab = self._build_vocab()

    def vocab_size(self) -> int:
        """Return the size of the vocabulary."""
        return len(self.vocab)

class GPTTokeniser(Tokeniser):
    """GPT BPE Tokeniser."""

    def __init__(self, tokeniser_file: str = None, pattern: str = None):
        super().__init__()
        self.pattern = GPT4_SPLIT_PATTERN if pattern is None else pattern # Default to GPT-4 pattern
        self.regex = re.compile(self.pattern) # Compiled regex pattern
        self._special_tokens = {} # Special tokens to be added to the vocab (str -> int)
        self._inv_special_tokens = {} # Inverse of special_tokens (int -> str)
        if tokeniser_file is not None:
            self.load(tokeniser_file)
        
    def train(self, text: str, vocab_size: int, verbose: bool = False) -> None:
        """Train on a text and build a vocabulary of size vocab_size."""
        assert vocab_size >= 256
        n_merges = vocab_size - 256
        text_chunks = re.findall(self.regex, text) # Split the text into chunks
        tokens = [list(text_chunk.encode('utf-8')) for text_chunk in text_chunks]
        merges = {} # Dictionary to store the merges
        vocab = {i: bytes([i]) for i in range(256)}

        # Merge the most frequent pair n_merges times to create new tokens
        for i in range(n_merges):
            # Find the most frequent consecutive pair of tokens
            freq_pairs = {}
            for token_chunk in tokens:
                consecutive_pairs(token_chunk, freq_pairs)
            max_pair = max(freq_pairs, key=freq_pairs.get)
            # Create a new token and assign it to an unused integer
            new_token = 256 + i
            # Replace the pair with the new token in each chunk
            tokens = [replace_pair(token_chunk, max_pair, new_token) for token_chunk in tokens]
            # Store the merge and the new token in the vocab
            merges[max_pair] = new_token
            vocab[new_token] = vocab[max_pair[0]] + vocab[max_pair[1]]
            if verbose:
                print(f'{i+1}/{n_merges}: {max_pair} -> {new_token}')

        self.merges = merges
        self.vocab = vocab

    def register_special_tokens(self, special_tokens: dict[str, int]) -> None:
        """Register special tokens to be added to the vocabulary."""
        self._special_tokens = special_tokens
        self._inv_special_tokens = {v: k for k, v in special_tokens.items()}

    def _encode_chunk(self, text_chunk: str) -> list[int]:
        """Encode a text chunk into a sequence of tokens."""
        token_chunk = list(text_chunk.encode('utf-8'))
        while len(token_chunk) > 1:
            freq_pairs = consecutive_pairs(token_chunk)
            # Find the most frequent consecutive pair that has been merged
            most_freq = min(freq_pairs, key=lambda pair: self.merges.get(pair, float('inf')))
            # If there are no more merges avaliable the keys all be inf and most_freq will be the first pair
            if most_freq not in self.merges:
                break # No more merges to apply
            # Merge the pair into a new token
            new_token = self.merges[most_freq]
            token_chunk = replace_pair(token_chunk, most_freq, new_token)
        return token_chunk
    
    def encode_ordinary(self, text: str) -> list[int]:
        """Encode text, ignoring any special tokens."""
        text_chunks = re.findall(self.regex, text) # Split the text into chunks
        # Encode each chunk separetely and concatenate the tokens
        tokens = []
        for text_chunk in text_chunks:
            token_chunk = self._encode_chunk(text_chunk)
            tokens.extend(token_chunk)
        return tokens

    def encode(self, text: str, allowed_special: str = 'none_raise') -> list[int]:
        """
        Encode text, handling special tokens. allowed_special can be 'all'|'none'|'none_raise'
        or a custom set of special tokens. If 'none_raise', then an error is raised if any
        special token is encountered in text.
        """
        special = None
        if allowed_special == 'all':
            special = self._special_tokens
        elif allowed_special == 'none':
            special = {}
        elif allowed_special == 'none_raise':
            special = {}
            assert all(token not in text for token in self._special_tokens)
        elif isinstance(allowed_special, set):
            special = {k: v for k, v in self._special_tokens.items() if k in allowed_special}
        else:
            raise ValueError(f'Invalid value for allowed_special: {allowed_special}')
        
        if not special:
            return self.encode_ordinary(text) # No special tokens to handle
        # Split the text into chunks and special tokens
        special_pattern = f'({"|".join(re.escape(k) for k in special)})'
        special_chunks = re.split(special_pattern, text)
        tokens = []
        # Encode each chunk and special token separetely and concatenate the tokens
        for part in special_chunks:
            if part in special:
                tokens.append(special[part]) # Encode the special token as an integer
            else:
                tokens.extend(self.encode_ordinary(part)) # Encode the chunk normally
        return tokens
    
    def decode(self, tokens: list[int]) -> str:
        """Decode a sequence of tokens into a string."""
        byte_chunk = []
        # Decode each token into a byte chunk and concatenate them
        for token in tokens:
            if token in self.vocab:
                byte_chunk.append(self.vocab[token])
            elif token in self._inv_special_tokens: # Token is a special token
                byte_chunk.append(self._inv_special_tokens[token].encode('utf-8'))
            else:
                raise ValueError(f'Invalid token: {token}')
        bytes_ = b''.join(byte_chunk)
        text = bytes_.decode('utf-8', errors='replace') # Replace unknown characters
        return text
    
if __name__ == '__main__':
    vocab_size = 50257 # Target number of unique tokens (50,000 BPE + 256 Byte tokens + <|endoftext|> token)

    # Special tokens to be added to the vocabulary.
    special_tokens = {
        '<|endoftext|>': 50257
    }

    remote_name = 'sample-10BT' # Dataset name on Hugging Face

    # Load training text from fineweb-edu dataset
    dataset = datasets.load_dataset('HuggingFaceFW/fineweb-edu', name=remote_name, split='train', streaming=True)

    # Train on the first 10,000 documents
    text = '\n'.join([item['text'] for i, item in enumerate(dataset) if i < int(1e5)])

    tokeniser = GPTTokeniser()
    tokeniser.train(text, vocab_size=vocab_size, verbose=True)
    tokeniser.register_special_tokens(special_tokens)
    tokeniser.save('gpt')