import regex as re

# The main Tokenizer class
class Tokenizer:
    """
    A BPE Tokenizer that learns merge rules from a corpus and can then
    encode/decode text using those rules. It operates at the byte level.
    This implementation is aligned with the GPT-2 paper, which uses a regex
    pattern for pre-tokenization.
    """
    def __init__(self):
        # self.merges stores the learned merge rules.
        # It's a dictionary where the key is a tuple of two token IDs to merge,
        # and the value is the new token ID resulting from the merge.
        # e.g., {(101, 32): 257} means 'e' and ' ' merge to token 257.
        self.merges = {}  # (int, int) -> int

        # self.vocab stores the mapping from a token ID to its byte representation.
        # This is essential for decoding. Initially, it contains all single bytes.
        # e.g., {257: b'e '}
        self.vocab = self._build_vocab()  # int -> bytes

        # GPT-2's regex pattern for pre-tokenization. This is key to its effectiveness.
        # It splits text into chunks that are "atomic" for the BPE algorithm.
        # This prevents merges across categories like letters, numbers, and punctuation.
        self.pat = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")


    def _build_vocab(self):
        # The initial vocabulary is simply the first 256 integers, each mapping
        # to its corresponding byte value.
        return {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size, verbose=False):
        """
        Trains the tokenizer on a given text to learn merge rules.

        Args:
            text (str): The training text corpus.
            vocab_size (int): The target final vocabulary size.
        """
        if vocab_size <= 256:
            return  # Nothing to learn if the vocab size is just the base bytes.

        # 1. Pre-tokenize the text into words/chunks using the regex pattern.
        text_chunks = re.findall(self.pat, text)

        # 2. Convert to a vocabulary of byte-encoded words and their frequencies.
        # The BPE algorithm works on the frequency of subword units within these words.
        word_freqs = {}
        for chunk in text_chunks:
            word_freqs[chunk.encode("utf-8")] = word_freqs.get(chunk.encode("utf-8"), 0) + 1

        # Represent each word as a list of its initial byte-level tokens.
        splits = {word: list(word) for word in word_freqs}
        
        num_merges = vocab_size - 256
        for i in range(num_merges):
            # 3. Calculate pair frequencies across the entire vocabulary of words.
            # This is different from the stream approach; we consider pairs within
            # each word, weighted by how often that word appears.
            pair_stats = self._get_pair_stats(splits, word_freqs)
            if not pair_stats:
                break
            
            # 4. Find the most frequent pair to merge.
            best_pair = max(pair_stats, key=pair_stats.get)
            
            # 5. Perform the merge.
            # This involves creating a new token ID and replacing the `best_pair`
            # in all words in our `splits` dictionary where it occurs.
            new_token_id = 256 + i
            splits = self._merge_pair_in_splits(splits, best_pair, new_token_id)
            
            # 6. Store the merge rule and update the vocabulary.
            self.merges[best_pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            if verbose:
                print(f"Merge {i+1}/{num_merges}: {best_pair} -> {new_token_id} ({self.vocab[new_token_id]})")

    def _get_pair_stats(self, splits, word_freqs):
        """Helper to count pair frequencies across the word vocabulary."""
        pair_counts = {}
        for word, freq in word_freqs.items():
            symbols = splits[word]
            for pair in zip(symbols, symbols[1:]):
                pair_counts[pair] = pair_counts.get(pair, 0) + freq
        return pair_counts

    def _merge_pair_in_splits(self, splits, pair, new_token_id):
        """Helper to perform a merge operation on the `splits` dictionary."""
        new_splits = {}
        for word, split in splits.items():
            new_split = []
            i = 0
            while i < len(split):
                if i < len(split) - 1 and (split[i], split[i+1]) == pair:
                    new_split.append(new_token_id)
                    i += 2
                else:
                    new_split.append(split[i])
                    i += 1
            new_splits[word] = new_split
        return new_splits

    def encode(self, text):
        """
        Encodes a string into a sequence of token IDs using the learned merges.
        This method should be called *after* training.
        """
        # 1. Pre-tokenize the input text into chunks.
        text_chunks = re.findall(self.pat, text)
        
        output_ids = []
        for chunk in text_chunks:
            # 2. For each chunk, convert to bytes and then apply merges.
            tokens = list(chunk.encode("utf-8"))
            
            # 3. Iteratively apply learned merges.
            while len(tokens) > 1:
                # Find the pair with the earliest merge rule (lowest rank).
                stats = self._get_stats(tokens)
                pair_to_merge = min(stats, key=lambda p: self.merges.get(p, float("inf")))
                if pair_to_merge not in self.merges:
                    break # No more merges possible for this chunk.
                
                new_token_id = self.merges[pair_to_merge]
                tokens = self._merge_pair(tokens, pair_to_merge, new_token_id)

            output_ids.extend(tokens)
            
        return output_ids

    def decode(self, ids):
        """
        Decodes a sequence of token IDs back into a string.
        """
        # Retrieve the byte representation for each token ID from the vocabulary.
        # The result is a list of bytes objects.
        tokens_bytes = [self.vocab[idx] for idx in ids]
        # Concatenate all byte strings into a single byte string.
        tokens = b"".join(tokens_bytes)
        # Decode the UTF-8 byte string back into a regular Python string.
        # `errors="replace"` will insert a placeholder for any invalid byte sequences.
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _get_stats(self, ids):
        """
        Calculates the frequency of adjacent pairs in a sequence of token IDs.
        """
        counts = {}
        for pair in zip(ids, ids[1:]): # A clever way to iterate over consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, idx):
        """
        Replaces all occurrences of a specific pair of token IDs with a single new token ID.
        This is a helper function used in both training and encoding.
        
        NOTE: This creates a new list, which is correct but can be slow for very
        large token sequences. More optimized implementations might modify the list in-place.
        """
        new_ids = []
        i = 0
        while i < len(ids):
            # Check for the target pair starting at the current position.
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2 # Skip over both elements of the merged pair.
            else:
                new_ids.append(ids[i])
                i += 1
        return new_ids


if __name__ == "__main__":
    # Example usage:
    text = "Arsenal have completed the signing of England winger Noni Madueke from Chelsea for an initial fee of £48.5m. The 23-year-old was part of Chelsea's squad at the Club World Cup in the United States but left the camp before last Sunday's 3-0 win against Paris St-Germain in the final to finalise his move to Mikel Arteta's side. Madueke has signed a five-year contract at Emirates Stadium, with his fee rising to just over £50m with add-ons. Humbled and blessed to be here. Thank you to everyone that made this possible, he wrote on Instagram."
    
    # Create a tokenizer instance
    tokenizer = Tokenizer()

    # Train it on the text, with verbose output to see the merges
    print("--- Training ---")
    tokenizer.train(text, vocab_size=300, verbose=True)

    # Encode the text
    print("\n--- Encoding ---")
    encoded_text = tokenizer.encode(text)
    print("\nEncoded Text:", encoded_text)
    print(f"Length: {len(encoded_text)}")

    # Decode it back
    print("\n--- Decoding ---")
    decoded_text = tokenizer.decode(encoded_text)
    print("\nDecoded Text:", decoded_text)
    
    print(f"\nCompression ratio: {len(text.encode('utf-8')) / len(encoded_text):.2f}x") 