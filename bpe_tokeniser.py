import re

# The main Tokenizer class
class Tokenizer:
    """
    A BPE Tokenizer that learns merge rules from a corpus and can then
    encode/decode text using those rules. It operates at the byte level.
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

    def _build_vocab(self):
        # The initial vocabulary is simply the first 256 integers, each mapping
        # to its corresponding byte value. This ensures that any raw byte stream
        # can be represented at the start of the tokenization process.
        return {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        """
        Trains the tokenizer on a given text to learn merge rules.

        Args:
            text (str): The training text corpus.
            vocab_size (int): The target final vocabulary size.
        """
        if vocab_size <= 256:
            return  # Nothing to learn if the vocab size is just the base bytes.

        num_merges = vocab_size - 256
        tokens = list(text.encode("utf-8")) # Start with the raw byte sequence.

        # This is the core training loop. We perform `num_merges` iterations.
        for i in range(num_merges):
            # 1. Find the most frequent adjacent pair of tokens.
            stats = self._get_stats(tokens)
            if not stats:
                break # Stop if there are no more pairs to merge.
            
            # This is the "greedy" part of the BPE algorithm. We find the pair
            # with the highest frequency and merge it. This choice is local
            # and may not be globally optimal, but it's effective in practice.
            pair = max(stats, key=stats.get)
            
            # 2. Assign a new token ID for this pair.
            # New tokens are numbered sequentially starting from 256.
            new_token_id = 256 + i
            
            print(f"Merging {pair} -> {new_token_id}")
            # 3. Replace all occurrences of the pair with the new token ID.
            tokens = self._merge_pair(tokens, pair, new_token_id)
            
            # 4. Store the merge rule and update the vocabulary.
            self.merges[pair] = new_token_id
            # The new token's byte representation is the concatenation of the
            # bytes of the two tokens that formed it.
            self.vocab[new_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        """
        Encodes a string into a sequence of token IDs using the learned merges.
        This method should be called *after* training.
        """
        # Start with the raw byte representation of the text.
        tokens = list(text.encode("utf-8"))
        
        # Keep merging until no more mergeable pairs are found.
        while True:
            stats = self._get_stats(tokens)
            
            # Find the next pair to merge. This is the most crucial part of encoding.
            # We must apply the merges in the same order they were learned during training.
            # We find the pair that exists in the current text AND has the lowest
            # rank in our learned `self.merges`. The rank is implicitly its value
            # (256, 257, 258...), so we just need to find the merge with the lowest value.
            # `self.merges.get(p, float("inf"))` returns the new token ID if the pair `p`
            # is in our merge rules, or infinity otherwise. Finding the `min`
            # ensures we pick the earliest learned rule.
            pair_to_merge = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            # If the best pair has a rank of infinity, it means no pairs in the
            # current text are present in our learned `self.merges` rules.
            # So, we are done encoding.
            if self.merges.get(pair_to_merge, float("inf")) == float("inf"):
                break
            
            # If we found a valid merge, get its new token ID and apply the merge.
            new_token_id = self.merges[pair_to_merge]
            tokens = self._merge_pair(tokens, pair_to_merge, new_token_id)
            
        return tokens

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

    # Train it on the text
    tokenizer.train(text, vocab_size=300) # Let's learn a few merges

    # Encode the text
    encoded_text = tokenizer.encode(text)
    print("\nEncoded Text:", encoded_text)
    print(f"Length: {len(encoded_text)}")

    # Decode it back
    decoded_text = tokenizer.decode(encoded_text)
    print("\nDecoded Text:", decoded_text)
    
    print(f"\nCompression ratio: {len(text.encode('utf-8')) / len(encoded_text):.2f}x") 