import re

# The main Tokenizer class
class Tokenizer:
    def __init__(self):
        # Initialize with a default vocabulary and merge rules
        self.merges = {}  # (int, int) -> int
        self.vocab = self._build_vocab()  # int -> bytes

    def _build_vocab(self):
        # The initial vocabulary is the 256 bytes
        return {i: bytes([i]) for i in range(256)}

    def train(self, text, vocab_size):
        if vocab_size <= 256:
            return

        num_merges = vocab_size - 256
        tokens = list(text.encode("utf-8"))

        for i in range(num_merges):
            stats = self._get_stats(tokens)
            if not stats:
                break
            
            pair = max(stats, key=stats.get)
            new_token_id = 256 + i
            
            print(f"Merging {pair} -> {new_token_id}")
            tokens = self._merge_pair(tokens, pair, new_token_id)
            
            self.merges[pair] = new_token_id
            self.vocab[new_token_id] = self.vocab[pair[0]] + self.vocab[pair[1]]

    def encode(self, text):
        tokens = list(text.encode("utf-8"))
        while True:
            stats = self._get_stats(tokens)
            # Find the pair to merge that has the lowest rank in our merge rules
            pair_to_merge = min(stats, key=lambda p: self.merges.get(p, float("inf")))
            
            # If the lowest rank is infinity, it means no pairs in the text are in our merge rules
            if self.merges.get(pair_to_merge, float("inf")) == float("inf"):
                break
            
            new_token_id = self.merges[pair_to_merge]
            tokens = self._merge_pair(tokens, pair_to_merge, new_token_id)
        return tokens

    def decode(self, ids):
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")
        return text

    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge_pair(self, ids, pair, idx):
        new_ids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i += 2
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
    
'''
Note that the Tokeniser is a completely separate, indepedent module from the LLM.
It has its own training dataset of text (which could be different from that of the LLM),
on which you train the vocabulary using BPE. It, then, translates back and forth between 
raw text and sequences of tokens. The LLM later only ever sees the tokens and never directly deal with any text

'''

