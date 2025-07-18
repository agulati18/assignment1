def bpe_tokeniser(text):
    """
    BPE tokeniser

    Tokenisation is at the heart of weirdness in LLMs. Lots of issues from the LLM
    are due to tokenisation. Spelling errors, punctuation, etc.

    BPE is a simple tokenisation method that is used in many LLMs. It is a greedy
    algorithm that finds the most frequent adjacent pair of characters and merges them.

    A lot more tokens are used in non-English languages. The non-English text is stretched 
    out in the context of the Transformer. 
    
    The algorithm is as follows:
    1. Split the text into words. This is the inital vocab
    2. Find the most frequent adjancent pair of characters
    3. Merge the pair and add the new token to the vocab
    4. Repeat until the desired vocabulary size is reached

    There is a sweet spot for the vocabulary size. Sufficiently dense and efficient.
    Iteratively compresses the vocabulary size as we mint new tokens and add them to the vocab, 
    replacing the existing tokens
    
    """

    tokens = text.encode('utf-8') # raw bytes
    tokens = list(map(int, tokens) ) # convert to list of integers
    # print("length of text: ", len(text))
    # print("length of tokens: ", len(tokens))

    # Find the most frequent adjacent pair of characters
    def get_stats(ids):
        counts = {}
        for pair in zip(ids, ids[1:]): # iterate over consecutive elements
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    stats = get_stats(tokens)
    # print(stats)
    # print(sorted(stats.items(), key=lambda x: x[1], reverse=True))

    # Merge the pair and add the new token to the vocab
    # top_pair = max(stats, key = stats.get)
    # print(top_pair)

    def merge_pair(ids, pair, idx):
        '''
        In the list of ints(ids), replace all consecutive occurrences of the pair with a new token idx
        '''
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


    # While Loop 
    vocab_size = 276 # target vocab size (desired final vocab size)
    num_merges = vocab_size - 256 # number of merges to perform
    ids = list(tokens) # copy so we don't destroy the original

    merges = {} # (int, int) -> int
    for i in range(num_merges):
        stats = get_stats(ids)
        pair = max(stats, key = stats.get)
        idx = 256 + i
        print(f"merging {pair} into a new token {idx}")
        ids = merge_pair(ids, pair, idx)
        merges[pair] = idx
    return merges, ids


if __name__ == "__main__":
    text = "Arsenal have completed the signing of England winger Noni Madueke from Chelsea for an initial fee of £48.5m. The 23-year-old was part of Chelsea's squad at the Club World Cup in the United States but left the camp before last Sunday's 3-0 win against Paris St-Germain in the final to finalise his move to Mikel Arteta's side. Madueke has signed a five-year contract at Emirates Stadium, with his fee rising to just over £50m with add-ons. Humbled and blessed to be here. Thank you to everyone that made this possible, he wrote on Instagram."
    merges, ids = bpe_tokeniser(text)
    print("Merges:", merges)
    print("Final token IDs:", ids)
    
