
# this function is wrong because it assumes every unicode char is a single byte, which is not true e.g. こ is represented by multiple bytes i.e. [227, 129, 147] or b'\xe3\x81\x93'
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])



# 257 -> (115, 116) -> b'st'
# 258 -> (101, 257) -> b'est'

# token_to_bytes(257, vocab)

# b"".join(tokens)

def token_to_bytes(token_id, vocab):
    value = vocab[token_id]

    if isinstance(value, bytes):
        return value
    
    return b''.join(token_to_bytes(t, vocab) for t in value)


# unicode defines around 155k characters across 168 scripts
# ord() gives us the integer representation of a unicode character
# chr() returns the char associated with an integer code
# print(chr(0))
# print(f"abc{chr(0)}d")
# print(ord("\x00"))

# training a tokenizer on all codepoints would lead to a large vocab (150k+) which is also sparse (a lot of chars are very rare)
# instead, we use a Unicode encoding scheme like UTF-8. UTF-16, UTF-32 
# test_string = "hello! こんにちは!"
# utf8_encoded = test_string.encode("utf-8")
# print(utf8_encoded)
# this gives us a bytes object
# print(type(utf8_encoded))

# print(len(test_string))

# it is not a 1-1 mapping, 13 input chars -> 23 bytes. one byte is not necessarily one unicode character
# print(len(utf8_encoded))
# print(list(utf8_encoded))

# [227, 129, 147] or b'\xe3\x81\x93' to represent this single char
# print(ord('こ'))
# print(chr(12371))
# print(list('hello world こ'.encode('utf-8')))

# we are basically mapping integers in the range of 0 - 155k to a more manageable range of 0 - 255
# this means that some chars will need multiple bytes to represent them
# this eliminates the "out of vocabulary" problem in language modeling
# since any input sequence can be represented as a sequence of integers from 0 to 255
# utf-16 or 32 why not? they lead to much larger sequences 
# utf-8 has the widest support 

# byte level tokenization solves the 'out of vocabulary problem' but it leads to very long input sequences which slows down model training
# language modeling on byte sequences is also hard because the longer input sequences create long-term dependencies in the data

# sub-word tokenization sits in the middle of word level and byte level tokenization
# it provides a good balance of sequence length and vocabulary size (as well as solving the out of vocab problem)
# it compresses commonly encountered byte sequences into one entry in the vocab
# e.g. if the byte sequence b'the' occurs often in the raw text training data, assigning it an entry in the vocab would reduce this 3 token sequence to one token

# how are these subword tokens identified? using the Byte Pair Encoding algorithm
# BPE is a compression algorithm that iteratively replaces ('merges') the most frequent pair of bytes with a single, new unused index
# this algo adds subword tokens to our vocab to maximise the compression of the input sequence
# if a word occurs enough times in the training data, it is added as a single subword unit
# BPE happens in 3 steps
# 1. Vocabulary initialization (special chars + 256 bytes values)
# 2. Pretokenization (rough pass over the corpus that reduces our work when counting how often pairs of characters)
# 3. Compute BPE merges

# our vocabulary initially is just the 256 byte values 0-255 and any special tokens e.g. <|endoftext|> which helps define document boundaries

def run_bpe():

    # input_str = """low low low low low\n lower lower widest widest widest\n newest newest newest newest newest newest"""
    import regex as re


    special_tokens = ["<|endoftext|>"]
    vocab = {i: bytes((i, )) for i in range(256)}
    next_id = 256

    for token in special_tokens:
        vocab[next_id] = token.encode("utf-8")
        next_id += 1

    merges = []
    # split documents by special tokens to prevent merging across document boundaries
    pattern = "|".join(re.escape(token) for token in special_tokens)


    f = open("/home/taihim/projects/cs336/assignment1-basics/tests/fixtures/corpus.en", "r")
    corpus = f.read()

    chunks  = re.split(pattern, corpus)
    
    PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""

    from collections import defaultdict

    counts = defaultdict(int)

    for chunk in chunks:
        for match in re.finditer(PAT, chunk):
            counts[tuple((match.group().encode('utf-8')))] += 1

    
    iterations = 500 - len(vocab)
    for i in range(iterations):

        merge_counts = defaultdict(int)
        
        for word_bytes, freq in counts.items():
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                merge_counts[pair] += freq
            

        max_count = max(merge_counts.values())
        candidates = [p for p, c in merge_counts.items() if c == max_count]
        
        def pair_to_bytes(pair):
            return (token_to_bytes(pair[0], vocab), token_to_bytes(pair[1], vocab))
        max_pair = max(candidates, key=pair_to_bytes)
    
        merges.append((token_to_bytes(max_pair[0], vocab), token_to_bytes(max_pair[1], vocab)))
        vocab[next_id] = token_to_bytes(max_pair[0], vocab) + token_to_bytes(max_pair[1], vocab)

        new_counts = defaultdict(int)
        for word_bytes, freq in counts.items():
            new_word = []
            i = 0
            while i < len(word_bytes):
                if i < len(word_bytes) - 1 and (word_bytes[i], word_bytes[i + 1]) == max_pair:
                    new_word.append(next_id)
                    i += 2  # Skip both tokens
                else:
                    new_word.append(word_bytes[i])
                    i += 1
            new_counts[tuple(new_word)] += freq  # Use += not =, in case duplicates after merge!

        counts = new_counts

        # print("New counts: ", new_counts)
        
        next_id += 1

    # for idx, merge in enumerate(merges):
    #     print(idx+1, merge, "\n")

    # print("\nMerges: ", merges, "\n")
    # print("Final vocab: ", vocab, "\n")

    # tokenize_str = "newest west"
    # encoded_str = list(tokenize_str.encode("utf-8"))

if __name__ == "__main__":
    import cProfile

    cProfile.run("run_bpe()")

