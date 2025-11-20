
# this function is wrong because it assumes every unicode char is a single byte, which is not true e.g. こ is represented by multiple bytes i.e. [227, 129, 147] or b'\xe3\x81\x93'
def decode_utf8_bytes_to_str_wrong(bytestring: bytes):
    return "".join([bytes([b]).decode("utf-8") for b in bytestring])


if __name__ == "__main__":
    # unicode defines around 155k characters across 168 scripts
    # ord() gives us the integer representation of a unicode character
    # chr() returns the char associated with an integer code
    print(chr(0))
    print(f"abc{chr(0)}d")
    print(ord("\x00"))

    # training a tokenizer on all codepoints would lead to a large vocab (150k+) which is also sparse (a lot of chars are very rare)
    # instead, we use a Unicode encoding scheme like UTF-8. UTF-16, UTF-32 
    test_string = "hello! こんにちは!"
    utf8_encoded = test_string.encode("utf-8")
    print(utf8_encoded)
    # this gives us a bytes object
    print(type(utf8_encoded))

    print(len(test_string))

    # it is not a 1-1 mapping, 13 input chars -> 23 bytes. one byte is not necessarily one unicode character
    print(len(utf8_encoded))
    print(list(utf8_encoded))

    # [227, 129, 147] or b'\xe3\x81\x93' to represent this single char
    print(ord('こ'))
    print(chr(12371))
    print(list('hello world こ'.encode('utf-8')))

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

    input_str = """low low low low low\n lower lower widest widest widest\n newest newest newest newest newest newest"""

    # using a naive pretokenizer e.g. one that splits on whitespace

    vocab = {i: (i,) for i in range(256)}
    vocab[256] = ("<|endoftext|>",)
    next_id = 257

    merges = []

    from collections import defaultdict
    counts = defaultdict(int)
    out = "".join(input_str.split("\n")).split(" ")
    print(out)
    for word in out:
        counts[tuple(word.encode('utf-8'))] += 1
    print("Initial counts: ", counts)

    for i in range(6):

        merge_counts = defaultdict(int)
        
        for word_bytes, freq in counts.items():
            for i in range(len(word_bytes) - 1):
                pair = (word_bytes[i], word_bytes[i + 1])
                merge_counts[pair] += freq
            
        print("Merge counts: ", merge_counts)

        max_pair = max(merge_counts, key=lambda pair: (merge_counts[pair], pair)) # get largest count, and if tied, get lexographically largest pair
        print("Max Pair: ", max_pair)

        merges.append(max_pair)
        vocab[next_id] = max_pair

        new_counts = defaultdict(int)
        for word_bytes, freq in counts.items():
            copy_bytes = list(word_bytes)
            for i in range(len(word_bytes) - 1):
                old_pair = (word_bytes[i], word_bytes[i + 1])
                if old_pair == max_pair:
                    copy_bytes[i:i+2] = [next_id]
                
            new_counts[tuple(copy_bytes)] = freq
        
        next_id += 1

        counts = new_counts
        print("New counts: ", new_counts)

    print(vocab)
    
    # print(vocab[256])
    # print(vocab[257])
    # print(vocab[258])
    # print(vocab[259])
    # print(vocab[260])
    # print(vocab[261])
    
    # print("".join([chr(uni) for uni in vocab[256]]))
    # print("".join([chr(uni) for uni in vocab[257]]))
    # print("".join([chr(uni) for uni in vocab[258]]))
    # print("".join([chr(uni) for uni in vocab[259]]))
    # print("".join([chr(uni) for uni in vocab[260]]))
    # print("".join([chr(uni) for uni in vocab[261]]))
    
    # for i in range(1):
    #     max_value = -1
    #     max_keys = [0]
    #     for key, val in merge_counts.items():
    #         if val >= max_value:
    #             if val == max_value:
    #                 elem = max_keys.pop()
    #                 max_keys.append(max(key, elem))

    #             else:
    #                 max_value = val

    #                 max_keys.pop()
    #                 max_keys.append(key)
        
    #     value_to_merge = tuple("".join(max_keys[0]).encode('utf-8'))
    #     print("Value to merge: ", value_to_merge, type(value_to_merge[0]))
        
    #     for key in counts.keys():
    #         print(key)
            
    #         if value_to_merge in key:
    #             print(key)
    
    # print((1,2,3) in (1,2,3,4,5))
    # print(bytes((110, 101, 119, 101, 115, 116)).decode('utf-8'))