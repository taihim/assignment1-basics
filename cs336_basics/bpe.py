
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
    #


