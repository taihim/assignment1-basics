import tiktoken
enc = tiktoken.get_encoding("cl100k_base")
# for n in ["100", "1000", "7294", "2024", "2047"]:
#     print(n, enc.encode(n))


# Years that might have different frequencies
for n in ["1999", "2000", "2001", "1776", "1777"]:
    print(n, enc.encode(n))

# Prices
for n in ["$100", "$1000", "$99", "$999"]:
    print(n, enc.encode(n))