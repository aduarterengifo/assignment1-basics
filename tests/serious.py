import itertools
import os
import regex as re
from typing import BinaryIO
import multiprocessing as mp

PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""


def find_chunk_boundaries(file: BinaryIO, desired_num_chunks: int, split_special_token: bytes) -> list[int]:
    """
    Chunk the file into parts that can be counted independently.
    May return fewer chunks if the boundaries end up overlapping.
    """
    assert isinstance(split_special_token, bytes), "Must represent special token as a bytestring"

    # Get total file size in bytes
    file.seek(0, os.SEEK_END)
    file_size = file.tell()
    file.seek(0)

    chunk_size = file_size // desired_num_chunks

    # Initial guesses for chunk boundary locations, uniformly spaced
    # Chunks start on previous index, don't include last index
    chunk_boundaries = [i * chunk_size for i in range(desired_num_chunks + 1)]
    chunk_boundaries[-1] = file_size

    mini_chunk_size = 4096  # Read ahead by 4k bytes at a time

    for bi in range(1, len(chunk_boundaries) - 1):
        initial_position = chunk_boundaries[bi]
        file.seek(initial_position)  # Start at boundary guess
        while True:
            mini_chunk = file.read(mini_chunk_size)  # Read a mini chunk

            # If EOF, this boundary should be at the end of the file
            if mini_chunk == b"":
                chunk_boundaries[bi] = file_size
                break

            # Find the special token in the mini chunk
            found_at = mini_chunk.find(split_special_token)
            if found_at != -1:
                chunk_boundaries[bi] = initial_position + found_at
                break
            initial_position += mini_chunk_size

    # Make sure all boundaries are unique, but might be fewer than desired_num_chunks
    return sorted(set(chunk_boundaries))


def pre_tokenize_chunk(special_tokens: list[str], input_path: str, start: int, end: int):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sub_chunks = re.split(
            "|".join([re.escape(tok) for tok in special_tokens + [re.escape("<|endoftext|>")]]), chunk
        )
        # run tokenization for each of the sub_chunks.
        iterators = [re.finditer(PAT, sub_chunk) for sub_chunk in sub_chunks]
        flattened_iterators = itertools.chain(*iterators)
        store = {}
        for match in flattened_iterators:
            res = match.group()
            res_bytes = tuple(bytes([c]) for c in res.encode("utf-8"))
            store[res_bytes] = store.get(res_bytes, 0) + 1
        return store


def get_simple_pairs(pre_tok_dic: dict[tuple[bytes], int]):
    cow: dict[tuple[bytes], int] = {}
    # for every pre_token.
    for pre_token_key, value in pre_tok_dic.items():
        # for every successive_pair:
        for start in range(0, len(pre_token_key), 2):
            if start + 1 >= len(pre_token_key):
                continue
            # set new_key to be the successive_pair of bytes
            new_key = pre_token_key[start] + pre_token_key[start + 1]
            # increment successive_pair of bytes by the frequency of words where they appear.
            cow[new_key] = cow.get(new_key, 0) + 1
    return cow


def merge_at_i(bytes: tuple[bytes], i: int):
    return bytes[0:i] + (bytes[i] + bytes[i + 1],) + bytes[i + 2 :]


def update_dic(pre_tok_dic: dict[tuple[bytes], int], max_pair: tuple[bytes, bytes]):
    # for every pre_token.
    new_pre_tok_dic = {}
    for pre_tok, value in pre_tok_dic.items():
        marks = []
        new_pre_tok = pre_tok
        for start in range(0, len(pre_tok), 2):
            if start + 1 >= len(pre_tok):
                continue
            pair = (pre_tok[start], pre_tok[start + 1])
            if pair == max_pair:
                marks.append(start)
        for mark in marks:
            new_pre_tok = merge_at_i(pre_tok, mark)
        new_pre_tok_dic[new_pre_tok] = new_pre_tok_dic.get(new_pre_tok, 0) + value

    return new_pre_tok_dic


# not optimized for now.
def merge(pre_tok_dic: dict[tuple[bytes], int], stopping_condition: int):
    cow = get_simple_pairs(pre_tok_dic)
    # print("cow", cow)
    # INSERT_YOUR_CODE
    # Get top 5 key-value pairs from cow based on values (descending)
    top5 = sorted(cow.items(), key=lambda x: x[1], reverse=True)[:5]
    print("Top 5 pairs by frequency:", top5)
    max_pairs: list[tuple[bytes, bytes]] = []
    while len(max_pairs) < stopping_condition:
        max_pair = max(cow, key=cow.get)
        max_pairs.append(max_pair)
        pre_tok_dic = update_dic(pre_tok_dic, max_pair)
        cow = get_simple_pairs(pre_tok_dic)

    return max_pairs


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens_len = len(special_tokens)
    stopping_condition = vocab_size - 256 - special_tokens_len
    # open the input path
    with open(input_path, "rb") as f:
        # find the chunk boundaries
        boundaries = find_chunk_boundaries(f, 8, "<|endoftext|>".encode("utf-8"))
        with mp.Pool() as pool:
            results = pool.starmap(
                pre_tokenize_chunk,
                [(special_tokens, input_path, start, end) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )
        # results = [pre_tokenize_chunk(chunk) for chunk in chunks]
        combined = {}
        for d in results:
            combined.update(d)

        max_pairs = merge(combined, stopping_condition)
        vocab: dict[int, bytes] = {}
        # single-bytes
        for i in range(0, 256):
            vocab[i] = chr(i)
        # special_tokens
        for i, token in enumerate(special_tokens):
            vocab[i + 256] = bytes(token.encode("utf-8"))

        # vocab
        for i, (a, b) in enumerate(max_pairs):
            vocab[i + 256 + special_tokens_len] = a + b

        return (vocab, max_pairs)


# suppose the best pair was 'a' 't'
# then '*' 'a' is messed up i.e. now 0?
# clearly 'a' 't' is messed up
# clearly 'a' '*' is also messed up. i.e. now 0?
# from the pre tokenization we have just single bytes. so even for something like 'k' 'a' 't'.
# do I need to find that?


if __name__ == "__main__":
    (vocab, max_pairs) = train_bpe("./tests/TinyStories-valid.txt", 400, [])
    print("max_pairs", max_pairs)
    # INSERT_YOUR_CODE
    # Print the first 10 items of res[0]
    print("vocab", vocab)
    # print(all_matches)

    # res_bytes = tuple(bytes([c]) for c in "hello".encode("utf-8"))
    # print("res_bytes", res_bytes)
    # print(merge_at_i(res_bytes, 2))
