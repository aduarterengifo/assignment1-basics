import itertools
import os
import regex as re
from typing import BinaryIO
import multiprocessing as mp
import cProfile
import pstats
from collections import Counter
import heapq


PAT = r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
PAT_RE = re.compile(PAT)


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


def pre_tokenize_chunk(input_path: str, start: int, end: int, pattern: re.Pattern[str]):
    with open(input_path, "rb") as f:
        f.seek(start)
        chunk = f.read(end - start).decode("utf-8", errors="ignore")
        sub_chunks = pattern.split(chunk)
        # run tokenization for each of the sub_chunks.
        iterators = [PAT_RE.finditer(sub_chunk) for sub_chunk in sub_chunks]
        flattened_iterators = itertools.chain(*iterators)
        store = Counter()
        for match in flattened_iterators:
            res = match.group()
            res_bytes = tuple(bytes([c]) for c in res.encode("utf-8"))
            store[res_bytes] += 1
        return store


def get_all_simple_pairs(
    pre_tok_dic: Counter[tuple[bytes], int],
) -> tuple[dict[tuple[bytes, bytes], int], dict[tuple[bytes, bytes], set[tuple[bytes]]]]:
    pair_to_count = Counter()
    pair_to_tokens = {}
    for pre_token_key, count in pre_tok_dic.items():
        if len(pre_token_key) < 2:
            continue
        for i in range(len(pre_token_key) - 1):
            pair = (pre_token_key[i], pre_token_key[i + 1])
            pair_to_count[pair] += count
            if pair not in pair_to_tokens:
                pair_to_tokens[pair] = set()
            pair_to_tokens[pair].add(pre_token_key)
    return (pair_to_count, pair_to_tokens)


# not optimized for now.
def merge(pre_tok_dic: Counter[tuple[bytes], int], stopping_condition: int):
    max_pairs: list[tuple[bytes, bytes]] = []
    (pair_to_count, pair_to_tokens) = get_all_simple_pairs(pre_tok_dic)

    # build heap from pair_to_count, to efficiently find max_pair
    heap = [(-count, pair) for pair, count in pair_to_count.items()]
    heapq.heapify(heap)

    while len(max_pairs) < stopping_condition and heap:
        neg_count, max_pair = heapq.heappop(heap)
        if max_pair not in pair_to_count or -neg_count != pair_to_count[max_pair]:
            continue
        # max_pair = max(pair_to_count, key=lambda pair: (pair_to_count[pair], pair))
        # max_pair = alt_max_pair
        max_pairs.append(max_pair)

        # INSERT_YOUR_CODE
        # Print the pair at the top of the heap (heap max) and the pair with the max count (max max)
        # print("heap max:", alt_max_pair, -neg_count, "max max:", max_pair, pair_to_count[max_pair])

        # ------------ EFFICIENTLY UPDATE ------------
        affected_pre_toks = set(pair_to_tokens[max_pair])
        for pre_tok in affected_pre_toks:
            count = pre_tok_dic[pre_tok]
            for i in range(len(pre_tok) - 1):
                # calculate pair
                pair = (pre_tok[i], pre_tok[i + 1])
                # decrement count from pair
                pair_to_count[pair] -= count
                # ------ HEAP MAINTENANCE ------
                heapq.heappush(heap, (-pair_to_count[pair], pair))
                # ------ HEAP MAINTENANCE ------
                # remove token from pairs_to_tokens
                pair_to_tokens[pair].discard(pre_tok)
                # if pair doesn't occur anymore remove all together.
                if pair_to_count[pair] == 0:
                    del pair_to_count[pair]
                    del pair_to_tokens[pair]
            # construct new merged_token.
            new_pre_tok = []
            i = 0
            while i < len(pre_tok):
                if i < len(pre_tok) - 1 and (pre_tok[i], pre_tok[i + 1]) == max_pair:
                    new_pre_tok.append(pre_tok[i] + pre_tok[i + 1])
                    i += 2
                else:
                    new_pre_tok.append(pre_tok[i])
                    i += 1
            new_pre_tok = tuple(new_pre_tok)

            # update dic.
            pre_tok_dic[new_pre_tok] += count
            pre_tok_dic[pre_tok] -= count

            # remove all together if appropriate
            if pre_tok_dic[pre_tok] == 0:
                del pre_tok_dic[pre_tok]

            for i in range(len(new_pre_tok) - 1):
                pair = (new_pre_tok[i], new_pre_tok[i + 1])
                pair_to_count[pair] += count
                # ------ HEAP MAINTENANCE ------
                heapq.heappush(heap, (-pair_to_count[pair], pair))
                # ------ HEAP MAINTENANCE ------
                if pair not in pair_to_tokens:
                    pair_to_tokens[pair] = set()
                pair_to_tokens[pair].add(new_pre_tok)

        # ------------ EFFICIENTLY UPDATE ------------
    return max_pairs


def train_bpe(
    input_path: str, vocab_size: int, special_tokens: list[str]
) -> tuple[dict[int, bytes], list[tuple[bytes, bytes]]]:
    special_tokens_len = len(special_tokens)
    stopping_condition = vocab_size - 256 - special_tokens_len
    pattern = re.compile("|".join([re.escape(tok) for tok in special_tokens]))
    # open the input path
    with open(input_path, "rb") as f:
        # find the chunk boundaries
        boundaries = find_chunk_boundaries(f, 16, "<|endoftext|>".encode("utf-8"))
        num_workers = min(mp.cpu_count(), len(boundaries) - 1)
        with mp.Pool(processes=num_workers) as pool:
            results = pool.starmap(
                pre_tokenize_chunk,
                [(input_path, start, end, pattern) for start, end in zip(boundaries[:-1], boundaries[1:])],
            )
        # results = [pre_tokenize_chunk(chunk) for chunk in chunks]
        combined = Counter()

        for d in results:
            combined.update(d)

        max_pairs = merge(combined, stopping_condition)
        vocab: dict[int, bytes] = {}
        # single-bytes
        for i in range(0, 256):
            vocab[i] = bytes([i])
        # special_tokens
        for i, token in enumerate(special_tokens):
            vocab[i + 256] = token.encode("utf-8")

        # vocab
        for i, (a, b) in enumerate(max_pairs):
            vocab[i + 256 + special_tokens_len] = a + b
        return (vocab, max_pairs)


if __name__ == "__main__":
    profiler = cProfile.Profile()
    profiler.enable()
    (vocab, max_pairs) = train_bpe("./tests/fixtures/tinystories_sample_5M.txt", 400, ["<|endoftext|>"])
    print("max_pairs", max_pairs)
    profiler.disable()
    stats = pstats.Stats(profiler).sort_stats("cumtime")
    stats.print_stats(20)  # Show top 20 slowest functions
