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


def get_pairs(pre_tok_dic: dict[tuple[bytes], int]):
    cow: dict[tuple[bytes], tuple[int, list[tuple[tuple[bytes], int, int]]]] = {}
    # for every pre_token.
    for pre_token_key, value in pre_tok_dic.items():
        # for every successive_pair:
        for start in range(0, len(pre_token_key), 2):
            if start + 1 >= len(pre_token_key):
                continue
            # set new_key to be the successive_pair of bytes
            new_key = tuple([pre_token_key[start], pre_token_key[start + 1]])
            # increment successive_pair of bytes by the frequency of words where they appear.
            before_count, lst = cow.get(new_key, (0, []))
            before_count += value
            lst = lst + [(pre_token_key, start)]
            cow[new_key] = (before_count, lst)
    return cow


def merge_at_i(bytes: tuple[bytes], i: int):
    return bytes[0:i] + (bytes[i] + bytes[i + 1],) + bytes[i + 2 :]


def update_dic(pre_tok_dic: dict[tuple[bytes], int], max_pair: tuple[bytes]):
    # for every pre_token.
    new_pre_tok_dic = {}
    for pre_tok, value in pre_tok_dic.items():
        marks = []
        new_pre_tok = pre_tok
        for start in range(0, len(pre_tok), 2):
            if start + 1 >= len(pre_tok):
                continue
            pair = tuple([pre_tok[start], pre_tok[start + 1]])
            if pair == max_pair:
                marks.append(start)
        for mark in marks:
            new_pre_tok = merge_at_i(pre_tok, mark)
        new_pre_tok_dic[new_pre_tok] = new_pre_tok_dic.get(new_pre_tok, 0) + value

    return new_pre_tok_dic


# not optimized for now.
def merge(pre_tok_dic: dict[tuple[bytes], int], stopping_condition: int):
    # for each successive pair we assoc the count and a index of
    cow = get_pairs(pre_tok_dic)

    max_pairs: list[tuple[bytes]] = []
    while len(max_pairs) < stopping_condition:
        max_pair = max(cow, key=lambda k: cow[k][0])
        max_pairs.append(max_pair)
        # again for every pre_token
        pre_tok_dic = update_dic(pre_tok_dic, max_pair)
        cow = get_pairs(pre_tok_dic)
        # for key, value in pre_tok_dic.items():
        #     # for every successive_pair:
        #     for i in range(0, len(key), 2):
        #         if i + 1 >= len(key):
        #             continue
        #         # set new_key to be the successive_pair of bytes
        #         new_key = tuple([key[i], key[i + 1]])
        #         if new_key == max_pair_key:
        #             # everything before i in the dic
        #             new_pre_token_key = key[0:i] + (key[i] + key[i + 1],) + key[i + 2 :]
        #             print("old", key)
        #             print("new", new_pre_token_key)
        #             # remove all key and assign its value to the new key
        #             pre_tok_dic[new_pre_token_key] = pre_tok_dic.pop(key)

        #         # increment successive_pair of bytes by the frequency of words where they appear.
        #         cow[new_key] = cow.get(new_key, 0) + value

        # for pre_token_key, start in lst:
        #     # index of max-pair occurence.
        #     pre_token_count = dic[pre_token_key]

        #     # merged the pair in the pre_token.
        #     # LIES LIES don't merge trust me don't merge.
        #     # new_pre_token_key = pre_token_key[0:i] + (pre_token_key[i] + pre_token_key[i + 1],) + pre_token_key[i + 2 :]
        #     # dic[new_pre_token_key] = dic.pop(pre_token_key)
        #     # process before pair
        #     if start - 1 >= 0:
        #         # fuck what if before there is something that was merged?
        #         before_pair = (pre_token_key[start - 1], max_pair_key[1])
        #         cow.pop(before_pair, None)
        #         new_before_pair_key = tuple([pre_token_key[start - 1], together])
        #         before_count, new_before_pair_key_idx = cow.get(new_before_pair_key, (0, []))
        #         before_count += pre_token_count
        #         new_before_pair_key_idx = new_before_pair_key_idx + [(pre_token_key, start - 1)]
        #         cow[new_before_pair_key] = (before_count, new_before_pair_key_idx)
        #     # process after pair
        #     if start + 1 < len(new_pre_token_key):
        #         after_pair = tuple([pre_token_key[start + 1], pre_token_key[start + 2]])
        #         cow.pop(after_pair, None)
        #         new_after_pair_key = tuple([new_pre_token_key[start], new_pre_token_key[start + 1]])
        #         after_count, new_after_pair_key_idx = cow.get(new_after_pair_key, (0, []))
        #         after_count += pre_token_count
        #         new_after_pair_key_idx = new_after_pair_key_idx + [(new_pre_token_key, start)]
        #         cow[new_after_pair_key] = (after_count, new_after_pair_key_idx)
        # cow.pop(max_pair_key)

    # return [max_pair_key, cow[max_pair_key]]

    return cow


# 1 2 3 4 5
# 1 2 3 4
# (1,2) then 3 (3,4) jump to
# its the odd case, because at 5 I will try to access 6!


# 1 the jump to 3  then jump to 5 then over
def train_bpe(input_path: str, vocab_size: int, special_tokens: list[str]):
    stopping_condition = vocab_size - 256 - len(special_tokens)
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

        return merge(combined, stopping_condition)


# suppose the best pair was 'a' 't'
# then '*' 'a' is messed up i.e. now 0?
# clearly 'a' 't' is messed up
# clearly 'a' '*' is also messed up. i.e. now 0?
# from the pre tokenization we have just single bytes. so even for something like 'k' 'a' 't'.
# do I need to find that?


if __name__ == "__main__":
    all_matches = train_bpe("./cs336_basics/TinyStories-valid.txt", 1000, [])
    sample_keys = list(all_matches.keys())[:1]
    for k in sample_keys:
        print(k, all_matches[k])
    # print(all_matches)

    # res_bytes = tuple(bytes([c]) for c in "hello".encode("utf-8"))
    # print("res_bytes", res_bytes)
    # print(merge_at_i(res_bytes, 2))
