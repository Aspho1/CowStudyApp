from random import shuffle, sample


def manual_chunking(cow_ids, n_chunks):
    # 1) Shuffle a copy (so we don't clobber the original)
    cows = cow_ids[:]       
    shuffle(cows)

    # 2) Compute how many go in each chunk
    k, r = divmod(len(cows), n_chunks)
    #    first 'r' chunks get size k+1, the rest get k

    chunks = []
    idx = 0
    for i in range(n_chunks):
        size = k + 1 if i < r else k
        chunks.append(cows[idx:idx + size])
        idx += size
    shuffle(chunks)
    return chunks
cow_ids = [1006, 1008, 1015, 1017, 1021, 1022, 1028, 1030, 824, 826, 827, 828, 830, 831, 832, 837, 838, 988, 993, 996, 998, 999]

chunks = manual_chunking(cow_ids=cow_ids, n_chunks=22)

print(chunks)

