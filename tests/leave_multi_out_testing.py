import pytest
from cowstudyapp.analysis.RNN.run_lstm import manual_chunking

def test_chunking_balanced_distribution():
    """Test if manual_chunking creates balanced chunks of cows"""
    # Setup
    cow_ids = list(range(30))
    n_chunks = 22

    # Execute
    chunks = manual_chunking(cow_ids=cow_ids, n_chunks=n_chunks)

    # Verify
    # Check correct number of chunks
    assert len(chunks) == n_chunks

    # Check that chunks have proper distribution
    # With 30 cows and 22 chunks, we should have:
    # - 8 chunks with 2 cows (30 - 22 = 8 extra cows)
    # - 14 chunks with 1 cow (22 - 8 = 14 remaining chunks)
    chunks_with_multiple = sum(1 for c in chunks if len(c) > 1)
    assert chunks_with_multiple == (30 - 22)

    # Check that chunk sizes are balanced (difference ≤ 1)
    chunk_lens = [len(c) for c in chunks]
    assert max(chunk_lens) - min(chunk_lens) <= 1

    # Check that all cows are included (no duplicates, no missing)
    all_cows = [cow for chunk in chunks for cow in chunk]
    assert sorted(all_cows) == cow_ids

def test_chunking_edge_cases():
    """Test edge cases for manual_chunking"""
    # Case 1: Equal number of cows and chunks
    cow_ids = list(range(10))
    chunks = manual_chunking(cow_ids=cow_ids, n_chunks=10)
    assert len(chunks) == 10
    assert all(len(chunk) == 1 for chunk in chunks)

    # Case 2: More chunks than cows (should raise an error or handle gracefully)
    with pytest.raises(ValueError):
        manual_chunking(cow_ids=list(range(5)), n_chunks=10)
