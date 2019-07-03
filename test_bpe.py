"""Tests for bpe.py.

One of the tests uses a separate process. 
The others mock out multiprocessing functionality, so should be lightweight.
"""
import multiprocessing
from collections import Counter
from unittest import mock

import pytest

import bpe


def test_get_pairs():
    s = 'aaabbb'
    pairs = list(bpe.get_pairs(s))
    assert len(pairs) == len(s) - 1
    assert list(bpe.get_pairs('abc')) == [('a', 'b'), ('b', 'c')]

def test_str_to_byte_list():
    assert [b'a',b'b'] == bpe.str_to_byte_list('ab')

def test_merge():
    l = bpe.str_to_byte_list('abc')
    to_merge = {x: b''.join(x) for x in [(b'a',b'b')]}
    merged = list(bpe.merge(to_merge, l))
    assert merged == [b'ab', b'c']

def test_compute_vocab_simple():
    TEST = 'The quick brown fox jumped over the lazy dog. Wow! Amazing.'
    vocab = bpe.compute_vocab(TEST)
    assert 41 == len(vocab)

def test_encode():
    encoder = bpe.Encoder(bpe.str_to_byte_list('abcdef'))
    test_str = 'aabb'
    encoded = list(encoder.encode(test_str))
    assert encoded == [0, 0, 1, 1]
    assert test_str == encoder.decode(encoded)
    # Test UNK
    assert encoder.UNK.decode('utf8') == encoder.decode(encoder.encode('t'))


@mock.patch('multiprocessing.Queue')
@mock.patch('bpe.Worker')
@mock.patch('multiprocessing.Condition')
def test_compute_vocab_multi(Condition, MockWorker, Queue):
    q = Queue.return_value

    q.get.side_effect = [
        # Return initial vocab.
        {b'a'}, 
        # Return the first set of counts.
        {(b'a', b'a'): 3, (b'b', b'b'): 2}]
    out = bpe.compute_vocab_multi('aaabb', n=1, max_vocab_size=2)
    assert 'aaabb' in MockWorker.call_args[0]
    assert out == {b'aa', b'a'}


@mock.patch('multiprocessing.Queue')
@mock.patch('bpe.Worker')
@mock.patch('multiprocessing.Condition')
def test_compute_vocab_multi_corpus_partition(Condition, MockWorker, Queue):
    # Get the instance
    q = Queue.return_value
    # Return the same thing every time.
    q.get.return_value = []
    out = bpe.compute_vocab_multi('aaabb', n=2, max_vocab_size=0)
    assert 'aaa' in MockWorker.call_args_list[0][0]
    assert 'abb' in MockWorker.call_args_list[1][0]
    # Queue returned nothing every time.
    assert out == set()

def test_worker():
    top_k_ready = multiprocessing.Condition()
    with multiprocessing.Manager() as m:
        top_k = m.dict()
        count_q = multiprocessing.Queue()
        vocab_q = multiprocessing.Queue()
        worker = bpe.Worker(top_k_ready, top_k, count_q, vocab_q, 'aaabb', 100, 2)
        worker.start()
        assert vocab_q.get() == {b'a', b'b'}
        counts = count_q.get()
        assert counts == {(b'a', b'a'): 2, (b'a', b'b'): 1, (b'b', b'b'): 1}
        top_k.update({x[0]: b''.join(x[0]) for x in Counter(counts).most_common(2)})
        with top_k_ready:
            top_k_ready.notify()
        # Round 2.
        counts = count_q.get()
        assert counts == {(b'aa', b'ab'): 1, (b'ab', b'b'): 1}
        # Finish.
        top_k.clear()
        with top_k_ready:
            top_k_ready.notify()
        worker.join()
        assert not worker.is_alive()

if __name__ == '__main__':
    pytest.main()
