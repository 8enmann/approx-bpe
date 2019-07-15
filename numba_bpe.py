import numba
from numba import njit, jit
import random
from numba import types
from numba.typed import Dict, List

def random_str(n: int):
    """Returns a random str of length n."""
    a = []
    for i in range(n):
        a.append(random.randint(1, 128))
    return bytes(a).decode('utf8', errors='ignore')

def fake_vocab():
    vocab = List()
    for i in range(128):
        vocab.append(bytes((i, i+1)).decode('utf8', errors='ignore'))
    return vocab

@njit
def numba_encode(corpus: str, vocab:List):
    d = Dict()
    for i,k in enumerate(vocab):
        d[k] = i
    for i in range(len(corpus) - 1):
        chunk = corpus[i:i+1]
        if chunk in d:
            yield d[chunk]

def encode(corpus, vocab):
    d = {}
    for i,k in enumerate(vocab):
        d[k] = i
    for i in range(len(corpus) - 1):
        chunk = corpus[i:i+1]
        if chunk in d:
            yield d[chunk]
