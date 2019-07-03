"""Implements an approximate BPE encoding over bytes with some tricks for efficiency.

https://arxiv.org/pdf/1508.07909.pdf section 3.2.

Basic algorithm from the paper:
    Initialize the vocab with the character vocabulary
    Each word is a sequence of characters plus an enod of word symbol '·'
    Count all symbol pairs
    Replace each occurence of the most frequent pair ('a', 'b') with 'ab'.
    Each merge represents a character n-gram
    Frequent n-grams are merged into a single symbol.
    Repeat until max vocab size or computation budget is reached.

Unlike the paper, this implementation operates directly on utf-8 bytes, 
so should work for any language or data type with no modification.

It provides the option to do multiple replacements per iteration for increased speed.
Encoding using a computed vocab is done greedily instead of by the standard algorithm.
TODO: benchmark against original.
"""

import logging
import multiprocessing as mp
import time
import fire
from collections import Counter, deque
from typing import Dict, Iterable, List, Set, Tuple

import tqdm


def get_pairs(seq: Iterable) -> Iterable[Tuple]:
    """Yield a sliding window of length 2 from seq."""
    d = deque(maxlen=2)
    # Consume first bit
    it = iter(seq)
    for _ in range(2):
        d.append(next(it))
    yield tuple(d)
    for i in it:
        d.append(i)
        yield tuple(d)


class Worker(mp.Process):
    """Computes counts on a subset of the corpus.

    Waits for the master to tell it what to merge based on its siblings.
    Queues are child -> parent only.
    `top_k` is read only.
    """
    def __init__(
            self, 
            top_k_ready: mp.Condition,
            top_k_merges: 'DictProxy',
            count_q: mp.Queue,
            vocab_q: mp.Queue, 
            corpus: str,
            max_merges: int,
            top_k: int):
        super(Worker, self).__init__()
        self.top_k_ready = top_k_ready
        self.top_k_merges = top_k_merges
        self.vocab_q = vocab_q
        self.count_q = count_q
        self.corpus = corpus
        self.top_k = top_k
        self.byte_list: Iterable[bytes] = None

    def run(self):
        """This shouldn't be called directly; call `worker.start()`."""
        logging.info(f'started {self.name}')
        self.byte_list = str_to_byte_list(self.corpus)
        self.vocab = set(self.byte_list)
        self.vocab_q.put(self.vocab)
        while True:
            counts = Counter(get_pairs(self.byte_list))
            # TODO: only put top_k * factor
            self.count_q.put(dict(counts.most_common(self.top_k * 2)))
            # self.count_q.put(counts)
            # Wait for main thread to send top k merges
            with self.top_k_ready:
                self.top_k_ready.wait()
            if len(self.top_k_merges) == 0:
                break
            self.byte_list = list(merge(self.top_k_merges, self.byte_list))

def compute_vocab_multi(
    corpus: str,
    max_vocab_size:int=3000, 
    max_merges:int=10, top_k=1, 
    n:int=mp.cpu_count()) -> Set[bytes]:
    """Multiprocess implementation of approximate BPE.
    
    Divides the corpus among n workers.

    Args:
        corpus: The corpus to encode. Could scale better by taking a list of filenames.
        max_vocab_size: Stop after generating this many vocab entries.
        max_merges: Stop after this many rounds.
        top_k: Each round merge the top k pairs. Standard BPE sets top_k=1.
    Returns:
        A set of all the vocab entries generated, each of which is a `bytes`.
    """
    top_k_ready = mp.Condition()
    vocab_q = mp.Queue()
    count_q = mp.Queue()
    chunk_size = len(corpus) // n
    vocab = set()
    # TODO: handle interrupt
    with mp.Manager() as manager:
        to_merge = manager.dict()

        procs = []
        logging.info('starting workers')
        for i in range(n):
            procs.append(Worker(
                top_k_ready,
                to_merge,
                count_q,
                vocab_q,
                # These overlap on purpose
                corpus[i * chunk_size:(i+1) * chunk_size + 1],
                max_merges,
                top_k
            ))
            procs[-1].start()

        # Get inital vocab from each worker.
        logging.info('waiting for vocab from worker')
        for _ in range(n):
            vocab.update(vocab_q.get())
        logging.debug(f'got vocab {vocab}')
        bar = tqdm.tqdm(total=max_vocab_size, )
        for i in range(max_merges):
            # Get counts from each worker.
            counts = Counter()
            for _ in range(n):
                counts.update(count_q.get())
            logging.debug(str(counts))
            to_merge.clear()
            to_merge.update({x[0]: b''.join(x[0]) for x in counts.most_common(top_k)})
            vocab.update(to_merge.values())
            with top_k_ready:
                top_k_ready.notify_all()
            bar.n = len(vocab)
            bar.refresh()
            if len(vocab) >= max_vocab_size:
                break
        bar.close()
        # Tell workers to stop.
        to_merge.clear()
        with top_k_ready:
            top_k_ready.notify_all()
        for p in procs:
            p.join(1)
    return vocab


def merge(to_merge: Dict[Tuple[bytes], bytes], seq: Iterable) -> Iterable:
    """Given a set of requested merges, go through the sequence and do the merges."""
    to_merge = {x: b''.join(x) for x in to_merge.keys()}
    just_merged = False
    for pair in get_pairs(seq):
        if just_merged:
            just_merged = False
            continue
        if pair in to_merge:
            just_merged = True
            yield to_merge[pair]
        else:
            yield pair[0]
    if not just_merged:
        yield pair[1]



def str_to_byte_list(s: str) -> Iterable[bytes]:
    return [bytes([x]) for x in s.encode('utf8')]

def compute_vocab(corpus: str, max_vocab_size:int=3000, max_merges:int=10, top_k=1) -> Set[bytes]:
    """Single threaded implementation of approximate BPE.
    
    Args:
        corpus: The corpus to encode. Could scale better by taking a list of filenames.
        max_vocab_size: Stop after generating this many vocab entries.
        max_merges: Stop after this many rounds.
        top_k: Each round merge the top k pairs. Standard BPE sets top_k=1.
    Returns:
        A set of all the vocab entries generated, each of which is a `bytes`.
    """
    if len(corpus) < min(max_merges, max_vocab_size):
        raise Exception('Corpus must be bigger than max_merges')
    l = str_to_byte_list(corpus)
    vocab = set(l)
    bar = tqdm.tqdm(total=max_vocab_size)
    for i in range(max_merges):
        counts = Counter(get_pairs(l))
        # Merge the most common.
        to_merge = {x[0]: b''.join(x[0]) for x in counts.most_common(top_k)}
        vocab.update(to_merge.values())
        l = list(merge(to_merge, l))
        bar.n = len(vocab)
        bar.refresh()
        if len(vocab) >= max_vocab_size:
            break
    bar.close()
    return vocab

class Encoder:
    DEFAULT_VOCAB_FILENAME = 'vocab.bpe'
    # Null bytes unlikely to occur in natural encoded text.
    DELIM = b'\0\n'
    # Must be 2 characters long because otherwise probably won't have intermediate merges for the greedy encoder to pick up.
    EOF = b'\0F'
    UNK = b'\0UNK'

    def __init__(self, vocab: Iterable[bytes]=None, vocab_file: str=DEFAULT_VOCAB_FILENAME):
        if vocab:
            self.vocab = vocab
        else:
            self.vocab = self.load(vocab_file)
        # Append special characters.
        self.vocab += [self.EOF, self.UNK]
        # break keys into tuples for faster match?
        self.encoder = {x:i for i,x in enumerate(vocab)}
        self.decoder = {i:x for i,x in enumerate(vocab)}
        self.max_length = max(map(len, self.vocab))
        self.UNK_EMB = len(self.vocab) - 1

    def encode(self, corpus: str) -> Iterable[int]:
        """Greedily encode `corpus` according to the vocab."""
        b = corpus.encode('utf8')
        start = 0
        while start < len(b):
            match = self.UNK_EMB
            for end in range(0, self.max_length):
                end += 1 + start
                substr = b[start:end]
                new_match = self.encoder.get(substr)
                if new_match is not None:
                    match = new_match
                    if end < len(b):
                        continue
                yield match
                start += max(1, len(substr) - 1)
                break

    def decode(self, corpus: Iterable[int], errors='ignore') -> str:
        """Decode `corpus` according to the vocab."""
        return b''.join([self.decoder[x] for x in corpus]).decode('utf8', errors=errors)

    @classmethod    
    def save(cls, vocab: Iterable[bytes], filename:str=DEFAULT_VOCAB_FILENAME):
        with open(filename, 'wb') as f:
            for v in vocab:
                f.write(v + cls.DELIM)
    
    @classmethod
    def load(cls, filename:str=DEFAULT_VOCAB_FILENAME):
        with open(filename, 'rb') as f:
            return f.read().split(cls.DELIM)[:-1]

class Timer:
    def __enter__(self):
        self.t = time.time()
    def __exit__(self, *args):
        logging.info(f'Elapsed time: {time.time() - self.t:.02f}s')

def main(
    multi=True, 
    vocab_filename='vocab_multi.bpe', 
    level='INFO', 
    max_vocab_size=30000,
    top_k=100,
    max_merges=int(1e9)):
    logging.basicConfig(level=level.upper())
    CORPUS_FILE = '/Users/ben/data/wikitext-2/wiki.train.tokens'
    with open(CORPUS_FILE) as f:
        corpus = f.read()
    logging.info(f'Loaded {len(corpus)} chars, building vocab')
    with Timer():
        if multi:
            mp.log_to_stderr()
            vocab = compute_vocab_multi(corpus, max_merges=max_merges, top_k=top_k, max_vocab_size=max_vocab_size)
        else:
            vocab = compute_vocab(corpus, max_merges=max_merges, top_k=top_k, max_vocab_size=max_vocab_size)

        logging.info('Vocab size %d', len(vocab))
    # Save the mapping
    Encoder.save(vocab, filename=vocab_filename)
    logging.info(f'Encoder vocab size: {len(Encoder.load(filename=vocab_filename))}')



if __name__ == '__main__':
    fire.Fire(main)
