import time
import fire
from bpe import Encoder
from pytorch_pretrained_bert import GPT2Tokenizer
import numba_bpe

class Timer:
    def __enter__(self):
        self.t = time.time()
    def __exit__(self, *args):
        print(f'Elapsed: {time.time() - self.t:.2f}s')

def main(mode:str='baseline', max_length:int=None):
    enc = GPT2Tokenizer.from_pretrained('gpt2')
    CORPUS_FILE = '/Users/ben/data/wikitext-2/wiki.train.tokens'
    with open(CORPUS_FILE) as f:
        corpus = f.read()
    if max_length:
        corpus = corpus[:max_length]

    # Reprocess vocab as real bytes
    vocab = [bytes(enc.byte_decoder[c] for c in token) for token in enc.encoder]
    encoder = dict(zip(vocab, range(len(vocab))))
    greedy = Encoder(vocab)

    with Timer():
        if mode == 'baseline':
            out = enc.encode(corpus)
        elif mode == 'greedy':
            out = list(greedy.encode(corpus))
        elif mode == 'greedy-c':
            pass
        elif mode == 'numba':
            out = list(numba_bpe.numba_encode(numba_bpe.random_str(100000), numba_bpe.fake_vocab()))
        elif mode == 'nonumba':
            out = list(numba_bpe.encode(numba_bpe.random_str(100000), numba_bpe.fake_vocab()))
        else:
            raise Exception('Uknown mode %s'.format(mode))
        print(f'Compression ratio {len(out)/len(corpus):.4f}')

if __name__ == '__main__':
    fire.Fire(main)