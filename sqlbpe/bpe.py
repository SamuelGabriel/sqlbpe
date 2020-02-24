from collections import Counter
import copy
from itertools import repeat
import random
import numpy as np

class BPEVocab():
    def __init__(self, min_idx=0, join_str='__'):
        self.next_idx=min_idx
        self.join_str = join_str
        self.itos = {}
        self.stoi = {}
        self.str_itos = {}
    def add(self, w):
        i = self.next_idx
        self.next_idx += 1
        w = tuple(w)
        self.itos[i] = w
        self.stoi[w] = i
        return i
    def get_str_itos(self, i, untransform_function):
        if i not in self.str_itos:
            self.str_itos[i] = self.join_str.join(untransform_function([[i]])[0])
        return self.str_itos[i]
    def built_stringified_vocab(self, untransform_function):
        self.str_itos = {i: self.join_str.join(l) for i, l in zip(self.itos.keys(), untransform_function([[i] for i in self.itos.keys()]))}
        self.str_stoi = {w: i for i, w in self.itos.items()}
        assert sorted(self.str_itos.keys()) == sorted(self.itos.keys())

class BPE():
    def __init__(self, vocab, map_of_dataset_restrictions={'train': None, 'val': None}):
        assert bool(map_of_dataset_restrictions['train']) == bool(map_of_dataset_restrictions['val'])
        self.base_vocab = vocab
        self.vocab = BPEVocab(min_idx=len(vocab.itos))
        self.map_of_dataset_restrictions = map_of_dataset_restrictions

    def fit_to_valid(self, listofseqs, val_listofseqs, vocab_map=False, retention_count=1, ignore_mask=None, min_freq=1):
        # ignore mask should be true for tokens to be ignored
        assert retention_count > 0
        assert ignore_mask is None or len(ignore_mask) == len(listofseqs) and all(len(s) == len(sm) for s, sm in zip(listofseqs, ignore_mask))
        if vocab_map:
            idxtrg = self.vocab_map(listofseqs)
            val_idxtrg = self.vocab_map(val_listofseqs)
        else:
            idxtrg = listofseqs
            val_idxtrg = val_listofseqs

        prev_unk_count = self.unk_count(idxtrg, val_idxtrg, min_freq)
        print(prev_unk_count)
        steps = 0
        breaks = 0
        bigrams_to_ignore = set()
        while True:
            bg = self.get_most_common_bigram(idxtrg, ignore_mask=ignore_mask, ignore_bigrams=bigrams_to_ignore)
            if bg is None:
                print('no more bigrams')
                break
            curr_unk_count = self.test_unk_count(idxtrg, val_idxtrg, bg, min_freq)
            out_of_vocab = prev_unk_count != curr_unk_count
            breaks += out_of_vocab
            if out_of_vocab:
                print('bigram', self.vocab.get_str_itos(bg[0], self.untransform), 'and', self.vocab.get_str_itos(bg[1], self.untransform), 'not in valset')

            if breaks >= retention_count:
                print('vocab difference increased to ', self.listofseqs2set(val_idxtrg, 1) - self.listofseqs2set(idxtrg, min_freq), 'with', self.vocab.get_str_itos(bg[0], self.untransform), 'and', self.vocab.get_str_itos(bg[1], self.untransform))
                break

            if not out_of_vocab:
                i = self.vocab.add(bg)
                idxtrg, ignore_mask = self.transform_step(idxtrg, bg, i, ignore_mask=ignore_mask)
                val_idxtrg, _ = self.transform_step(val_idxtrg, bg, i, dataset='val')

                prev_unk_count = curr_unk_count

                steps += 1
            else:
                bigrams_to_ignore.add(bg)
        print('BPE took', steps, 'restricted steps.')
        self.vocab.built_stringified_vocab(self.untransform)
        return steps
    
    @classmethod
    def unk_count(cls, idxtrg, val_idxtrg, min_freq):
        return len(cls.listofseqs2set(val_idxtrg, 1) - cls.listofseqs2set(idxtrg, min_freq))
    
    def test_unk_count(self, idxtrg, val_idxtrg, bg, min_freq):
        i = self.vocab.next_idx
        test_idxtrg, _ = self.transform_step(idxtrg, bg, i)
        test_val_idxtrg, _ = self.transform_step(val_idxtrg, bg, i, dataset='val')
        return self.unk_count(test_idxtrg, test_val_idxtrg, min_freq)
        
    @staticmethod
    def listofseqs2set(listofseqs, min_freq=1):
        token_freqs = Counter(t for s in listofseqs for t in s)
        return {t for t, c in token_freqs.items() if c >= min_freq}

    def vocab_map(self, listofseqsoftokens):
        return [[self.base_vocab.stoi[w] for w in s] for s in listofseqsoftokens]

    def fit(self, listofseqs, t=100, vocab_map=False, ignore_mask=None):
        assert ignore_mask is None or len(ignore_mask) == len(listofseqs) and all(len(s) == len(sm) for s, sm in zip(listofseqs, ignore_mask))
        if vocab_map:
            idxtrgs = self.vocab_map(listofseqs)
        else:
            idxtrgs = listofseqs
        for s in range(t):
            bg = self.get_most_common_bigram(idxtrgs, ignore_mask=ignore_mask)
            if bg is None:
                print('Stopped BPE early at step {}'.format(s))
                t = s
                break
            i = self.vocab.add(bg)
            idxtrgs, ignore_mask = self.transform_step(idxtrgs, bg, i, ignore_mask=ignore_mask)
        self.vocab.built_stringified_vocab(self.untransform)
        return t

    def compute_score(self, Ec, vocab_size, count_entropy, ca, cb, cab):
        Ec, vocab_size, count_entropy, ca, cb, cab, bca, bcb = float(Ec), float(vocab_size), float(count_entropy), float(ca), float(cb), float(cab), float(ca-cab), float(cb-cab)
        bEc = Ec - cab
        f = cab / bEc / Ec
        term = f * count_entropy - f * (ca * np.log2(ca) + cb * np.log2(cb)) + np.log2(Ec / bEc) + np.log2((vocab_size + 1) / vocab_size) \
            - (ca * np.log2(ca)) / bEc - (cb * np.log2(cb)) / bEc + (cab * np.log2(cab)) / bEc
        etb = lambda: bcb * np.log2(bcb) / bEc
        eta = lambda: bca * np.log2(bca) / bEc
        return term + (etb() if bcb != 0. else 0.) + (eta() if bca != 0. else 0.)

    def fit_kl(self, listofseqs, vocab_map=False, min_freq=0):
        if vocab_map:
            idxtrgs = self.vocab_map(listofseqs)
        else:
            idxtrgs = listofseqs
        s = 0
        def compute_score(Ec, vocab_size, count_entropy, ca, cb, cab):
            Ec, vocab_size, count_entropy, ca, cb, cab, bca, bcb = float(Ec), float(vocab_size), float(count_entropy), \
                                                                   float(ca), float(cb), float(cab), float(ca - cab), float(cb - cab)
            bEc = Ec - cab
            bca = ca - cab
            bcb = cb - cab
            if bca < min_freq or bcb < min_freq or cab < min_freq:
                return 1
            new_vocab_size = vocab_size + 1.
            log = np.log2
            new_terms = cab*log(cab) + (bca*log(bca) if bca != 0. else 0.) + (bcb*log(bcb) if bcb != 0. else 0.)
            return (count_entropy - ca*log(ca) - cb*log(cb) + new_terms)/bEc - count_entropy/Ec + log(new_vocab_size/vocab_size) + log(Ec/bEc)

        while True:
            bgs = self.get_most_common_bigram(idxtrgs, return_all=True)
            counts = Counter(idx for l in idxtrgs for idx in l)
            Ec = sum(counts.values())
            count_entropy = sum(float(c)*np.log2(c) for c in counts.values())
            scores = {bg: compute_score(Ec,self.vocab.next_idx, count_entropy, counts[bg[0]], counts[bg[1]], c) for bg, c in bgs.items()}
            min_bg = min(scores, key=scores.get)
            min_score = scores[min_bg]
            s+=1
            if min_score >= 0:
                break
            else:
                i = self.vocab.add(min_bg)
                idxtrgs, _ = self.transform_step(idxtrgs, min_bg, i, ignore_mask=None)
                print('added {} with score {} and count {}'.format(min_bg, min_score, bgs[min_bg]))

        self.vocab.built_stringified_vocab(self.untransform)
        return s


    

    def transform_step(self, idxtrgs, bg, idx, ignore_mask=None, dataset='train', restrictions=None):
        """
        
        Arguments:
            idxtrgs {List[List[int]]} -- 
            bg {Tuple[int, int]} -- 
            idx {int} --
        
        Keyword Arguments:
            ignore_mask {List[List[bool]]} -- (default: {None})
            dataset {str} -- which dataset to work on in {'train', 'val', 'test'} (default: {'train'})
        
        Returns:
            [type] -- [description]
        """
        ignore_mask = ignore_mask or [[False for _ in l] for l in idxtrgs]
        result = []
        ignore_mask_result = []

        for s, ms, restrict in zip(idxtrgs, ignore_mask, restrictions or (self.map_of_dataset_restrictions[dataset] or repeat(None))):
            r = []
            ir = []
            for i, (t,m) in enumerate(zip(s, ms)):
                if s[i-1:i+1] == list(bg) and (restrict is None or ((self.vocab.get_str_itos(s[i-1], self.untransform), self.vocab.get_str_itos(s[i], self.untransform)) in restrict)):
                    r[-1] = idx
                    ir[-1] = False
                    # This can happen: assert not ir[-1] and not m, 'Problem with {} {} at {}'.format(s, ms, i)
                else:
                    r.append(t)
                    ir.append(m)
            result.append(r)
            ignore_mask_result.append(ir)
        return result, ignore_mask_result

    def untransform_step(self, idxtrgs, bg, idx):
        result = []
        for s in idxtrgs:
            r = []
            for t in s:
                if t == idx:
                    r.extend(list(bg))
                else:
                    r.append(t)
            result.append(r)
        return result

    def transform(self, listofseqs, vocab_map=True, vocab_back_map=False, restrictions=None, step_override_p=0.):
        if isinstance(listofseqs[0][0], int) and vocab_map:
            print(Warning('You might want to change `transform`s `vocab_map` argument to False.'))
        if vocab_map:
            idxtrgs = [[self.base_vocab.stoi[w] for w in s] for s in listofseqs]
        else:
            idxtrgs = listofseqs
        for i,bg in self.vocab.itos.items():
            if random.random() >= step_override_p:
                idxtrgs, _ = self.transform_step(idxtrgs, list(bg), i, restrictions=restrictions)
        if vocab_back_map:
            return [[self.vocab.str_itos[i] if i in self.vocab.str_itos else self.base_vocab.itos[i] for i in s] for s in idxtrgs]
        return idxtrgs

    def untransform(self, idxtrgs, vocab_map=True):
        for i in sorted(self.vocab.itos.keys(), reverse=True):
            bg = self.vocab.itos[i]
            idxtrgs = self.untransform_step(idxtrgs, list(bg), i)
        if vocab_map:
            return [[self.base_vocab.itos[w] for w in s] for s in idxtrgs]
        return idxtrgs

    def get_most_common_bigram(self, idxtrgs, ignore_mask=None, ignore_bigrams=set(), return_all=False):
        assert ignore_mask is None or len(ignore_mask) == len(idxtrgs) and all(len(s) == len(sm) for s, sm in zip(idxtrgs, ignore_mask))

        bg = []
        for s, m, r in zip(idxtrgs, ignore_mask or repeat(None), self.map_of_dataset_restrictions['train'] or repeat(None)):
            for i in range(len(s)-1):
                if m is not None and (m[i] or m[i+1]):
                    continue
                if r is not None and \
                   ((self.vocab.get_str_itos(s[i], self.untransform), self.vocab.get_str_itos(s[i+1], self.untransform)) not in r):
                    continue
                if (s[i],s[i+1]) in ignore_bigrams:
                    continue
                bg.append((s[i],s[i+1]))
        if not bg:
            return None
        if return_all:
            return Counter(bg)
        return Counter(bg).most_common(1)[0][0]

if __name__ == '__main__':
    def sample_vocab(ls):
        class v():
            itos = {}
            stoi = {}

        vocab = v()
        vocab.itos = ls
        vocab.stoi = {w: i for i, w in enumerate(ls)}
        return vocab
    vocab = sample_vocab(['0', '1', '2', '3', '4'])
    enc = BPE(vocab)
    idxtrgs = [[1, 2, 3, 4], [3, 4]]
    enc.fit_kl(idxtrgs)
    print(enc.vocab.str_itos)
        