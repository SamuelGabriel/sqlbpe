import torch
from io import open
import unicodedata
import string
import re
import random
import json
import sqlparse
from collections import defaultdict
from typing import Optional, Callable
from sqlbpe.sqlast import SqlAst

import torchtext
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import spacy
from torchtext.datasets import TranslationDataset, Multi30k
from functools import reduce

from sqlbpe.bpe import BPE

def get_device(no_cuda=False):
    return torch.device("cuda" if torch.cuda.is_available() and (not no_cuda) else "cpu")

spacy_en = spacy.load('en')
spacy_de = spacy.load('de')

######################################################################
# The files are all in Unicode, to simplify we will turn Unicode
# characters to ASCII, trim most
# punctuation.
#
# trim, and remove non-letter characters
def get_string_normalizer(reverse):
    def normalizeString(s):
        s = s.strip().split(' ')
        if reverse:
            s = list(reversed(s))
        return s
    return normalizeString

def get_sql_normalizer(bpe_encoder: Optional[BPE], ast_sql: bool, bpe_step_override_p_schedule:Callable=lambda _: 0.):
    def normalizeSql(s, epoch=1):
        ss = s.split(' ')
        if bpe_encoder is None:
            return ss
        if ast_sql:
            sa = SqlAst(s)
            restrictions = [sa.get_all_possible_neighbor_pairs(join_char='__')]
        else:
            restrictions = None
        return bpe_encoder.transform([ss], vocab_back_map=True, vocab_map=True, restrictions=restrictions, step_override_p=bpe_step_override_p_schedule(epoch))[0]
    return normalizeSql

######################################################################
# To read the data file we will split the file into lines, and then split
# lines into pairs. The files are all English → Other Language, so if we
# want to translate from Other Language → English I added the ``reverse``
# flag to reverse the pairs.
#

def find_dynamic_vocab(pair, src_field, trg_field):
    s,t = pair
    s,t = src_field.preprocess(s), trg_field.preprocess(t)
    mini_vocab = {}
    def transform(sentence):
        trans = []
        for w in sentence:
            if w not in mini_vocab:
                mini_vocab[w] = len(mini_vocab)
            trans.append(mini_vocab[w])
        return trans
    ts, tt = transform(s), transform(t)
    return [ts, tt]

def to_target_dynamic_vocab(batch: torchtext.data.Batch, src_to_trg: torch.Tensor, trg_vocab: torchtext.vocab.Vocab):
    '''
    Example with fields `src`, `trg`, `dyn_voc_src` and `dyn_voc_trg` is converted into a src-trg pair with all <unk>
    replaced with dynamic vocab. All special tokens are assumed to be -100 in the dynamic vocab fields. These dynamic vocabs only work per pair and not for multiple pairs at once.
    Returns:
        (dyn_src_in_trg_rep: Tensor[src len, batch size], dyn_trg: Tensor[trg len, batch size]), dyn_vocab_size: Tensor[batch size]
    '''
    o_inputs = src_to_trg[batch.src[0].flatten()].view(batch.src[0].size())
    o_outputs = batch.trg[0]
    unk_i = trg_vocab.stoi['<unk>']
    unks = o_inputs == unk_i
    trg_unks = o_outputs == unk_i
    voc_size = len(trg_vocab.itos)
    num_unks = torch.sum(unks, dim=0)
    dyn_voc_lens = torch.zeros(o_inputs.size()[1], dtype=torch.long)
    for i in range(o_inputs.size()[1]):
        fill_in_tokens = batch.dyn_voc_src[:, i][unks[:, i]]

        fill_in_tokens_itos = list(set(fill_in_tokens.cpu().flatten().numpy()) - {-100})
        if fill_in_tokens_itos:
            fill_in_tokens_stoi = torch.ones(max(fill_in_tokens_itos)+1, dtype=torch.long).to(o_inputs.device) * trg_vocab.stoi['<unk>']
            fill_in_tokens_stoi[fill_in_tokens_itos] = torch.arange(len(fill_in_tokens_itos), dtype=torch.long).to(o_inputs.device) + voc_size

            dyn_voc_lens[i] = len(fill_in_tokens_itos)
            o_inputs[:, i][unks[:, i]] = fill_in_tokens_stoi[fill_in_tokens]

            trg_fill_in_tokens = batch.dyn_voc_trg[:, i][trg_unks[:, i]]
            trg_fill_in_tokens[trg_fill_in_tokens >= len(fill_in_tokens_stoi)] = unk_i
            trg_fill_in_tokens[trg_fill_in_tokens < len(fill_in_tokens_stoi)] =  fill_in_tokens_stoi[trg_fill_in_tokens[trg_fill_in_tokens < len(fill_in_tokens_stoi)]]

            o_outputs[:, i][trg_unks[:, i]] = trg_fill_in_tokens
    return (o_inputs, o_outputs), dyn_voc_lens

def get_example_copy_mask(example, trg_vocab):
    """Returns positions which could be copied in the target list as a list of bools.
    """
    ignore_set = {'(', ')', ',', '.', '/', ';', 'A', 'S'}
    src_set = set(example.dyn_voc_src)
    return [dw in src_set and w not in ignore_set for w, dw in zip(example.trg, example.dyn_voc_trg)]


class RandomBPEDataset(torchtext.data.Dataset):
    def __init__(self, *args, bpe_encoder, ast_bpe, bpe_step_override_p_schedule, **kwargs):
        super().__init__(*args, **kwargs)
        self.bpe_encoder = bpe_encoder
        self.sql_normalizer = get_sql_normalizer(bpe_encoder, ast_bpe, bpe_step_override_p_schedule=bpe_step_override_p_schedule)
        self.fields_list = [('src', self.fields['src']), ('trg', self.fields['trg']), ('dyn_voc_src', self.fields['dyn_voc_src']), ('dyn_voc_trg', self.fields['dyn_voc_trg'])]
        self.access_count = 0
    
    def get_accessed_epochs(self)->float: 
        return self.access_count/len(self)
    
    def transform_example(self, example: torchtext.data.Example):
        src = example.src
        trg = ' '.join(example.trg)
        trg = self.sql_normalizer(trg, epoch=self.get_accessed_epochs())
        return [src, trg]

    def __iter__(self):
        for e in self.examples:
            e = self.transform_example(e)
            yield torchtext.data.Example.fromlist(e + find_dynamic_vocab(e, self.fields['src'], self.fields['trg']), self.fields_list)
    
    def __getitem__(self, i):
        self.access_count += 1
        e = self.transform_example(self.examples[i])
        return torchtext.data.Example.fromlist(e + find_dynamic_vocab(e, self.fields['src'], self.fields['trg']), self.fields_list)
    
def get_pairs(data_path):
    return [line.split(' ||| ') for line in open(data_path, 'r').read().splitlines()]

def make_dataset_with_fields(pairs, src_field, trg_field):
    dyn_voc_src = torchtext.data.Field(use_vocab=False,init_token=-100, eos_token=-100, pad_token=-100)
    dyn_voc_trg = torchtext.data.Field(use_vocab=False,init_token=-100, eos_token=-100, pad_token=-100)
    fields = [('src', src_field), ('trg', trg_field), ('dyn_voc_src', dyn_voc_src), ('dyn_voc_trg', dyn_voc_trg)]
    return torchtext.data.Dataset(
                [torchtext.data.Example.fromlist(e + find_dynamic_vocab(e, src_field, trg_field), fields) for e in pairs],
                fields
            )

def get_data_src_trg_vocab(pairs, min_count, reverse_input, bpe_encoder, ast_bpe):
    SRC = torchtext.data.Field(tokenize=get_string_normalizer(reverse_input), init_token='<sos>', eos_token='<eos>', include_lengths=True)
    TRG = torchtext.data.Field(tokenize=get_sql_normalizer(bpe_encoder, ast_bpe), init_token='<sos>', eos_token='<eos>', include_lengths=True)

    data = make_dataset_with_fields(pairs, SRC, TRG)
    SRC.build_vocab(data, min_freq=min_count)
    TRG.build_vocab(data, min_freq=min_count)
    print('Vocab sizes, src:', len(SRC.vocab), 'trg:', len(TRG.vocab))
    return data, SRC, TRG

def get_random_bpe_data_src_trg_vocab(pairs, min_count, reverse_input, bpe_encoder, ast_bpe, bpe_step_override_p_schedule):
    SRC = torchtext.data.Field(tokenize=get_string_normalizer(reverse_input), init_token='<sos>', eos_token='<eos>', include_lengths=True)
    TRG = torchtext.data.Field(tokenize=get_sql_normalizer(None, False), init_token='<sos>', eos_token='<eos>', include_lengths=True)
    dyn_voc_src = torchtext.data.Field(use_vocab=False,init_token=-100, eos_token=-100, pad_token=-100)
    dyn_voc_trg = torchtext.data.Field(use_vocab=False,init_token=-100, eos_token=-100, pad_token=-100)
    fields = [('src', SRC), ('trg', TRG), ('dyn_voc_src', dyn_voc_src), ('dyn_voc_trg', dyn_voc_trg)]
    data = RandomBPEDataset(
                [torchtext.data.Example.fromlist(e + [[],[]], fields) for e in pairs],
                fields,
                bpe_encoder=bpe_encoder,
                ast_bpe=ast_bpe,
                bpe_step_override_p_schedule=bpe_step_override_p_schedule
            )
    SRC.build_vocab(data, min_freq=min_count)
    TRG.build_vocab([[w] for w in list(bpe_encoder.vocab.str_itos.values())+bpe_encoder.base_vocab.itos], min_freq=1)
    print('Vocab sizes, src:', len(SRC.vocab), 'trg:', len(TRG.vocab))
    return data, SRC, TRG

def get_datasets(min_count=1, data_path='../data/datasets/nl_to_sql/advising', reverse_input=False, bpe_encoding_steps=0, bpe_copy_masking=False, auto_bpe_min_freq=1, ast_bpe=False, bpe_step_override_p_schedule=lambda x:0.0):
    train_pairs = get_pairs(data_path+'.train')
    val_pairs = get_pairs(data_path+'.dev')
    test_pairs = get_pairs(data_path+'.test')
    if bpe_encoding_steps != 0:
        pre_data, pre_SRC, pre_TRG = get_data_src_trg_vocab(train_pairs, 1, reverse_input, None, False)
        pre_val_data = make_dataset_with_fields(val_pairs, pre_SRC, pre_TRG)

        if ast_bpe:
            sql_asts = [SqlAst(target) for _, target in train_pairs]
            restrict_bpe_pairs_to = [sa.get_all_possible_neighbor_pairs(join_char='__') for sa in sql_asts]
            val_sql_asts = [SqlAst(target) for _, target in val_pairs]
            val_restrict_bpe_pairs_to = [sa.get_all_possible_neighbor_pairs(join_char='__') for sa in val_sql_asts]
            restriction_map = {'train': restrict_bpe_pairs_to, 'val': val_restrict_bpe_pairs_to}
        else:
            restriction_map = {'train': None, 'val': None}
        copy_mask = [get_example_copy_mask(e, pre_TRG.vocab) for e in pre_data] if bpe_copy_masking else None
        bpe_encoder = BPE(pre_TRG.vocab, restriction_map)

        if bpe_encoding_steps > 0:
            bpe_encoder.fit([e.trg for e in pre_data], t=bpe_encoding_steps, vocab_map=True, ignore_mask=copy_mask)
        else:
            bpe_encoder.fit_to_valid([e.trg for e in pre_data], [e.trg for e in pre_val_data], retention_count=-bpe_encoding_steps, vocab_map=True, ignore_mask=copy_mask, min_freq=auto_bpe_min_freq)
    else:
        bpe_encoder = None
    if bpe_step_override_p_schedule(100) == 0.:
        train_data, SRC, TRG = get_data_src_trg_vocab(train_pairs, min_count, reverse_input, bpe_encoder, ast_bpe)
    else:
        print("A probablistic dataset is used.")
        train_data, SRC, TRG = get_random_bpe_data_src_trg_vocab(train_pairs, min_count, reverse_input, bpe_encoder, ast_bpe,bpe_step_override_p_schedule)

    val_data = make_dataset_with_fields(val_pairs, SRC, TRG)
    test_data = make_dataset_with_fields(test_pairs, SRC, TRG)

    return (train_data, val_data, test_data), (SRC, TRG)

def get_translation_dataset():
    def tokenize_de(text):
        """
        Tokenizes German text from a string into a list of strings
        """
        return [tok.text for tok in spacy_de.tokenizer(text)]

    def tokenize_en(text):
        """
        Tokenizes English text from a string into a list of strings
        """
        return [tok.text for tok in spacy_en.tokenizer(text)]
    SRC = torchtext.data.Field(tokenize=tokenize_de, init_token='<sos>', eos_token='<eos>', include_lengths=True)
    TRG = torchtext.data.Field(tokenize=tokenize_en, init_token='<sos>', eos_token='<eos>', include_lengths=True)
    train_data, valid_data, test_data = Multi30k.splits(exts=('.de', '.en'), fields=(SRC, TRG), root='../data/datasets/multi30')
    SRC.build_vocab(train_data, min_freq=2)
    TRG.build_vocab(train_data, min_freq=2)
    return train_data, valid_data, test_data, SRC, TRG

def batch_to_list_of_sentences(b, itos, max_len=20, max_count=1, transpose=True):
    max_len = min(b.size(0), max_len)
    e = b[:max_len+1, :max_count]
    if transpose:
        e = torch.t(e)
    init = [[itos[t] if t < len(itos) else str(t) for t in d] for d in e]
    return ['|'.join(i) for i in init]

def sequence_mask(seq_lens, max_len):
    return torch.arange(max_len, device=seq_lens.device).unsqueeze(1).repeat(1, len(seq_lens)) < seq_lens

def line_tensor_to_list(t, eos_idx):
    r = [x.item() for x in t]
    e = r.index(eos_idx)
    return r[1:e]

def flatten_bpe_encodings(list_of_ints, trg_vocab):
    return sum([trg_vocab.itos[i].split('__') if i < len(trg_vocab.itos) else [i] for i in list_of_ints], [])