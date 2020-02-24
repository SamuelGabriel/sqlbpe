import torch
from torch import nn
from torch.nn import functional as F
import utils
import random
from sql_check import QuerySyntaxChecker

class SimpleSeq2Seq(torch.nn.Module):
    def __init__(self, h_dim, trg_vocab, encoder, decoder, activation=nn.ReLU, only_encode=False):
        """
        VAE initializer
        :param h_dim: dimension of the hidden layers
        :param z_dim: dimension of the latent representation
        :param activation: callable activation function
        """
        super(SimpleSeq2Seq, self).__init__()
        
        self.h_dim, self.activation, self.only_encode = h_dim, activation, only_encode
        
        self.encoder = encoder
        
            
        self.decoder = decoder if decoder else \
            nn.Sequential(
                nn.Linear(h_dim, h_dim * 2),
                activation(),
                nn.Linear(h_dim * 2, 784),
                nn.Sigmoid()
            )
        self.query_syntax_checker = QuerySyntaxChecker(trg_vocab)

    def encode(self, x):
        x = self.encoder(*x)
        return x
        
    def forward(self, x, beam_size=0, beam_syntax_check=False): 
        x, x_len, y = x
        z = self.encode(([x], [x_len]))
        if self.only_encode:
            return None, z
        if self.training or y is None and beam_size == 0:
            x_, preds = self.decoder(z, y) # passing x is needed for teacher forcing
        elif y is None:
            seqs, probs = self.decoder.beam_search(z, beam_size=beam_size)
            return [s[0] for s in seqs], [-p[0] for p in probs]
        elif beam_size == 0:
            x_, preds = self.decoder(z, None, inference_max_len=len(y))
        else:
            seqs, probs = self.decoder.beam_search(z, inference_max_len=len(y)+10, beam_size=beam_size)
            assert len(z) == len(seqs)
            if beam_syntax_check:
                res_seqs = []
                res_ps = []
                for sps in zip(seqs, probs):
                    for i, (s, p) in enumerate(zip(*sps)):
                        if self.query_syntax_checker.is_syntactical(s):
                            res_seqs.append(s)
                            res_ps.append(-p)
                            print('SCP {} SCP'.format(i))
                            break
                    else:
                        res_seqs.append(sps[0][0])
                        res_ps.append(sps[1][0])
                        print('SCP {} SCP'.format(-1))
                return res_seqs, res_ps
            return [s[0] for s in seqs], [-p[0] for p in probs]
        return x_, preds ,z