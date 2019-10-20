import torch
from torch import nn
from encoder import Encoder

from rolloutdecoder import RollOutDecoder
import utils
from sql_check import QuerySyntaxChecker

class Seq2Seq(nn.Module):
    '''This Seq2Seq model has the most important functions for standard seq2seq, attention and copying.
    '''

    def __init__(self, encoder, decoder, src_vocab, trg_vocab, copying, copy_forcing=False):
        super(Seq2Seq, self).__init__()
        self.encoder = encoder #type: Encoder
        self.decoder = decoder #type: RollOutDecoder
        self.pad_idx = trg_vocab.stoi['<pad>']
        self.copying = copying
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad_idx)
        self.copy_forcing = copy_forcing
        self.copy_criterion = nn.CrossEntropyLoss(ignore_index=-100)
        self.src_vocab = src_vocab
        self.trg_vocab = trg_vocab
        self.query_syntax_checker = QuerySyntaxChecker(trg_vocab)
        
        self.src_to_trg = nn.Parameter(torch.Tensor([self.trg_vocab.stoi[w] if w in self.trg_vocab.stoi else self.trg_vocab.stoi['<unk>'] for w in self.src_vocab.itos]).long(), requires_grad=False)
    
    def generate_matchings(self, src_vocab, trg_vocab):
        s_t = torch.Tensor([trg_vocab.stoi[w] if w in trg_vocab.stoi else -1 for w in src_vocab.itos])
        return s_t
    
    def generate_mask(self, inputs, input_lens):
        r = [torch.arange(i.size()[0], device=i.device).repeat(len(il), 1) < il.unsqueeze(1) for i, il in zip(inputs, input_lens)]
        return r
    
    def correct_copy_positions(self, inputs: list, output: torch.Tensor, trg_pad_idx: torch.Tensor):
        input_inputs = self.encoder.src_inputs
        copy_targets = torch.zeros_like(output).repeat(len(inputs), 1, 1)
        src_to_trg = self.src_to_trg.to(output.device)
        for i, (is_in, inp) in enumerate(zip(input_inputs, inputs)):
            if is_in:
                flat_in = inp.flatten()
                flat_in = src_to_trg[flat_in]
                inp = flat_in.view(inp.size())
            inp = inp.permute(1,0)
            maxs, max_ind = (output.unsqueeze(-1) == inp).max(dim=-1)
            max_ind[~maxs] = -100
            max_ind[output == trg_pad_idx] = -100
            copy_targets[i] = max_ind
        return copy_targets
    
    def generate_copy_target(self, inputs: list, output: torch.Tensor, copy_space: int, trg_pad_idx: torch.Tensor):
        correct_copies = self.correct_copy_positions(inputs, output, trg_pad_idx)
        spaced_correct_copies = torch.where(correct_copies != -100, correct_copies+(torch.arange(len(inputs), device=output.device)*copy_space).unsqueeze(-1).unsqueeze(-1), correct_copies)
        _, ind = (correct_copies != -100).max(0)
        # TODO chosse different index randomly
        return torch.gather(spaced_correct_copies,0,ind.unsqueeze(0)).squeeze(0)
    
    def loss(self, recon_x, x, copy_inputs=None, p_gen=None):
        # the [1:] is since the first token always is the beginning token <sos>
        # works only for copy forcing with one input
        recon_x = recon_x[1:]
        x = x[1:]
        inp = copy_inputs[0][1:]

        f_recon_x = recon_x.view(-1, recon_x.shape[2])
        f_x = x.view(-1)
        if self.copying:
            not_padding = f_x != self.pad_idx
            generation_loss = -torch.log(f_recon_x[torch.arange(len(f_x)).to(recon_x.device), f_x])[not_padding].mean()
            if self.copy_forcing:
                copyable = torch.any(inp.permute(1,0).unsqueeze(1) == x.permute(1,0).unsqueeze(2), dim=2).permute(1,0).flatten()
                p_gen = p_gen.flatten()
                copy_loss = -torch.log(torch.cat((p_gen[copyable & not_padding], 1 - p_gen[~copyable & not_padding]))).mean()
                return generation_loss + copy_loss * 0.1

            return generation_loss
        else:
            f_x = torch.where(f_x < f_recon_x.shape[1], f_x, torch.ones_like(f_x)*self.trg_vocab.stoi['<unk>'])
            return self.criterion(f_recon_x, f_x)
    
    def generate_copy_inp_out(self, batch):
        '''
        Returns (copy_input, new_output), dyn_vocab_lens
        '''
        return utils.to_target_dynamic_vocab(batch, self.src_to_trg, self.trg_vocab)

    def forward(self, inputs: list, input_lens: list, output: torch.Tensor, inference: bool=False, beam_size: int=0, copy_inputs: list=[], beam_syntax_check=False):
        """The forward function for the editor, takes the retrieved example (input and output), as well as the input for this question as inputs. It returns a loss.
        
        Arguments:
            inputs {list} -- A list of tensors of shape [src sent len, batch size]
            input_lens {list} -- A list of tensors of shape [batch size]
            output {Optional[torch.Tensor]} -- shape [trg sent len, batch size]
        """
        assert all([i.shape[1] == len(il) for i, il in zip(inputs, input_lens)])
        assert beam_size <= 0 or inference # beam size > 0 => inference
        encoder_outputs, agenda = self.encoder(inputs, input_lens)
        # from now on decoder takes inputs encoded with its vocabulary + extra tokens
        if beam_size and inference:
            seqs, probs = self.decoder.beam_search(agenda, encoder_outputs=encoder_outputs, att_masks=self.generate_mask(inputs, input_lens), input_lens=input_lens, inputs=copy_inputs, inference_max_len=output.shape[0]+10, beam_size=beam_size)
            assert len(agenda) == len(seqs)

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
        else:
            outputs, predictions, attentions, p_gen = self.decoder(agenda, None if inference else output, encoder_outputs=encoder_outputs, att_masks=self.generate_mask(inputs, input_lens), input_lens=input_lens, inputs=copy_inputs, inference_max_len=output.shape[0])
            return outputs, predictions, attentions, p_gen
        

        

        
# TODO add general get_seq2seq method