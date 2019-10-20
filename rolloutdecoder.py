import numpy as np
import torch
from torch import nn
from torch.nn import functional as F
import random
from typing import Optional, List, Tuple

class RollOutDecoder(nn.Module):
    def __init__(self, teacher_forcing_ratio, decoder, target_vocab, device, with_attention=False, extra_copy_vocab=0):
        super().__init__()
        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.decoder = decoder
        self.device = device
        self.eos_idx, self.sos_idx, self.unk_idx, self.pad_idx = target_vocab.stoi['<eos>'], target_vocab.stoi['<sos>'], target_vocab.stoi['<unk>'], target_vocab.stoi['<pad>']
        self.with_attention = with_attention
        self.extra_copy_vocab = extra_copy_vocab

    
    def forward(self, latent_rep: torch.Tensor, target: Optional[torch.Tensor], encoder_outputs: Optional[list]=None, att_masks: Optional[list]=None, input_lens: Optional[list]=None, inputs: Optional[list]=None, inference_max_len: Optional[int]=None):
        if target is None:
            inference = True
            max_len = inference_max_len or 100
        else:
            inference = False
            max_len = target.shape[0]
        device = latent_rep.device
        batch_size = latent_rep.shape[0]
        outputs = torch.zeros(max_len, batch_size, self.decoder.output_dim+self.extra_copy_vocab).to(self.device)
        predictions = torch.ones(max_len, batch_size).long().fill_(self.sos_idx).to(self.device)
        p_gens = torch.zeros(max_len, batch_size).to(self.device)
        # batch of <sos> tokens
        output = torch.ones([batch_size], dtype=torch.long).fill_(self.sos_idx).to(self.device)
        hidden = None
        if self.with_attention:
            assert encoder_outputs
            attentions = [torch.zeros(max_len, batch_size, eo.shape[0]).to(self.device) for eo in encoder_outputs]
        for t in range(1, max_len):
            teacher_force = (not inference) and (random.random() < self.teacher_forcing_ratio)
            top1, _, hidden, output, attention, p_gen = self.decoder_step(output, hidden, inputs, input_lens, latent_rep, att_masks, encoder_outputs)
            for i,a in enumerate(attention):
                attentions[i][t] = a
            outputs[t] = output
            predictions[t] = top1
            if p_gen is not None:
                p_gens[t] = p_gen.squeeze(1)
            output = (target[t] if teacher_force else top1)
        if inference:
            _, first_eos_outs = ((predictions == self.eos_idx).type(torch.long) * torch.arange(max_len).to(device).unsqueeze(1)).min(0)
            predictions[torch.arange(max_len).to(device).unsqueeze(1) > first_eos_outs] = self.pad_idx
        if self.with_attention:    
            return outputs, predictions, attentions, p_gens[1:] # first p_gen is unused, attentions are logit attentions
        return outputs, predictions
    
    def beam_search(self, latent_rep: torch.Tensor, encoder_outputs: Optional[list]=None, att_masks: Optional[list]=None, input_lens: Optional[list]=None, inputs: Optional[list]=None, inference_max_len:int=200, beam_size:int=10, max_batch_size:int=32, length_normalization:bool=True):
        # output is batch first and has all probs and sequences for every item in the batch
        batch_size = len(latent_rep)
        encoder_outputs = torch.stack(encoder_outputs, 0) if encoder_outputs else torch.zeros(3, 1, batch_size, 1).to(self.device)
        att_masks = torch.stack(att_masks, 0) if att_masks else torch.zeros(3, batch_size, 1).to(self.device)
        input_lens = torch.stack(input_lens, 0) if input_lens else torch.zeros(3, batch_size).to(self.device)
        inputs = torch.stack(inputs, 0) if inputs else torch.zeros(3,1,batch_size).to(self.device)
        out_idxs: List[list] = []
        out_losses: List[list] = []
        for lr, eo, am, il, i in zip(latent_rep.unsqueeze(1), encoder_outputs.permute(2,0,1,3).unsqueeze(3), att_masks.permute(1,0,2).unsqueeze(2), input_lens.permute(1,0).unsqueeze(2), inputs.permute(2,0,1).unsqueeze(3)):
            idxs, loss = self._beam_search(lr, eo, am, il, i, inference_max_len, beam_size, max_batch_size, length_normalization)
            out_idxs.append(idxs)
            out_losses.append(loss)
        return out_idxs, out_losses

    def _beam_search(self, latent_rep: torch.Tensor, encoder_outputs: Optional[torch.Tensor]=None, att_masks: Optional[torch.Tensor]=None, input_lens: Optional[torch.Tensor]=None, inputs: Optional[torch.Tensor]=None, inference_max_len:int=200, beam_size:int=10, max_batch_size:int=32, length_normalization:bool=True):
        '''This function performs beam-search and expects a batch-size of 1. `max_batch_size` only specifies the maximum batch size that should be used inside this function.
        
        Arguments:
            latent_rep {torch.Tensor} -- [description]
        
        Keyword Arguments:
            encoder_outputs {Optional[list]} -- [description] (default: {None})
            att_masks {Optional[list]} -- [description] (default: {None})
            input_lens {Optional[list]} -- [description] (default: {None})
            inputs {Optional[list]} -- [description] (default: {None})
            inference_max_len {Optional[int]} -- [description] (default: {None})
        '''
        assert len(latent_rep) == 1
        def get_step_inputs(dec_inputs, hiddens):
            batch_size = len(dec_inputs)
            dec_input = torch.cat([d.view(1) for d in dec_inputs], 0)
            if hiddens is None:
                hidden = None
            elif isinstance(hiddens[0], tuple):
                hidden = (torch.cat([h[0] for h in hiddens], 1), torch.cat([h[1] for h in hiddens], 1))
            else:
                hidden = torch.cat(hiddens, 1)
            input_lens_b = input_lens.expand(-1, batch_size)
            latent_rep_b = latent_rep.expand(batch_size, -1)
            inputs_b = inputs.expand(-1,-1,batch_size)
            # input_lens_b and latent_rep_b are both 0!?
            return dec_input, hidden, inputs_b, input_lens_b, latent_rep_b, att_masks.expand(-1,batch_size,-1), encoder_outputs.expand(-1,-1,batch_size,-1)

        idx_sequences: List[Tuple[List[int], float, Tuple[torch.Tensor, Optional[torch.Tensor]]]] = [([], 0.0, (torch.ones([1], dtype=torch.long).fill_(self.sos_idx).to(self.device), self.decoder.get_initial_hidden(latent_rep)))]
        # elements of `idx_sequences` have following form [<seq of preds so far>, <log prob sum so far>, (last output / next input, hidden)]
        done = []
        for t in range(inference_max_len):
            all_candidates = []
            for b_i in range(0, len(idx_sequences), max_batch_size):
                idx_seq_batch = idx_sequences[b_i:b_i+max_batch_size]
                dec_inputs, hiddens = zip(*[s[2] for s in idx_seq_batch])
                _, _, hidden, gen_output, _, _ = self.decoder_step(*get_step_inputs(dec_inputs, hiddens))
                log_probs = F.log_softmax(gen_output, 1)
                topk_log_probs, topks = log_probs.topk(beam_size, 1)
                def iterate_hidden(hidden):
                    if isinstance(hidden,tuple):
                        h1, h2 = hidden
                        h1 = h1.permute(1,0,2)
                        h2 = h2.permute(1,0,2)
                        for h_1, h_2 in zip(h1,h2):
                            yield (h_1.unsqueeze(1), h_2.unsqueeze(1))
                    else:
                        for h in hidden.permute(1,0,2):
                            yield h.unsqueeze(1)
                for topk, topk_log_prob, idx_seq, h in zip(topks, topk_log_probs, idx_seq_batch, iterate_hidden(hidden)):
                    for idx, log_prob in zip(topk, topk_log_prob):
                        if idx == self.eos_idx:
                            done.append(idx_seq)
                        else:
                            all_candidates.append((idx_seq[0]+[idx.item()],
                                idx_seq[1]+log_prob.item(),
                                (idx, h)))
            ordered = sorted(all_candidates, key=lambda x: x[1], reverse=True)
            idx_sequences = ordered[:beam_size]
        idx_sequences = sorted((e for e in idx_sequences + done if e[0]), key=lambda x: x[1]/len(x[0]) if length_normalization else x[1], reverse=True)
        return [e[0] for e in idx_sequences], [e[1] for e in idx_sequences]
    
    def decoder_step(self, dec_input, hidden, inputs, input_lens, latent_rep, att_masks, encoder_outputs):
        if self.decoder.copy_mode:
            output, hidden, attention, p_gen = self.decoder(dec_input, hidden, latent_rep, mask=att_masks, encoder_outputs=encoder_outputs, input_lens=input_lens if input_lens is not None and self.decoder.copy_mode else None)
            output, gen_output, _ = self.compute_copy_output(output, attention, p_gen, inputs, input_lens)
        elif self.with_attention:
            gen_output, hidden, attention = self.decoder(dec_input, hidden, latent_rep, mask=att_masks, encoder_outputs=encoder_outputs, input_lens=input_lens if input_lens is not None and self.decoder.copy_mode else None)
            p_gen = None
        else:
            gen_output, hidden = self.decoder(dec_input, hidden, latent_rep, mask=att_masks, encoder_outputs=encoder_outputs)
            attention = []
            p_gen = None
        top1 = gen_output.max(1)[1]
        return top1, gen_output, hidden, gen_output, attention, p_gen

    def compute_copy_output(self, output, attention, p_gen, inputs, input_lens):
        assert inputs is not None
        if isinstance(input_lens, list):
            input_lens_v = torch.stack(input_lens)
        else:
            input_lens_v = input_lens
        if isinstance(inputs, list):
            inputs_v = torch.stack(inputs)
        else:
            inputs_v = inputs
        inp_max_len = inputs_v.size()[1]
        batch_size = len(output)
        gen_probs = F.softmax(output, dim=1)
        att_dist = F.softmax(torch.stack(attention, dim=0).permute(1,0,2).contiguous().view(batch_size, -1), dim=1)
        # distribution (sums to 1 for each batch) over [batch size, num inputs * src len]
        copy_source = inputs_v.permute(2,0,1).view(batch_size, -1).long()
        # copy_source = [batch size, num inputs * src len], of long
        copy_probs = torch.zeros(batch_size, (self.decoder.output_dim+self.extra_copy_vocab)).to(output.device)
        copy_probs.scatter_add_(1, copy_source, att_dist)

        padded_gen_probs = torch.cat((gen_probs, torch.zeros(batch_size, self.extra_copy_vocab).to(output.device)), dim=1)
        mixed_probs = padded_gen_probs*p_gen + copy_probs*(1-p_gen)
        return torch.cat((copy_probs, gen_probs), 1), mixed_probs, copy_probs