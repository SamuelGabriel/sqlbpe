import torch
from torch import nn
from torch.nn import functional as F
from typing import Optional, Tuple
from enum import Enum

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, unk_i, attentions=[], num_layers=4, copy_mode=False, embedding=None, rnn_type=nn.GRU):
        super().__init__()

        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.output_dim = output_dim
        self.dropout = dropout
        self.attentions = nn.ModuleList(attentions)
        self.num_att = len(attentions)
        self.rnn_input_dim = (enc_hid_dim * 2) * self.num_att + emb_dim if attentions else emb_dim
        self.num_layers = num_layers
        self.copy_mode = copy_mode
        self.rnn_type = rnn_type
        self.unk_i = unk_i
        
        self.embedding = embedding or nn.Embedding(output_dim, emb_dim)
        
        self.rnn = rnn_type(self.rnn_input_dim, self.dec_hid_dim, num_layers=self.num_layers, dropout=dropout)

        self.rnn_init_hidden = nn.Parameter(torch.rand(self.num_layers, self.dec_hid_dim))
        self.rnn_init_cell = nn.Parameter(torch.rand(self.num_layers, self.dec_hid_dim))
        

        self.out = nn.Linear(((enc_hid_dim*2) * self.num_att + emb_dim if self.attentions else self.emb_dim) + self.dec_hid_dim, output_dim)
        if self.copy_mode:
            self.p_gen = nn.Sequential(
                nn.Linear((self.enc_hid_dim*2) * self.num_att + self.dec_hid_dim, 1),
                nn.Sigmoid()
            )
        
        self.dropout = nn.Dropout(dropout)
    
    def get_initial_hidden(self, encoder_agenda):
        def grow_init_state_for_batch(init_state, batch_size):
            return init_state.unsqueeze(1).repeat(1,batch_size,1)
        if self.rnn_type == nn.GRU:
            return encoder_agenda.repeat(self.num_layers, 1, 1)
        elif self.rnn_type == nn.LSTM:
            return (encoder_agenda.repeat(self.num_layers, 1, 1), grow_init_state_for_batch(self.rnn_init_cell, len(encoder_agenda)))
        else:
            raise ValueError('`rnn_type` unsupported.')
        
    def forward(self, input, hidden, encoder_agenda, encoder_outputs=None, mask=None, input_lens: Optional[Tuple[list]]=None):

        batch_size = input.size()[0]

        if hidden is None:
            hidden = self.get_initial_hidden(encoder_agenda)
             
        #input = [batch size]
        #hidden = [n layers, batch size, dec hid dim]
        #encoder_outputs = Tuple([src sent len, batch size, enc hid dim * 2])
        #encoder_agenda = [batch size, dec hid dim]
        
        input = input.unsqueeze(0)
        
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(torch.where(input<self.output_dim, input, torch.ones_like(input)*self.unk_i)))
        #embedded = [1, batch size, emb dim]

        if self.attentions:
            att_hidden = hidden[0][-1] if isinstance(hidden, tuple) else hidden[-1]
            atts = [a(att_hidden, eo, m).unsqueeze(1) for a, eo, m in zip(self.attentions, encoder_outputs, mask)]
            ## a = self.attention(hidden.unsqueeze(1), encoder_outputs.permute(1,0,2), mask)
            ## a = a.unsqueeze(1)
            # a = List[batch size, 1, src len]
            

            weighted = [torch.bmm(F.softmax(a,dim=2), eo.permute(1,0,2)) for a, eo in zip(atts, encoder_outputs)]
            weighted = [w.permute(1, 0, 2) for w in weighted]
            weighted = torch.cat(weighted, dim=2)
            
            #weighted = [1, batch size, enc hid dim * 2]
            
            rnn_input = torch.cat((embedded, weighted), dim=2)
            
        else:
            rnn_input = embedded

        output, hidden = self.rnn(rnn_input, hidden)
        #output = [sent len, batch size, dec hid dim * n directions]
        #hidden = [n layers * n directions, batch size, dec hid dim]
        # sent len is always 1 in this decoder and n directions as well
        
        embedded = embedded.squeeze(0)
        output = self.dropout(output.squeeze(0))

        if self.attentions:
            weighted = weighted.squeeze(0)
            output = self.out(torch.cat((output, weighted, embedded), dim=1))
            if self.copy_mode:
                return output, hidden, [a.squeeze(1) for a in atts], self.p_gen(torch.cat((weighted, att_hidden), dim=1))
                
            return output, hidden, [a.squeeze(1) for a in atts]
        else:
            output = self.out(torch.cat((output, embedded), dim=1))
            return output, hidden
        
        #output = [bsz, output dim]
        