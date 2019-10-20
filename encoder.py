import torch
from torch import nn

class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, enc_hid_dim, dec_hid_dim, dropout, output_dim, src_inputs=[True], num_layers=2, return_outputs=True, pos_emb_dim=0, copy_space=300, bidirectional=True, embedding=None, out_embedding=None, rnn_type=nn.GRU):
        """This is the general encoder for both the VAE and the editor.
        Set pos_emb_dim=0 to not use positional embeddings for the token-based copy approach.
        
        Arguments:
            input_dim {int} 
            emb_dim {int}
            enc_hid_dim {int}
            dec_hid_dim {int}
            dropout {float} -- in [0.,1.]
            output_dim {int} 
        
        Keyword Arguments:
            src_inputs {list} -- List of all inputs and if they come from the source or the target distribution (default: {[True]})
            num_layers {int} -- (default: {2})
            return_outputs {bool} -- (default: {True})
        """

        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.enc_hid_dim = enc_hid_dim
        self.dec_hid_dim = dec_hid_dim
        self.dropout = dropout
        self.return_outputs = return_outputs
        self.num_layers = num_layers
        self.src_inputs = src_inputs
        self.embedding = embedding or nn.Embedding(input_dim, emb_dim)
        self.out_embedding = out_embedding or nn.Embedding(output_dim, emb_dim)
        self.pos_emb_dim = pos_emb_dim
        self.copy_space = copy_space
        self.dec_hid_dim = dec_hid_dim
        if self.pos_emb_dim:
            # the longest sequence has length 256 in out dataset, and we have len(src_inputs) many such sequences
            self.pos_embedding = nn.Embedding(len(src_inputs)*self.copy_space, self.pos_emb_dim) 
        
        self.rnn = rnn_type(emb_dim+pos_emb_dim, enc_hid_dim, bidirectional=bidirectional, num_layers=self.num_layers, dropout=dropout)
        self.trg_rnn = rnn_type(emb_dim+pos_emb_dim, enc_hid_dim, bidirectional=bidirectional, num_layers=self.num_layers, dropout=dropout)
        
        self.fc = nn.Linear(enc_hid_dim * len(self.src_inputs) * self.num_layers * (2 ** int(bidirectional)), dec_hid_dim) # times 2 bc biderectional
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, src, src_len):
        
        #src = List[src sent len, batch size]
        #src_len = List[batch size]
        hiddens = []
        outputs = []
        if self.pos_emb_dim:
            pos_embeddings_to_use = [self.pos_embedding(torch.arange(self.copy_space).to(src[0].device) + (i*self.copy_space)) for i, _ in enumerate(self.src_inputs)]
        else:
            pos_embeddings_to_use = [None for i, _ in enumerate(self.src_inputs)]

        for s, sl, pos_embeddings, is_source in zip(src, src_len, pos_embeddings_to_use, self.src_inputs):
            sorted_sl, sorted_idx = sl.sort(descending=True)
            sorted_s = s[:, sorted_idx]        

            embedded = self.dropout(self.embedding(sorted_s)) if is_source else self.dropout(self.out_embedding(sorted_s))
            if self.pos_emb_dim:
                l = embedded.size()[0]
                b = embedded.size()[1]
                pos_embeddings = pos_embeddings[:l].unsqueeze(1).repeat(1, b, 1)
                embedded = torch.cat((embedded, pos_embeddings), 2)

            
            #embedded = [src sent len, batch size, emb dim + pos emb dim]
            packed_embedded = nn.utils.rnn.pack_padded_sequence(embedded, sorted_sl)
            packed_outputs, hidden = self.rnn(packed_embedded) if is_source else self.trg_rnn(packed_embedded)
            if type(self.rnn) == nn.LSTM:
                hidden = hidden[0]
            output, _ = nn.utils.rnn.pad_packed_sequence(packed_outputs, total_length=s.size()[0])
            _, unsort_idx = sorted_idx.sort(0) 
            assert torch.all(sl == sorted_sl[unsort_idx])
            output = output[:, unsort_idx]
            hidden = hidden[:, unsort_idx]
            hiddens.append(hidden)
            outputs.append(output)
        
        #outputs = List[sent len, batch size, hid dim * num directions]
        #hidden = List[n layers * num directions, batch size, hid dim]
        
        #hiddens elements are stacked [forward_1, backward_1, forward_2, backward_2, ...]
        #outputs are always from the last layer
        
        #hidden [-2, :, : ] is the last of the forward RNNs
        #hidden [-1, :, : ] is the last of the backward RNNs
        
        #initial decoder hidden is final hidden state of the forwards and backwards encoder RNNs fed through a linear layer
        concatinated_hidden = torch.cat(hiddens, dim=-1)
        concatinated_hidden = concatinated_hidden.permute(1,0,2).contiguous().view(concatinated_hidden.shape[1],-1)
        if concatinated_hidden.shape[1] == self.dec_hid_dim:
            hidden = concatinated_hidden
        else:
            hidden = torch.tanh(self.fc(concatinated_hidden))
        
        
        #outputs = List[sent len, batch size, enc hid dim * 2]
        #hidden = [batch size, dec hid dim]
        
        if self.return_outputs:
            return outputs, hidden
        else:
            return hidden