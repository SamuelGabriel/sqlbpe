from __future__ import print_function
import argparse
import numpy as np
import torch
import torch.utils.data
from torch import nn, optim
from torch.nn import functional as F
import utils
import torchtext
import encoder
import decoder
import attention
import embedding
import random
import spacy
from typing import NamedTuple, Optional
import json
from os import path, makedirs
from rolloutdecoder import RollOutDecoder
import hyperspherical_uniform
from params import Params
import pickle
import time
from simpleseq2seq import SimpleSeq2Seq
from seq2seq import Seq2Seq

from torchtext.datasets import Multi30k
from torchtext.data import Field, BucketIterator

random.seed(2019)
torch.manual_seed(2019)
torch.backends.cudnn.deterministic = True

parser = argparse.ArgumentParser(description='Seq2seq Trainer')
parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 32)')
parser.add_argument('--retention-epochs', type=int, default=10, metavar='N',
                    help='number of epochs to train (default: 10)')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--model-save-path', default='../newly_trained_model')
parser.add_argument('--model-load-path', default=None)
parser.add_argument('--shorter-epochs', action='store_true')
parser.add_argument('--no-intermediate-printing', action='store_false')
parser.add_argument('-p', '--params', default='{}', type=json.loads)
parser.add_argument('--train-path', default='../data/datasets/nl_to_sql/advising')
parser.add_argument('--test-mode', action='store_true')
parser.add_argument('--valid-mode', action='store_true')
parser.add_argument('--wordvec-dir', default='../glove')
parser.add_argument('--beam-size', default=0, type=int)
parser.add_argument('--beam-syntax-check', action='store_true')

def get_datasets_and_fields(train_path, reverse_input, bpe_encoding_steps, bpe_copy_masking, auto_bpe_min_freq, ast_bpe, bpe_step_override_p_schedule):
    if True:
        (dataset, validset, testset), (src, trg) = utils.get_datasets(data_path=train_path, reverse_input=reverse_input, bpe_encoding_steps=bpe_encoding_steps, bpe_copy_masking=bpe_copy_masking, auto_bpe_min_freq=auto_bpe_min_freq, ast_bpe=ast_bpe, bpe_step_override_p_schedule=bpe_step_override_p_schedule)
        return dataset, validset, testset, src, trg
    else:
        return utils.get_translation_dataset()


def get_data_loader(dataset, bs, device):
    return torchtext.data.BucketIterator(dataset, bs, sort_key = lambda x : len(x.src),
        sort_within_batch=True, device=device)

def get_rnn_type(rnn_type_name):
    if rnn_type_name == 'GRU':
        return nn.GRU
    elif rnn_type_name == 'LSTM':
        return nn.LSTM
    else:
        raise ValueError('RNN_TYPE {} not supported.'.format(rnn_type_name))

def get_model(params, wordvec_dir, src, trg, device):
    rnn_type = get_rnn_type(params.RNN_TYPE)
    return SimpleSeq2Seq(h_dim=params.HID_DIM, trg_vocab=trg.vocab, distribution='vmf',
                encoder=encoder.Encoder(len(src.vocab), params.HID_DIM, params.HID_DIM, params.HID_DIM, params.DROPOUT, len(trg.vocab), return_outputs=False, num_layers=params.NUM_ENCODER_LAYERS, bidirectional=params.BIDIRECTIONAL_ENCODER, embedding=embedding.get_embedding(src.vocab.itos, params.HID_DIM, wordvec_dir, freeze_embeddings=params.FREEZE_LOADED_EMBEDDINGS) if params.MIXED_SRC_EMBEDDINGS else None, rnn_type=rnn_type),
                decoder=RollOutDecoder(params.TEACHER_FORCING_RATIO, 
                    decoder.Decoder(len(trg.vocab), params.HID_DIM,
                        params.HID_DIM, params.HID_DIM, params.DROPOUT, trg.vocab.stoi['<unk>'], rnn_type=rnn_type,
                        embedding=embedding.get_embedding(trg.vocab.itos, params.HID_DIM, wordvec_dir,
                        freeze_embeddings=params.FREEZE_LOADED_EMBEDDINGS) if params.MIXED_TRG_EMBEDDINGS else None,
                        num_layers=params.NUM_DECODER_LAYERS).to(device), 
                    target_vocab=trg.vocab, 
                    device=device),
                fixed_kappa=params.FIXED_KAPPA
            ).to(device)

def get_attention_model_getter(copying=False, extra_copy_vocab=0):
    def get_att_model(params, wordvec_dir, src, trg, device):
        rnn_type = get_rnn_type(params.RNN_TYPE)
        src_inputs = [True]
        enc = encoder.Encoder(len(src.vocab), params.HID_DIM, params.HID_DIM, params.HID_DIM, params.DROPOUT, len(trg.vocab), src_inputs=src_inputs, return_outputs=True, pos_emb_dim=params.HID_DIM if copying else 0, rnn_type=rnn_type, num_layers=params.NUM_ENCODER_LAYERS,).to(device)

        attentions = [attention.Attention(params.HID_DIM, params.HID_DIM).to(device)]

        dec = decoder.Decoder(len(trg.vocab), params.HID_DIM, params.HID_DIM, params.HID_DIM, params.DROPOUT, trg.vocab.stoi['<unk>'], attentions=attentions, copy_mode=copying, rnn_type=rnn_type, num_layers=params.NUM_DECODER_LAYERS).to(device)
        rollout_decoder = RollOutDecoder(
            params.TEACHER_FORCING_RATIO, 
            dec,
            target_vocab=trg.vocab,
            device=device,
            with_attention=True,
            extra_copy_vocab=extra_copy_vocab
        ).to(device)

        return Seq2Seq(enc, rollout_decoder, src.vocab, trg.vocab, copying, copy_forcing=params.COPY_FORCING).to(device)
    return get_att_model

def make_loss_function(criterion):
    # Reconstruction + KL divergence losses summed over all elements and batch
    def loss_function(y_, y, z):
        # the [1:] is since the first token always is the beginning token <sos>
        y_ = y_[1:].view(-1, y_.shape[2])
        y = y[1:].view(-1)
        example_loss = criterion(y_, y)

        return example_loss
    return loss_function

# TODO strategy to get rid of all the dependencies in train... (train_loader, model, loss_function, ...)

def train(epoch, train_loader, model, loss_function, optimizer, trg, log_interval, params, attention, copying, intermediate_printing, shorter_epochs):
    model.train()
    train_loss = 0
    num_corr = 0
    num_total = 0
    t = time.time()
    for batch_idx, data in enumerate(train_loader):
        if shorter_epochs and batch_idx > 10:
            break
        src, src_len = data.src
        target, target_len = data.trg
        # data is a batch tensor [batch_]
        # data = data.view(-1, 784)
        optimizer.zero_grad()
        if not attention:
            recon_batch, preds, mu = model((src, src_len, target))
            loss = loss_function(recon_batch, target, mu)
        else:
            (copy_src, target), dyn_lens = model.generate_copy_inp_out(data)
            recon_batch, preds, _, p_gen = model([src], [src_len], target, inference=False, copy_inputs=[copy_src])
            loss = loss_function(recon_batch, target, copy_inputs=[copy_src], p_gen=p_gen)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.GRADIENT_NORM_CLIP)
        torch.nn.utils.clip_grad_norm_(model.encoder.embedding.parameters(), params.EMBEDDING_GRADIENT_NORM_CLIP)
        torch.nn.utils.clip_grad_norm_(model.decoder.decoder.embedding.parameters(), params.EMBEDDING_GRADIENT_NORM_CLIP)
        eq = (preds==target)
        m = torch.arange(len(eq)).to(target.device).unsqueeze(1).repeat(1, eq.shape[1])
        # TODO check this condition, it should include <sos> and <eos>?
        eq[m>=target_len] = 1
        num_corr += torch.sum(eq.all(dim=0)).item()
        num_total += target.shape[1]
        train_loss += loss.item()
        #print([p.grad for p in model.encoder.rnn.parameters()])
        #print([p.grad[:,0].nonzero() for p in model.encoder.embedding.parameters()])
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\tAcc: {}'.format(
                epoch, batch_idx, len(train_loader),
                100. * batch_idx / len(train_loader),
                loss.item(), num_corr/num_total))
        if random.random() < .002 and intermediate_printing:
            print('example ', batch_idx)
            print('trg: ', utils.batch_to_list_of_sentences(target[:,:1], trg.vocab.itos, max_len=10000))
            print('pred: ', utils.batch_to_list_of_sentences(preds[:,:1], trg.vocab.itos, max_len=10000))
    print('====> Epoch: {} Average loss: {:.4f} Acc: {}'.format(
          epoch, train_loss / (batch_idx + 1), num_corr/num_total), ' Processed in {}'.format(time.time()-t))

def line_tensor_to_list(t, eos_idx):
    r = [x.item() for x in t]
    e = r.index(eos_idx)
    return r[1:e]

def flatten_bpe_encodings(list_of_ints, trg_vocab):
    return sum([trg_vocab.itos[i].split('__') if i < len(trg_vocab.itos) else [i] for i in list_of_ints], [])

def test(epoch, test_loader, model, loss_function, trg, attention, intermediate_printing, beam_size=3, beam_syntax_check=False, return_all_predictions=False):
    model.eval()
    test_loss = 0
    begin_time = time.time()
    num_corr = 0
    num_total = 0
    mini_acc_corr = 0
    mini_acc_count = 0
    sm = torch.nn.Softmax(2)
    all_preds = []
    with torch.no_grad():
        for i, data in enumerate(test_loader):
            inp, inp_len = data.src
            target, target_len = data.trg
            if beam_size == 0:
                recon_batch, preds, mu = model((inp, inp_len, target))
                test_loss += loss_function(recon_batch, target, mu).item()
                mask = utils.sequence_mask(target_len, target.shape[0])
                num_corr += torch.sum((preds==target)[mask].all(dim=0))
                num_total += preds.shape[1]
                mini_acc_corr += torch.sum((preds.long() == target)[mask].type(torch.float))
                mini_acc_count += torch.sum(mask.type(torch.float))

                w = sm(recon_batch)[torch.arange(len(recon_batch)), 0, preds[:,0]]
                if random.random() < .005 and intermediate_printing:
                    print('example ', i)
                    print('trg: ', utils.batch_to_list_of_sentences(target[:,:1], trg.vocab.itos, max_len=1000))
                    print('pred: ', utils.batch_to_list_of_sentences(preds[:,:1], trg.vocab.itos, max_len=1000))
                    print('weight: ', list(w.cpu().numpy()[:1000]))
            else:
                if attention:
                    (copy_src, target), dyn_lens = model.generate_copy_inp_out(data)
                    seqs, neg_log_probs = model([inp], [inp_len], target, inference=True, beam_size=beam_size, copy_inputs=[copy_src], beam_syntax_check=beam_syntax_check)
                    recon_batch, preds, _, p_gen = model([inp], [inp_len], target, inference=False, copy_inputs=[copy_src])
                    ce_loss = loss_function(recon_batch, target, copy_inputs=[copy_src], p_gen=p_gen)
                else:
                    seqs, neg_log_probs = model((inp, inp_len, target), beam_size=beam_size, beam_syntax_check=beam_syntax_check)
                    recon_batch, _, mu = model((inp, inp_len, target))
                    ce_loss = loss_function(recon_batch, target, mu)
                #num_corr += torch.sum((preds==target)[mask].all(dim=0))
                #num_total += preds.shape[1]
                for sb, tb in zip(seqs, target.permute(1,0)):
                    tb = line_tensor_to_list(tb, trg.vocab.stoi['<eos>'])
                    tb = flatten_bpe_encodings(tb, trg.vocab)
                    unflattened_pred = [trg.vocab.itos[i] if i < len(trg.vocab.itos) else i for i in sb]
                    sb = flatten_bpe_encodings(sb, trg.vocab)
                    cp = 0
                    for i, (t, p) in enumerate(zip(tb, sb)):
                        cp += t == p and t != trg.vocab.stoi['<unk>']
                    mini_acc_corr += cp
                    mini_acc_count += max(len(tb), len(sb))
                    num_corr += len(tb) == len(sb) and cp == len(sb)
                    num_total += 1
                    if random.random() < .005 and intermediate_printing:
                        print('target:', tb)
                        print('prediction:', sb)
                        print('raw pred:', unflattened_pred)
                    if return_all_predictions:
                        all_preds.append(sb)

    test_loss = -num_corr/max(num_total, 1)
    print('====> Valid set loss: {:.4f}'.format(test_loss), 'Acc: {}'.format(num_corr/max(num_total, 1)), 'Mini Acc: {}'.format(mini_acc_corr/max(mini_acc_count,1)), 'Cross-Entropy Loss without Beam-Search {:.4f}'.format(ce_loss), ' Processed in {}'.format(time.time()-begin_time))
    if return_all_predictions:
        return test_loss, num_corr/max(num_total,1), all_preds
    return test_loss, num_corr/max(num_total,1)

def run(params: Params, device: torch.device, model_save_path: Optional[str], model_load_path: Optional[str], batch_size: int, wordvec_dir: str, train_path: str, retention_epochs: int, beam_size: int, log_interval: int = 100000, intermediate_printing: bool = True, test_mode: bool = False, valid_mode: bool = False, shorter_epochs: bool = False, beam_syntax_check: bool =False):
    def get_extra_copy_vocab(train_loader, valid_loader, test_loader):
        return max(torch.max(b.src[1]).item() for l in [train_loader, valid_loader, test_loader] for b in l) \
               if params.COPY_RETRIEVER else 0
    if model_load_path is not None:
        with open(path.join(model_load_path, 'hyper_params.pickle'), 'rb') as f:
            params = pickle.load(f)
        dataset, validset, testset, src, trg = get_datasets_and_fields(train_path, params.REVERSE_INPUT, params.BPE_ENCODING_STEPS, params.BPE_COPY_MASKING, params.AUTO_BPE_MIN_FREQ, params.AST_BPE, eval(params.BPE_STEP_OVERRIDE_P_SCHEDULE))
        train_loader = get_data_loader(dataset, batch_size, device)
        valid_loader = get_data_loader(validset, batch_size, device)
        test_loader = get_data_loader(testset, batch_size, device)
        model_getter = get_attention_model_getter(params.COPY_RETRIEVER, get_extra_copy_vocab(train_loader, valid_loader, test_loader)) if params.ATTENTION_RETRIEVER else get_model
        model = model_getter(params, wordvec_dir, src, trg, device)
        state_dict = torch.load(path.join(model_load_path, 'model_params.pt'), map_location=device)
        if 'decoder.decoder.p_gen.0.weight' in state_dict and not params.COPY_RETRIEVER:
            state_dict.pop('decoder.decoder.p_gen.0.weight')
            state_dict.pop('decoder.decoder.p_gen.0.bias')
        model.load_state_dict(state_dict)
    else:
        dataset, validset, testset, src, trg = get_datasets_and_fields(train_path, params.REVERSE_INPUT, params.BPE_ENCODING_STEPS, params.BPE_COPY_MASKING, params.AUTO_BPE_MIN_FREQ, params.AST_BPE, eval(params.BPE_STEP_OVERRIDE_P_SCHEDULE))
        train_loader = get_data_loader(dataset, batch_size, device)
        valid_loader = get_data_loader(validset, batch_size, device)
        test_loader = get_data_loader(testset, batch_size, device)
        model_getter = get_attention_model_getter(params.COPY_RETRIEVER, get_extra_copy_vocab(train_loader, valid_loader, test_loader)) if params.ATTENTION_RETRIEVER else get_model
        model = model_getter(params, wordvec_dir, src, trg, device)
        
    
    if params.ATTENTION_RETRIEVER:
        optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
        loss_function = model.loss
    else:
        pad_idx = trg.vocab.stoi['<pad>']
        criterion = nn.CrossEntropyLoss(ignore_index=pad_idx)
        optimizer = optim.Adam(model.parameters(), lr=params.LEARNING_RATE)
        loss_function = make_loss_function(criterion)
    if test_mode or valid_mode:
        return test(1, test_loader if test_mode else valid_loader, model, loss_function, trg, params.ATTENTION_RETRIEVER, intermediate_printing, beam_size=beam_size, beam_syntax_check=beam_syntax_check)
    min_loss = 4
    max_acc = 0
    epoch = 1
    last_improve_epoch = 0
    val_acc_list = []
    while True:
        train(epoch, train_loader, model, loss_function, optimizer, trg, log_interval, params, params.ATTENTION_RETRIEVER, params.COPY_RETRIEVER, intermediate_printing, shorter_epochs)
        if (epoch-1) % 1 == 0:
            l, acc = test(epoch, valid_loader, model, loss_function, trg, params.ATTENTION_RETRIEVER, intermediate_printing, beam_size=beam_size, beam_syntax_check=beam_syntax_check)
            val_acc_list.append(acc)
            max_acc = max(acc, max_acc)
            if l < min_loss:
                last_improve_epoch = epoch
                min_loss = l
                if model_save_path:
                    makedirs(model_save_path, exist_ok=True)
                    torch.save(model.state_dict(), path.join(model_save_path, 'model_params.pt'))
                    with open(path.join(model_save_path, 'hyper_params.pickle'), 'wb') as f:
                        pickle.dump(params, f)
                    print('Saved model.')
        if epoch - retention_epochs > last_improve_epoch:
            break
        epoch += 1
    return min_loss, max_acc, val_acc_list


if __name__ == "__main__":
    args = parser.parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = utils.get_device(args.no_cuda)

    params = Params(**args.params)
    run_out = run(params, device, args.model_save_path, args.model_load_path, args.batch_size, args.wordvec_dir, args.train_path, args.retention_epochs, args.beam_size, args.log_interval, test_mode=args.test_mode, valid_mode=args.valid_mode, shorter_epochs=args.shorter_epochs, intermediate_printing=args.no_intermediate_printing, beam_syntax_check=args.beam_syntax_check)
    if args.test_mode or args.valid_mode:
        print(('7', dict(params._asdict()), args.train_path, run_out[0], run_out[1]))
    else:
        print(('123', dict(params._asdict()), args.train_path, run_out[1], run_out[2]))