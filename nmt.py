# coding=utf-8
"""
    A very basic implementation of neural machine translation
    
    Usage:
    nmt.py train --train-src=<file> --train-tgt=<file> --dev-src=<file> --dev-tgt=<file> --vocab=<file> [options]
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE OUTPUT_FILE
    nmt.py decode [options] MODEL_PATH TEST_SOURCE_FILE TEST_TARGET_FILE OUTPUT_FILE
    
    Options:
    -h --help                               show this screen.
    --cuda                                  use GPU
    --train-src=<file>                      train source file
    --train-tgt=<file>                      train target file
    --dev-src=<file>                        dev source file
    --dev-tgt=<file>                        dev target file
    --vocab=<file>                          vocab file
    --seed=<int>                            seed [default: 0]
    --batch-size=<int>                      batch size [default: 32]
    --embed-size=<int>                      embedding size [default: 256]
    --hidden-size=<int>                     hidden size [default: 256]
    --clip-grad=<float>                     gradient clipping [default: 5.0]
    --log-every=<int>                       log every [default: 10]
    --max-epoch=<int>                       max epoch [default: 30]
    --patience=<int>                        wait for how many iterations to decay learning rate [default: 5]
    --max-num-trial=<int>                   terminate training after how many trials [default: 5]
    --lr-decay=<float>                      learning rate decay [default: 0.5]
    --beam-size=<int>                       beam size [default: 5]
    --lr=<float>                            learning rate [default: 0.001]
    --uniform-init=<float>                  uniformly initialize all parameters [default: 0.1]
    --save-to=<file>                        model save path
    --valid-niter=<int>                     perform validation after how many iterations [default: 2000]
    --dropout=<float>                       dropout [default: 0.2]
    --max-decoding-time-step=<int>          maximum number of decoding time steps [default: 70]
"""



from collections import namedtuple
import sys
import math
import pickle
import time
from typing import List, Tuple, Dict, Set, Union
import torch
import torch.nn as nn
import torch.nn.utils
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_packed_sequence, pack_padded_sequence
from docopt import docopt
from nltk.translate.bleu_score import corpus_bleu, sentence_bleu, SmoothingFunction
import numpy as np
from typing import List, Tuple, Dict, Set, Union
from tqdm import tqdm
from utils import read_corpus, batch_iter
from vocab import Vocab, VocabEntry

Hypothesis = namedtuple('Hypothesis', ['value', 'score'])


class CreateEmbeddings(nn.Module): # Done
    def __init__(self, embedding_size,vocab):
        super().__init__()
        self.source, self.target =  None, None
        pad_source_index = vocab.src['<pad>']
        pad_target_index = vocab.tgt['<pad>']
        
        self.source = nn.Embedding(num_embeddings=len(vocab.src), embedding_dim=embedding_size, padding_idx=pad_source_index)
        
        self.target = nn.Embedding(num_embeddings=len(vocab.tgt), embedding_dim=embedding_size, padding_idx=pad_target_index)


class NMT(nn.Module):

    def __init__(self, embed_size, hidden_size, vocab, dropout_rate=0.2): #Done

        super().__init__()
        
        #1. Create variables
        self.embed_size = embed_size
        self.hidden_size = hidden_size
        self.dropout_rate = dropout_rate
        self.vocab = vocab

        self.source_vocab_len = len(self.vocab.src)
        self.target_vocab_len = len(self.vocab.tgt)
        
        #2. Embedding Layer
        self.create_embeddings = CreateEmbeddings(embed_size, vocab)
        
        #3. Initialize Encoder and Decoder
        self.encoder_lstm = nn.LSTM(input_size=embed_size, hidden_size=hidden_size, bias=True, bidirectional=True)
        self.decoder_lstm = nn.LSTMCell(input_size=embed_size + hidden_size, hidden_size=hidden_size, bias=True)
        
        #4. Initilize Dropout layer
        self.dropout = nn.Dropout(self.dropout_rate)
    
        #5. Linear Layer for Prob. Distro over vocab
        self.target_vocab_linear = torch.nn.Linear(in_features=self.hidden_size, out_features=self.target_vocab_len, bias=False)
        
        #6. Linear Layer for combined output
        self.combined_output_linear = torch.nn.Linear(in_features=3 * hidden_size, out_features=self.hidden_size, bias=False)
        
        #7. Attention Layers (Bi-directional (x2) input to single output (x1) )
        self.h_linear = torch.nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.c_linear = torch.nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
        self.att_linear = torch.nn.Linear(in_features=2 * hidden_size, out_features=hidden_size, bias=False)
    
    
    def getTensor(self, X,target=False): # Done
        if target==False:
            t = self.vocab.src.to_input_tensor(X, device=self.device)
            return t
        else:
            t = self.vocab.tgt.to_input_tensor(X, device=self.device)
            return t



    def mask_sequence(self, current_hiddens, len_list): #Done
        #1. Get a tensor which will be masked
        def getZeroTensor(X):
            batch_size = X.size(0)
            input_length = X.size(1)
            return torch.zeros(batch_size, input_length, dtype=torch.float)
        
        masks = getZeroTensor(current_hiddens)
        
        # 2. Mask with 1
        for index, length in enumerate(len_list):
            masks[index,length:]=1
        return masks.to(self.device)
    
    
    def __call__(self, src_sents, tgt_sent): #Done
    
        def getVocabDistro(final, output_p, d):
            probs = torch.nn.functional.log_softmax(self.target_vocab_linear(final), dim=d)
            t_masks = (output_p != self.vocab.tgt['<pad>']).float()
            log_prob = torch.gather(probs, index=output_p[1:].unsqueeze(d), dim=d).squeeze(d) * t_masks[1:]
            return log_prob.sum(dim=0)
        
        
        #1. Convert sentences to tensors of length X batch_size
        input_p, output_p = self.getTensor(src_sents), self.getTensor(tgt_sent,target=True)
        
        #2. Get length of sentences
        length=[]
        for i in src_sents:
            length.append(len(i))
        
        input_len =None
        input_len =length
        #3. Encode the input
        current_hiddens, initial_decoder_state = self.encode(input_p, input_len)
        
        #4. Mask the sentences
        masks = self.mask_sequence(current_hiddens, input_len)
        
        #print("Working upto step 4")
        
        #5. Decode the masked sequence
        final= self.decode(current_hiddens,masks, initial_decoder_state, output_p)
        
        #print("Working upto step 5")
        
        #6. Compute Prob. Ditro.
        scores = getVocabDistro(final, output_p, -1)
        
        #print("Working upto step 6")

        return scores


    def getZeroVectorDevice(self,X,Y):
        return torch.zeros(X, Y, device=self.device)

    def getZeroVectorSizeFloatDevice(self,x):
        return torch.zeros(x, dtype=torch.float, device=self.device)


    def encode(self, sent_tensor, len_list): #Done
        # sent_tensor = Max_length_of_sent X Batch_size
        
        def get_bidirectional_state(H):
            return torch.cat((H[0],H[1]),1)
        
        def exchange_dims(X):
            return X.permute(1,0,2)
        
        #1. Create source embeddings from input text
        source_embeddings = self.create_embeddings.source(sent_tensor)
        
        
        #2. PACK- Converting RNNs into Dynamic RNNs (i.e. mask/don't consider padding while encoding the input and add padding later)
        # More: https://discuss.pytorch.org/t/understanding-pack-padded-sequence-and-pad-packed-sequence/4099/6
        
        packed_source_embeddings = torch.nn.utils.rnn.pack_padded_sequence(source_embeddings, lengths=len_list)
        
        #3. Pass it through the encoder
        current_hiddens_pack, (prev_hidden, prev_cell) = self.encoder_lstm(packed_source_embeddings)
        
        #4. PAD-(Unpack) & remove extra dimension
        max_sent_len = sent_tensor.size(0)
        current_hiddens,_ = torch.nn.utils.rnn.pad_packed_sequence(current_hiddens_pack)
        
        #5. Fix dimensions
        # Note: current_hiddens is max_sent_len x batch_size x hidden_size*2
        # We want to return as return as batch_size x  max_sent_len x hidden_size*2
        current_hiddens = exchange_dims(current_hiddens)
        
        #6. Get initial hidden state of the decoder (along with linear form)
        initial_hidden_decoder= get_bidirectional_state(prev_hidden)
        initial_hidden_decoder_linear= self.h_linear(initial_hidden_decoder)
        
        #7. Get initial cell state of the decoder (along with linear form)
        initial_cell_decoder=get_bidirectional_state(prev_cell)
        initial_cell_decoder_linear= self.c_linear(initial_cell_decoder)
        
        
        return current_hiddens, (initial_hidden_decoder_linear, initial_cell_decoder_linear)
    


    def decode(self, current_hiddens, masks, initial_decoder_state, output_padding): #Done
        def getzero_batch(current_hiddens):
            batch_size = current_hiddens.size(0)
            vec = self.getZeroVectorDevice(batch_size, self.hidden_size)
            return vec, batch_size
        
        
        def decode_step(embed_t, final_prev, current_state, current_hiddens_linear):
            embed_t = embed_t.squeeze()  # #shape of embed_t : (1, b, e) -> (b,e)
            embed_concat = torch.cat((embed_t, final_prev), dim=1)
            current_state, final_prev, temp = self.forward_attention(embed_concat, current_state, current_hiddens, current_hiddens_linear, masks)
            return current_state, final_prev
        
        
        #1. final_results at each time step
        final=[]
        
        #2. Remove the end token
        last_index = len(output_padding)-1
        output_padding = output_padding[:last_index]
        
        #3. Initial decoder state
        current_state = initial_decoder_state  # (hidden, cell)
        
        
        #4. previous final output is zero vector
        final_prev, batch_size = getzero_batch (current_hiddens)
        
        
        #5. Attention 1: Compute W * h_encoder
        current_hiddens_linear = self.att_linear(current_hiddens)   # (b, input_len, h)
        
        #6. Attention 2: Create target embeddings
        target_embeddings = self.create_embeddings.target(output_padding) # (output_len, b, e)
        
        #7. Attention 3: Iterate over target-embeddings and do step-by-step decoding
        
        #Split along the output_len dimension
        target_decoded_embed = torch.split(target_embeddings, 1,dim=0)
        
        for embed_t in target_decoded_embed:
            current_state, final_prev = decode_step(embed_t, final_prev, current_state,current_hiddens_linear)
            final.append(final_prev)
        
        
        #8. Attention 4: Return Final outputs (stacked)
        return torch.stack(final, dim=0)



    def forward_attention(self, embed_concat, current_state, current_hiddens, current_hiddens_linear, masks): #Done
        final=None
        
        def batch_mm(A,B,d):
            K = B.unsqueeze(dim=d)
            return torch.bmm(A,K).squeeze(dim=d)
        
        def batch_mm2(A,B,d):
            K = A.unsqueeze(dim=d)
            return torch.bmm(K,B).squeeze(dim=d)
        
        def mask_attention(attention_scores):
            if masks is None:
                return attention_scores
            else:
                return attention_scores.data.masked_fill_(masks.byte(), -float('inf')) # Ref: https://bastings.github.io/annotated_encoder_decoder/
        
        
        def multiplicative_attention(attention_scores,current_hiddens, current_hidden_state_decoder,d ):
            # Ref: Bilinear Function- http://phontron.com/class/mtandseq2seq2018/assets/slides/mt-fall2018.chapter8.pdf
            alpha_t = torch.nn.functional.softmax(attention_scores, dim=d)
            a_t = batch_mm2(alpha_t,current_hiddens,d)
            
            U_t = torch.cat((a_t, current_hidden_state_decoder), dim=d) #current_hidden_state_decoder
            
            V_t = self.combined_output_linear(U_t)
            O_t = self.dropout(torch.tanh(V_t))
            return O_t
        
        
        #1. Pass via decoder to get the new state
        current_state_decoder = self.decoder_lstm(embed_concat, current_state)
        
        #2. Break current state into current hidden and cell state
        current_hidden_state_decoder, current_cell_state_decoder = current_state
        
        #3. Batch matrix multiplication for computing attention scores- (b, source_len)
        attention_scores = batch_mm(current_hiddens_linear,current_hidden_state_decoder,2)
        
        #4. Mask out invalid positions | If mask==1, set to -inf
        attention_scores = mask_attention(attention_scores)
        
        #5. Multiplicative Attention
        final = multiplicative_attention(attention_scores,current_hiddens, current_hidden_state_decoder,1 )
        
        return current_state_decoder, final, attention_scores

    def getEncodedHiddenStates(self, sent): #Done
        length_list = [len(sent)]
        #1. Convert Sentence to Embeddings
        source_embedding = self.getTensor([sent])

        #2. Get Encoder hidden states and final hidden state of the encoder= initial state of decoder
        source_hidden_states, initial_decode_var = self.encode(source_embedding, length_list)

        return source_hidden_states, initial_decode_var

    def getHiddenAndCellStates(self, branch, attention_vector, dec_init_vec, exp_src_encodings, exp_src_encodings_att_linear,last_index):
        target_sequence = []
        
        for b in range(len(branch)):
            target_sequence.append(self.vocab.tgt[branch[b][last_index]])
        
        embed = self.create_embeddings.target(torch.tensor(target_sequence, dtype=torch.long, device=self.device))
        
        return self.forward_attention(torch.cat([embed, attention_vector], dim=last_index), dec_init_vec, exp_src_encodings, exp_src_encodings_att_linear, masks=None)
    

    def getSourceEncodings(self, branch_size, src_enc,s1,s2,src_encodings, src_encodings_att_linear):
            dim1 = src_enc.size(s1)
            dim2 = src_enc.size(s2)
            dim3 = src_encodings_att_linear.size(s1)
            dim4 = src_encodings_att_linear.size(s2)
            enc = src_encodings.expand(branch_size, dim1, dim2)
            enc_linear = src_encodings_att_linear.expand(branch_size,dim3, dim4)
            return enc, enc_linear
    
    def getCumuProb(self, last_index, log_time, prob):
        prob_un = prob.unsqueeze(-last_index)
        expan_p = prob_un.expand_as(log_time)
        reduc = expan_p+log_time
        return reduc.view(last_index)
    

    def get_log_prob(self,attention, beam_size, final_branch ,last_index,prob): # Ref: https://ufal.mff.cuni.cz/pbml/109/art-caglayan-et-al.pdf
        log_time = F.log_softmax(self.target_vocab_linear(attention), dim=last_index)
        current_len = len(final_branch)
        contiuating_prob = self.getCumuProb(last_index, log_time, prob)
        top_cand_prob, top_cand_hyp_pos = torch.topk(contiuating_prob, k=beam_size - current_len)
        
        prev_hyp_ids = top_cand_hyp_pos / len(self.vocab.tgt)
        hyp_word_ids = top_cand_hyp_pos % len(self.vocab.tgt)

        return prev_hyp_ids, hyp_word_ids, top_cand_hyp_pos

    def branchBuilder(self, prev_id, word_id, cand_new_score, new_branch, final_branch, live_hyp_ids, new_prob, branch):
        new_sent = branch[prev_id] + [self.vocab.tgt.id2word[word_id]]
        if self.vocab.tgt.id2word[word_id] == '</s>':
            final_branch.append(Hypothesis(value=new_sent[1:-1], score=cand_new_score))
        else:
            new_branch.append(new_sent)
            live_hyp_ids.append(prev_id)
            new_prob.append(cand_new_score)
        return new_branch, final_branch, live_hyp_ids, new_prob, branch

    
    def beam_step(self,branch, src_encodings, dec_init_vec, attention_vector, src_encodings_att_linear, final_branch, prob, beam_size):
        
        #1.Obtain Source Encodings and with linear attention layer
        exp_src_encodings, exp_src_encodings_att_linear = self.getSourceEncodings(len(branch), src_encodings,1,2, src_encodings, src_encodings_att_linear)
        
        
        #2. Get hidden and cell states
        (hidden_state, cell_state), attention, temp = self.getHiddenAndCellStates(branch, attention_vector, dec_init_vec, exp_src_encodings, exp_src_encodings_att_linear,-1)
        

        #3.  log probabilities over target words
        prev_ids, word_ids, top_prob = self.get_log_prob(attention, beam_size, final_branch ,-1, prob)
        
        
        new_branch, live_hyp_ids, new_prob = [], [], []

        
        #4. Iterate over word_ids and compute the correct branch
        
        for i in range(len(word_ids)):
            
            new_branch, final_branch, live_hyp_ids, new_prob, branch = self.branchBuilder(prev_ids[i].item(), word_ids[i].item() , top_prob[i].item() ,new_branch, final_branch, live_hyp_ids, new_prob, branch)
        


        #5. Beam size achieved, stop decoding
        if len(final_branch) == beam_size:
            return final_branch, src_encodings, dec_init_vec, attention_vector, branch, src_encodings_att_linear , prob, True


        #6.Compute Attention Vector

        attention_vector = attention[torch.tensor(live_hyp_ids, dtype=torch.long, device=self.device)]
    
        return final_branch, src_encodings, (hidden_state[live_hyp_ids], cell_state[live_hyp_ids]), attention_vector, new_branch, src_encodings_att_linear , prob, False, beam_size, torch.tensor(new_prob, dtype=torch.float, device=self.device)

    def beam_search(self, src_sent: List[str], beam_size: int = 5, max_decoding_time_step: int = 70) -> List[Hypothesis]:
        
       
        #1. Get Hidden States from the Encoder
        src_encodings, dec_init_vec = self.getEncodedHiddenStates(src_sent) #

        #2. Initial Zero Vector and Linear Transformation for Bilinear attention # Ref: Bilinear Function- http://phontron.com/class/mtandseq2seq2018/assets/slides/mt-fall2018.chapter8.pdf
        src_encodings_att_linear = self.att_linear(src_encodings) #
        attention_vector = self.getZeroVectorDevice(1, self.hidden_size)#

        #3. Initilialize branches for beam search
        time_step = 0 #
        branch = [] #
        final_branch = [] #
        prob = self.getZeroVectorSizeFloatDevice(1) #
        branch.append(['<s>'])
        
        #4. Decode each time step until we are over the size of the beam or the max time step
        
        for time_step in range(max_decoding_time_step):
            if len(final_branch) >= beam_size:
                break
            else:
                final_branch, src_encodings, dec_init_vec, attention_vector, branch, src_encodings_att_linear, prob, binary, beam_size,prob = self.beam_step(branch, src_encodings, dec_init_vec, attention_vector, src_encodings_att_linear, final_branch, prob, beam_size)
                if binary==True:
                    break

    
        if not final_branch:
            final_branch.append(Hypothesis(value=branch[0][1:], score=prob[0].item()))

        return final_branch.sort(key=lambda hyp: hyp.score, reverse=True)


    @property
    def device(self) -> torch.device:
        return self.create_embeddings.source.weight.device

    @staticmethod
    def load(model_path: str):
        params = torch.load(model_path, map_location=lambda storage, loc: storage)
        args = params['args']
        model = NMT(vocab=params['vocab'], **args)
        model.load_state_dict(params['state_dict'], strict=False)

        return model

    def save(self, path: str):
        print('save model parameters to [%s]' % path, file=sys.stderr)

        params = {
            'args': dict(embed_size=self.create_embeddings.embed_size, hidden_size=self.hidden_size,
                         dropout_rate=self.dropout_rate),
            'vocab': self.vocab,
            'state_dict': self.state_dict()
        }

        torch.save(params, path)



def evaluate_ppl(model, dev_data, batch_size=32):
    was_training = model.training
    model.eval()
    
    cum_loss = 0.
    cum_tgt_words = 0.
    
    with torch.no_grad():
        for src_sents, tgt_sents in batch_iter(dev_data, batch_size):
            loss = -model(src_sents, tgt_sents).sum()
            
            cum_loss += loss.item()
            tgt_word_num_to_predict = sum(len(s[1:]) for s in tgt_sents)
            cum_tgt_words += tgt_word_num_to_predict
        
        ppl = np.exp(cum_loss / cum_tgt_words)
    
    if was_training:
        model.train()
    
    return ppl


def compute_corpus_level_bleu_score(references: List[List[str]], hypotheses: List[Hypothesis]) -> float:

    if references[0][0] == '<s>':
        references = [ref[1:-1] for ref in references]
    bleu_score = corpus_bleu([[ref] for ref in references],
                             [hyp.value for hyp in hypotheses])
    return bleu_score


def train(args: Dict):
    train_data_src = read_corpus(args['--train-src'], source='src')
    train_data_tgt = read_corpus(args['--train-tgt'], source='tgt')
    
    dev_data_src = read_corpus(args['--dev-src'], source='src')
    dev_data_tgt = read_corpus(args['--dev-tgt'], source='tgt')
    
    train_data = list(zip(train_data_src, train_data_tgt))
    dev_data = list(zip(dev_data_src, dev_data_tgt))
    
    train_batch_size = int(args['--batch-size'])
    clip_grad = float(args['--clip-grad'])
    valid_niter = int(args['--valid-niter'])
    log_every = int(args['--log-every'])
    model_save_path = args['--save-to']
    
    vocab = Vocab.load(args['--vocab'])
    
    model = NMT(embed_size=int(args['--embed-size']),
                hidden_size=int(args['--hidden-size']),
                dropout_rate=float(args['--dropout']),
                vocab=vocab)
    model.train()


    uniform_init = float(args['--uniform-init'])
    if np.abs(uniform_init) > 0.:
        print('uniformly initialize parameters [-%f, +%f]' % (uniform_init, uniform_init), file=sys.stderr)
        for p in model.parameters():
            p.data.uniform_(-uniform_init, uniform_init)


    vocab_mask = torch.ones(len(vocab.tgt))
    vocab_mask[vocab.tgt['<pad>']] = 0

    device = torch.device("cuda:0" if args['--cuda'] else "cpu")
    print('use device: %s' % device, file=sys.stderr)

    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=float(args['--lr']))

    num_trial = 0
    train_iter = patience = cum_loss = report_loss = cum_tgt_words = report_tgt_words = 0
    cum_examples = report_examples = epoch = valid_num = 0
    hist_valid_scores = []
    train_time = begin_time = time.time()
    print('begin Maximum Likelihood training')


    while True:
        epoch += 1

        for src_sents, tgt_sents in batch_iter(train_data, batch_size=train_batch_size, shuffle=True):
            train_iter += 1
            
            optimizer.zero_grad()
            
            batch_size = len(src_sents)
            
            example_losses = -model(src_sents, tgt_sents) # (batch_size,)
            batch_loss = example_losses.sum()
            loss = batch_loss / batch_size
            
            loss.backward()
            
        
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), clip_grad)
            
            optimizer.step()


            batch_losses_val = batch_loss.item()
            report_loss += batch_losses_val
            cum_loss += batch_losses_val
    
            tgt_words_num_to_predict = sum(len(s[1:]) for s in tgt_sents)  # omitting leading `<s>`
            report_tgt_words += tgt_words_num_to_predict
            cum_tgt_words += tgt_words_num_to_predict
            report_examples += batch_size
            cum_examples += batch_size



            if train_iter % log_every == 0:
                print('epoch %d, iter %d, avg. loss %.2f, avg. ppl %.2f ' \
                      'cum. examples %d, speed %.2f words/sec, time elapsed %.2f sec' % (epoch, train_iter,
                                                                                         report_loss / report_examples,
                                                                                         math.exp(report_loss / report_tgt_words),
                                                                                         cum_examples,
                                                                                         report_tgt_words / (time.time() - train_time),
                                                                                         time.time() - begin_time), file=sys.stderr)
                          
                train_time = time.time()
                report_loss = report_tgt_words = report_examples = 0.


  
            if train_iter % valid_niter == 0:
                print('epoch %d, iter %d, cum. loss %.2f, cum. ppl %.2f cum. examples %d' % (epoch, train_iter,
                                                                                             cum_loss / cum_examples,
                                                                                             np.exp(cum_loss / cum_tgt_words),
                                                                                             cum_examples), file=sys.stderr)

                cum_loss = cum_examples = cum_tgt_words = 0.
                valid_num += 1
        
                print('begin validation ...', file=sys.stderr)
        
  
                dev_ppl = evaluate_ppl(model, dev_data, batch_size=128)
                valid_metric = -dev_ppl
                
                print('validation: iter %d, dev. ppl %f' % (train_iter, dev_ppl), file=sys.stderr)
                
                is_better = len(hist_valid_scores) == 0 or valid_metric > max(hist_valid_scores)
                hist_valid_scores.append(valid_metric)
                
                if is_better:
                    patience = 0
                    print('save currently the best model to [%s]' % model_save_path, file=sys.stderr)
                    model.save(model_save_path)
                    
         
                    torch.save(optimizer.state_dict(), model_save_path + '.optim')
                elif patience < int(args['--patience']):
                    patience += 1
                    print('hit patience %d' % patience, file=sys.stderr)
                    
                    if patience == int(args['--patience']):
                        num_trial += 1
                        print('hit #%d trial' % num_trial, file=sys.stderr)
                        if num_trial == int(args['--max-num-trial']):
                            print('early stop!', file=sys.stderr)
                            exit(0)
                    
                 
                        lr = optimizer.param_groups[0]['lr'] * float(args['--lr-decay'])
                        print('load previously best model and decay learning rate to %f' % lr, file=sys.stderr)
                        
            
                        params = torch.load(model_save_path, map_location=lambda storage, loc: storage)
                        model.load_state_dict(params['state_dict'], strict=False)
                        model = model.to(device)
                        
                        print('restore parameters of the optimizers', file=sys.stderr)
                        optimizer.load_state_dict(torch.load(model_save_path + '.optim'))
                        
                      
                        for param_group in optimizer.param_groups:
                            param_group['lr'] = lr
                        
                       
                        patience = 0
                
                if epoch == int(args['--max-epoch']):
                    print('reached maximum number of epochs!', file=sys.stderr)
                    exit(0)


def decode(args: Dict[str, str]):
    print("load test source sentences from [{}]".format(args['TEST_SOURCE_FILE']), file=sys.stderr)
    test_data_src = read_corpus(args['TEST_SOURCE_FILE'], source='src')
    if args['TEST_TARGET_FILE']:
        print("load test target sentences from [{}]".format(args['TEST_TARGET_FILE']), file=sys.stderr)
        test_data_tgt = read_corpus(args['TEST_TARGET_FILE'], source='tgt')

    print("load model from {}".format(args['MODEL_PATH']), file=sys.stderr)
    model = NMT.load(args['MODEL_PATH'])

    if args['--cuda']:
        model = model.to(torch.device("cuda:0"))
        
    hypotheses = beam_search(model, test_data_src,
                                 beam_size=int(args['--beam-size']),
                                 max_decoding_time_step=int(args['--max-decoding-time-step']))
            
    if args['TEST_TARGET_FILE']:
        top_hypotheses = [hyps[0] for hyps in hypotheses]
        bleu_score = compute_corpus_level_bleu_score(test_data_tgt, top_hypotheses)
        print('Corpus BLEU: {}'.format(bleu_score * 100), file=sys.stderr)

    with open(args['OUTPUT_FILE'], 'w') as f:
        for src_sent, hyps in zip(test_data_src, hypotheses):
            top_hyp = hyps[0]
            hyp_sent = ' '.join(top_hyp.value)
            f.write(hyp_sent + '\n')


def beam_search(model: NMT, test_data_src: List[List[str]], beam_size: int, max_decoding_time_step: int) -> List[List[Hypothesis]]:

    was_training = model.training
    model.eval()
    
    hypotheses = []
    with torch.no_grad():
        for src_sent in tqdm(test_data_src, desc='Decoding', file=sys.stdout):
            example_hyps = model.beam_search(src_sent, beam_size=beam_size, max_decoding_time_step=max_decoding_time_step)
            
            hypotheses.append(example_hyps)

    if was_training: model.train(was_training)

    return hypotheses


def main():

    args = docopt(__doc__)
    seed = int(args['--seed'])
    torch.manual_seed(seed)
    if args['--cuda']:
        torch.cuda.manual_seed(seed)
    np.random.seed(seed * 13 // 7)

    if args['train']:
        train(args)
    elif args['decode']:
        decode(args)
    else:
        raise RuntimeError('invalid run mode')


if __name__ == '__main__':
    main()
