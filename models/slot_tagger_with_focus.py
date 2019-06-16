"""Slot Tagger models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

from models.Beam import Beam

class LSTMTagger_focus(nn.Module):
    
    def __init__(self, embedding_dim, tag_embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=True, num_layers=1, dropout=0., device=None, extFeats_dim=None, decoder_tied=False, elmo_model=None, bert_model=None):
        """Initialize model."""
        super(LSTMTagger_focus, self).__init__()
        self.embedding_dim = embedding_dim
        self.tag_embedding_dim = tag_embedding_dim
        self.extFeats_dim = extFeats_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        #self.pad_token_idxs = pad_token_idxs
        #self.pad_tag_idxs = pad_tag_idxs
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.decoder_tied = decoder_tied

        self.num_directions = 2 if self.bidirectional else 1
        self.dropout_layer = nn.Dropout(p=self.dropout)
        self.tag_embeddings = nn.Embedding(self.tagset_size, self.tag_embedding_dim)
        
        self.elmo_model = elmo_model
        self.bert_model = bert_model
        if self.elmo_model and self.bert_model:
            self.embedding_dim = self.elmo_model.get_output_dim() + self.bert_model.config.hidden_size
        elif self.elmo_model:
            self.embedding_dim = self.elmo_model.get_output_dim()
        elif self.bert_model:
            self.embedding_dim = self.bert_model.config.hidden_size
        else:
            self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.append_feature_dim = 0
        if self.extFeats_dim:
            self.append_feature_dim += self.extFeats_dim
            self.extFeats_linear = nn.Linear(self.append_feature_dim, self.append_feature_dim)
        else:
            self.extFeats_linear = None
        # with dimensionality hidden_dim.
        self.encoder = nn.LSTM(self.embedding_dim + self.append_feature_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        
        self.decoder = nn.LSTM(self.tag_embedding_dim + self.num_directions * self.hidden_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=False, batch_first=True, dropout=self.dropout)
        
        # The linear layer that maps from hidden state space to tag space  # self.num_directions * self.hidden_dim
        self.hidden2tag = nn.Linear(1 * self.hidden_dim, self.tagset_size)
        if self.decoder_tied:
            self.hidden2tag.weight = self.tag_embeddings.weight

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        if not self.elmo_model and not self.bert_model:
            self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        #for pad_token_idx in self.pad_token_idxs:
        #    self.word_embeddings.weight.data[pad_token_idx].zero_()
        if self.extFeats_linear:
            self.extFeats_linear.weight.data.uniform_(-initrange, initrange)
            self.extFeats_linear.bias.data.uniform_(-initrange, initrange)
        for weight in self.encoder.parameters():
            weight.data.uniform_(-initrange, initrange)

        self.tag_embeddings.weight.data.uniform_(-initrange, initrange)
        for weight in self.decoder.parameters():
            weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.uniform_(-initrange, initrange)
    
    def forward(self, word_seqs, tag_seqs, lengths, extFeats=None, with_snt_classifier=False, masked_output=None):
        # encoder
        if self.elmo_model and self.bert_model:
            elmo_embeds = self.elmo_model(word_seqs['elmo'])
            elmo_embeds = elmo_embeds['elmo_representations'][0]
            tokens, segments, selects, copies, attention_mask = word_seqs['bert']['tokens'], word_seqs['bert']['segments'], word_seqs['bert']['selects'], word_seqs['bert']['copies'], word_seqs['bert']['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            bert_embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            bert_embeds = bert_embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
            embeds = torch.cat((elmo_embeds, bert_embeds), dim=2)
        elif self.elmo_model:
            emlo_embeds = self.elmo_model(word_seqs)
            embeds = emlo_embeds['elmo_representations'][0]
        elif self.bert_model:
            tokens, segments, selects, copies, attention_mask = word_seqs['tokens'], word_seqs['segments'], word_seqs['selects'], word_seqs['copies'], word_seqs['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        else:
            embeds = self.word_embeddings(word_seqs)
        if type(extFeats) != type(None):
            concat_input = torch.cat((embeds, self.extFeats_linear(extFeats)), 2)
        else:
            concat_input = embeds
        concat_input = self.dropout_layer(concat_input)
        packed_word_embeds = rnn_utils.pack_padded_sequence(concat_input, lengths, batch_first=True)
        packed_word_lstm_out, (enc_h_t, enc_c_t) = self.encoder(packed_word_embeds)  # bsize x seqlen x dim
        word_lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_word_lstm_out, batch_first=True)

        # decoder
        if self.bidirectional:
            index_slices = [2*i+1 for i in range(self.num_layers)]  # generated from the reversed path
            index_slices = torch.tensor(index_slices, dtype=torch.long, device=self.device)
            h_t = torch.index_select(enc_h_t, 0, index_slices)
            c_t = torch.index_select(enc_c_t, 0, index_slices)
        else:
            h_t = enc_h_t
            c_t = enc_c_t
        tag_embeds = self.dropout_layer(self.tag_embeddings(tag_seqs))
        decode_inputs = torch.cat((self.dropout_layer(word_lstm_out), tag_embeds), 2)
        packed_decode_inputs = rnn_utils.pack_padded_sequence(decode_inputs, lengths, batch_first=True)
        packed_tag_lstm_out, (dec_h_t, dec_c_t) = self.decoder(packed_decode_inputs, (h_t, c_t))  # bsize x seqlen x dim
        tag_lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_tag_lstm_out, batch_first=True)

        tag_lstm_out_reshape = tag_lstm_out.contiguous().view(tag_lstm_out.size(0)*tag_lstm_out.size(1), tag_lstm_out.size(2))
        tag_space = self.hidden2tag(self.dropout_layer(tag_lstm_out_reshape))
        if masked_output is None:
            tag_scores = F.log_softmax(tag_space, dim=1)
        else:
            tag_scores = masked_function.index_masked_log_softmax(tag_space, masked_output, dim=1)
        tag_scores = tag_scores.view(tag_lstm_out.size(0), tag_lstm_out.size(1), tag_space.size(1))
        
        if with_snt_classifier:
            return tag_scores, ((enc_h_t, enc_c_t), word_lstm_out, lengths)
        else:
            return tag_scores
    
    def decode_greed(self, word_seqs, init_tags, lengths, extFeats=None, with_snt_classifier=False, masked_output=None):
        minibatch_size = len(lengths) #word_seqs.size(0) if self.encoder.batch_first else word_seqs.size(1)
        max_length = max(lengths) #word_seqs.size(1) if self.encoder.batch_first else word_seqs.size(0)
        # encoder
        if self.elmo_model and self.bert_model:
            elmo_embeds = self.elmo_model(word_seqs['elmo'])
            elmo_embeds = elmo_embeds['elmo_representations'][0]
            tokens, segments, selects, copies, attention_mask = word_seqs['bert']['tokens'], word_seqs['bert']['segments'], word_seqs['bert']['selects'], word_seqs['bert']['copies'], word_seqs['bert']['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            bert_embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            bert_embeds = bert_embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
            embeds = torch.cat((elmo_embeds, bert_embeds), dim=2)
        elif self.elmo_model:
            emlo_embeds = self.elmo_model(word_seqs)
            embeds = emlo_embeds['elmo_representations'][0]
        elif self.bert_model:
            tokens, segments, selects, copies, attention_mask = word_seqs['tokens'], word_seqs['segments'], word_seqs['selects'], word_seqs['copies'], word_seqs['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        else:
            embeds = self.word_embeddings(word_seqs)
        if type(extFeats) != type(None):
            concat_input = torch.cat((embeds, self.extFeats_linear(extFeats)), 2)
        else:
            concat_input = embeds
        concat_input = self.dropout_layer(concat_input)
        packed_word_embeds = rnn_utils.pack_padded_sequence(concat_input, lengths, batch_first=True)
        packed_word_lstm_out, (enc_h_t, enc_c_t) = self.encoder(packed_word_embeds)  # bsize x seqlen x dim
        word_lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_word_lstm_out, batch_first=True)

        # decoder
        if self.bidirectional:
            index_slices = [2*i+1 for i in range(self.num_layers)]  # generated from the reversed path
            index_slices = torch.tensor(index_slices, dtype=torch.long, device=self.device)
            h_t = torch.index_select(enc_h_t, 0, index_slices)
            c_t = torch.index_select(enc_c_t, 0, index_slices)
        else:
            h_t = enc_h_t
            c_t = enc_c_t
        
        top_path = []
        top_path_tag_scores = []
        top_dec_h_t, top_dec_c_t = [0]*minibatch_size, [0]*minibatch_size
        last_tags = init_tags # bsize x 1
        for i in range(max_length):
            tag_embeds = self.dropout_layer(self.tag_embeddings(last_tags))
            decode_inputs = torch.cat((self.dropout_layer(word_lstm_out[:, i:i+1]), tag_embeds), 2) # bsize x 1 x insize
            tag_lstm_out, (dec_h_t, dec_c_t) = self.decoder(decode_inputs, (h_t, c_t)) # bsize x 1 x insize => bsize x 1 x hsize

            for j in range(minibatch_size):
                if lengths[j] == i + 1:
                    top_dec_h_t[j] = dec_h_t[:, j:j+1, :]
                    top_dec_c_t[j] = dec_c_t[:, j:j+1, :]

            tag_lstm_out_reshape = tag_lstm_out.contiguous().view(tag_lstm_out.size(0)*tag_lstm_out.size(1), tag_lstm_out.size(2))
            tag_space = self.hidden2tag(self.dropout_layer(tag_lstm_out_reshape))
            if masked_output is None:
                tag_scores = F.log_softmax(tag_space, dim=1) # bsize x outsize
            else:
                tag_scores = masked_function.index_masked_log_softmax(tag_space, masked_output, dim=1)
            top_path_tag_scores.append(torch.unsqueeze(tag_scores.data, 1))

            max_probs, decoder_argmax = torch.max(tag_scores, 1)
            last_tags = decoder_argmax
            if len(last_tags.size()) == 1:
                last_tags = last_tags.unsqueeze(1)
            h_t, c_t = dec_h_t, dec_c_t
            top_path.append(last_tags.data)
        top_path = torch.cat(top_path, 1)
        top_path_tag_scores = torch.cat(top_path_tag_scores, 1)

        top_dec_h_t = torch.cat(top_dec_h_t, 1)
        top_dec_c_t = torch.cat(top_dec_c_t, 1)
        
        if with_snt_classifier:
            return top_path_tag_scores, top_path, ((enc_h_t, enc_c_t), word_lstm_out, lengths)
        else:
            return top_path_tag_scores, top_path
    
    def decode_beam_search(self, word_seqs, lengths, beam_size, tag2idx, extFeats=None, with_snt_classifier=False, masked_output=None):
        minibatch_size = len(lengths) #word_seqs.size(0) if self.encoder.batch_first else word_seqs.size(1)
        max_length = max(lengths) #word_seqs.size(1) if self.encoder.batch_first else word_seqs.size(0)
        # encoder
        if self.elmo_model and self.bert_model:
            elmo_embeds = self.elmo_model(word_seqs['elmo'])
            elmo_embeds = elmo_embeds['elmo_representations'][0]
            tokens, segments, selects, copies, attention_mask = word_seqs['bert']['tokens'], word_seqs['bert']['segments'], word_seqs['bert']['selects'], word_seqs['bert']['copies'], word_seqs['bert']['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            bert_embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            bert_embeds = bert_embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
            embeds = torch.cat((elmo_embeds, bert_embeds), dim=2)
        elif self.elmo_model:
            emlo_embeds = self.elmo_model(word_seqs)
            embeds = emlo_embeds['elmo_representations'][0]
        elif self.bert_model:
            tokens, segments, selects, copies, attention_mask = word_seqs['tokens'], word_seqs['segments'], word_seqs['selects'], word_seqs['copies'], word_seqs['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        else:
            embeds = self.word_embeddings(word_seqs)
        if type(extFeats) != type(None):
            concat_input = torch.cat((embeds, self.extFeats_linear(extFeats)), 2)
        else:
            concat_input = embeds
        concat_input = self.dropout_layer(concat_input)
        packed_word_embeds = rnn_utils.pack_padded_sequence(concat_input, lengths, batch_first=True)
        packed_word_lstm_out, (enc_h_t, enc_c_t) = self.encoder(packed_word_embeds)  # bsize x seqlen x dim
        enc_word_lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_word_lstm_out, batch_first=True)

        # decoder
        if self.bidirectional:
            index_slices = [2*i+1 for i in range(self.num_layers)]  # generated from the reversed path
            index_slices = torch.tensor(index_slices, dtype=torch.long, device=self.device)
            h_t = torch.index_select(enc_h_t, 0, index_slices)
            c_t = torch.index_select(enc_c_t, 0, index_slices)
        else:
            h_t = enc_h_t
            c_t = enc_c_t
        
        h_t = h_t.repeat(1, beam_size, 1)
        c_t = c_t.repeat(1, beam_size, 1)
        word_lstm_out = enc_word_lstm_out.repeat(beam_size, 1, 1)

        beam = [Beam(beam_size, tag2idx, device=self.device) for k in range(minibatch_size)]
        batch_idx = list(range(minibatch_size))
        remaining_sents = minibatch_size

        top_dec_h_t, top_dec_c_t = [0]*minibatch_size, [0]*minibatch_size
        for i in range(max_length):
            last_tags = torch.stack([b.get_current_state() for b in beam if not b.done]).t().contiguous().view(-1, 1)  # after t() -> beam_size * batch_size
            last_tags = last_tags.to(self.device)
            tag_embeds = self.dropout_layer(self.tag_embeddings(last_tags))
            decode_inputs = torch.cat((self.dropout_layer(word_lstm_out[:, i:i+1]), tag_embeds), 2) # (batch*beam) x 1 x insize
            tag_lstm_out, (dec_h_t, dec_c_t) = self.decoder(decode_inputs, (h_t, c_t)) # (batch*beam) x 1 x insize => (batch*beam) x 1 x hsize

            tag_lstm_out_reshape = tag_lstm_out.contiguous().view(tag_lstm_out.size(0)*tag_lstm_out.size(1), tag_lstm_out.size(2))
            tag_space = self.hidden2tag(self.dropout_layer(tag_lstm_out_reshape))
            if masked_output is None:
                out = F.log_softmax(tag_space) # (batch*beam) x outsize
            else:
                out = masked_function.index_masked_log_softmax(tag_space, masked_output, dim=1)
            
            word_lk = out.view(beam_size, remaining_sents, -1).transpose(0, 1).contiguous()
            
            active = []
            for b in range(minibatch_size):
                if beam[b].done:
                    continue
                if lengths[b] == i + 1:
                    beam[b].done = True
                    top_dec_h_t[b] = dec_h_t[:, b:b+beam_size, :]
                    top_dec_c_t[b] = dec_c_t[:, b:b+beam_size, :]
                idx = batch_idx[b]
                beam[b].advance(word_lk.data[idx])
                if not beam[b].done:
                    active.append(b)
                for dec_state in (dec_h_t, dec_c_t):
                    # (layer*direction) x beam*sent x Hdim
                    sent_states = dec_state.view(-1, beam_size, remaining_sents, dec_state.size(2))[:, :, idx]
                    sent_states.data.copy_(sent_states.data.index_select(1, beam[b].get_current_origin()))
            if not active:
                break

            active_idx = torch.tensor([batch_idx[k] for k in active], dtype=torch.long, device=self.device)
            batch_idx = {beam:idx for idx, beam in enumerate(active)}
            
            def update_active(t, hidden_dim):
                #t_reshape = t.data.view(-1, remaining_sents, hidden_dim)
                t_reshape = t.contiguous().view(-1, remaining_sents, hidden_dim)
                new_size = list(t.size())
                new_size[-2] = new_size[-2] * len(active_idx) // remaining_sents  # beam*len(active_idx)
                return t_reshape.index_select(1, active_idx).view(*new_size)

            h_t = update_active(dec_h_t, self.hidden_dim)
            c_t = update_active(dec_c_t, self.hidden_dim)
            word_lstm_out = update_active(word_lstm_out.transpose(0, 1), self.num_directions * self.hidden_dim).transpose(0, 1)

            remaining_sents = len(active)
        
        allHyp, allScores = [], []
        n_best = 1
        for b in range(minibatch_size):
            scores, ks = beam[b].sort_best()
            allScores += [scores[:n_best]]
            hyps = zip(*[beam[b].get_hyp(k) for k in ks[:n_best]])
            allHyp += [hyps]
            top_dec_h_t[b] = top_dec_h_t[b].data.index_select(1, ks[:n_best])
            top_dec_c_t[b] = top_dec_c_t[b].data.index_select(1, ks[:n_best])
        top_dec_h_t = torch.cat(top_dec_h_t, 1)
        top_dec_c_t = torch.cat(top_dec_c_t, 1)
        allScores = torch.cat(allScores) 

        if with_snt_classifier:
            return allScores, allHyp, ((enc_h_t, enc_c_t), enc_word_lstm_out, lengths)
        else:
            return allScores, allHyp
        
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

