"""Slot Tagger models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import models.crf as crf

class LSTMTagger_CRF(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, bidirectional=True, num_layers=1, dropout=0., device=None, extFeats_dim=None, elmo_model=None, bert_model=None):
        """Initialize model."""
        super(LSTMTagger_CRF, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        #self.pad_token_idxs = pad_token_idxs
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device
        self.extFeats_dim = extFeats_dim

        self.num_directions = 2 if self.bidirectional else 1
        
        self.dropout_layer = nn.Dropout(p=self.dropout)

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
        self.lstm = nn.LSTM(self.embedding_dim + self.append_feature_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.num_directions * self.hidden_dim, self.tagset_size + 2)

        self.crf_layer = crf.CRF(self.tagset_size, self.device)
        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        if not self.elmo_model and not self.bert_model:
            self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        if self.extFeats_linear:
            self.extFeats_linear.weight.data.uniform_(-initrange, initrange)
            self.extFeats_linear.bias.data.uniform_(-initrange, initrange)
        for weight in self.lstm.parameters():
            weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.uniform_(-initrange, initrange)

    def _get_lstm_features(self, sentences, lengths, extFeats=None, with_snt_classifier=False):
        # step 1: word embedding
        if self.elmo_model and self.bert_model:
            elmo_embeds = self.elmo_model(sentences['elmo'])
            elmo_embeds = elmo_embeds['elmo_representations'][0]
            tokens, segments, selects, copies, attention_mask = sentences['bert']['tokens'], sentences['bert']['segments'], sentences['bert']['selects'], sentences['bert']['copies'], sentences['bert']['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            bert_embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            bert_embeds = bert_embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
            embeds = torch.cat((elmo_embeds, bert_embeds), dim=2)
        elif self.elmo_model:
            emlo_embeds = self.elmo_model(sentences)
            embeds = emlo_embeds['elmo_representations'][0]
        elif self.bert_model:
            tokens, segments, selects, copies, attention_mask = sentences['tokens'], sentences['segments'], sentences['selects'], sentences['copies'], sentences['mask']
            bert_top_hiddens, _ = self.bert_model(tokens, segments, attention_mask, output_all_encoded_layers=False)
            batch_size, bert_seq_length, hidden_size = bert_top_hiddens.size(0), bert_top_hiddens.size(1), bert_top_hiddens.size(2)
            chosen_encoder_hiddens = bert_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        else:
            embeds = self.word_embeddings(sentences)
        if type(extFeats) != type(None):
            concat_input = torch.cat((embeds, self.extFeats_linear(extFeats)), 2)
        else:
            concat_input = embeds
        concat_input = self.dropout_layer(concat_input)
        
        # step 2: BLSTM encoder
        packed_embeds = rnn_utils.pack_padded_sequence(concat_input, lengths, batch_first=True)
        packed_lstm_out, packed_h_t_c_t = self.lstm(packed_embeds)  # bsize x seqlen x dim
        lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_lstm_out, batch_first=True)

        lstm_out_reshape = lstm_out.contiguous().view(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
        tag_space = self.hidden2tag(self.dropout_layer(lstm_out_reshape))
        tag_space = tag_space.view(lstm_out.size(0), lstm_out.size(1), tag_space.size(1))

        if with_snt_classifier:
            return tag_space, (packed_h_t_c_t, lstm_out, lengths)
        else:
            return tag_space

    def neg_log_likelihood(self, feats, masks, tags):
        return self.crf_layer.neg_log_likelihood_loss(feats, masks, tags)

    def forward(self, feats, masks):
        path_score, best_path = self.crf_layer._viterbi_decode(feats, masks)
        return path_score, best_path

    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

