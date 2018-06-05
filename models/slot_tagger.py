"""Slot Tagger models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

class LSTMTagger(nn.Module):
    
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, pad_token_idx, bidirectional=True, num_layers=1, dropout=0., device=None):
        """Initialize model."""
        super(LSTMTagger, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.pad_token_idx = pad_token_idx
        self.bidirectional = bidirectional
        self.num_layers = num_layers
        self.dropout = dropout
        self.device = device

        self.num_directions = 2 if self.bidirectional else 1
        
        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim, self.pad_token_idx)

        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        # The LSTM takes word embeddings as inputs, and outputs hidden states
        # with dimensionality hidden_dim.
        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim, num_layers=self.num_layers, bidirectional=self.bidirectional, batch_first=True, dropout=self.dropout)
        
        # The linear layer that maps from hidden state space to tag space
        self.hidden2tag = nn.Linear(self.num_directions * self.hidden_dim, self.tagset_size)

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        self.word_embeddings.weight.data.uniform_(-initrange, initrange)
        for weight in self.lstm.parameters():
            weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.uniform_(-initrange, initrange)
        
    def init_hidden(self, input):
        # Before we've done anything, we dont have any hidden state.
        # Refer to the Pytorch documentation to see exactly why they have this dimensionality.
        # The axes semantics are (num_layers*num_directions, minibatch_size, hidden_dim)
        minibatch_size = input.size(0) \
                if self.lstm.batch_first else input.size(1)
        h0 = torch.zeros(self.num_layers*self.num_directions, minibatch_size, self.hidden_dim, device=self.device)
        c0 = torch.zeros(self.num_layers*self.num_directions, minibatch_size, self.hidden_dim, device=self.device)
        return (h0, c0)
        
    def forward(self, sentences, lengths):
        h0_c0 = self.init_hidden(sentences)
        embeds = self.dropout_layer(self.word_embeddings(sentences))
        
        packed_embeds = rnn_utils.pack_padded_sequence(embeds, lengths, batch_first=True)
        packed_lstm_out, packed_h_t_c_t = self.lstm(packed_embeds, h0_c0)  # bsize x seqlen x dim
        lstm_out, unpacked_len = rnn_utils.pad_packed_sequence(packed_lstm_out, batch_first=True)

        lstm_out_reshape = lstm_out.contiguous().view(lstm_out.size(0)*lstm_out.size(1), lstm_out.size(2))
        tag_space = self.hidden2tag(self.dropout_layer(lstm_out_reshape))
        tag_scores = F.log_softmax(tag_space, dim=1)
        tag_scores = tag_scores.view(lstm_out.size(0), lstm_out.size(1), tag_space.size(1))
        
        return tag_scores, packed_h_t_c_t
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

