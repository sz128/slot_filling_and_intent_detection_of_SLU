"""Slot Tagger models."""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn_utils

import models.crf as crf

from transformers.modeling_utils import SequenceSummary 

class Transformers_joint_slot_and_intent(nn.Module):
    
    def __init__(self, pretrained_model_type, pretrained_model, tagset_size, class_size, dropout=0., device=None, extFeats_dim=None, multi_class=False, task_st='NN', task_sc='CLS'):
        """Initialize model."""
        super(Transformers_joint_slot_and_intent, self).__init__()
        self.tagset_size = tagset_size
        self.class_size = class_size
        self.dropout = dropout
        self.device = device
        self.extFeats_dim = extFeats_dim
        self.multi_class = multi_class
        self.task_st = task_st # 'NN', 'NN_crf'
        self.task_sc = task_sc # None, 'CLS', 'max', 'CLS_max'

        self.dropout_layer = nn.Dropout(p=self.dropout)
        
        self.pretrained_model_type = pretrained_model_type
        self.pretrained_model = pretrained_model
        if self.pretrained_model_type == 'xlnet':
            self.sequence_summary = SequenceSummary(self.pretrained_model.config)
        self.embedding_dim = self.pretrained_model.config.hidden_size

        # The LSTM takes word embeddings as inputs, and outputs hidden states
        self.append_feature_dim = 0
        if self.extFeats_dim:
            self.append_feature_dim += self.extFeats_dim
            self.extFeats_linear = nn.Linear(self.append_feature_dim, self.append_feature_dim)
        else:
            self.extFeats_linear = None

        # The linear layer that maps from hidden state space to tag space
        if self.task_st == 'NN':
            self.hidden2tag = nn.Linear(self.embedding_dim + self.append_feature_dim, self.tagset_size)
        else:
            self.hidden2tag = nn.Linear(self.embedding_dim + self.append_feature_dim, self.tagset_size + 2)
            self.crf_layer = crf.CRF(self.tagset_size, self.device)
        if self.task_sc == 'CLS' or self.task_sc == 'max':
            self.hidden2class = nn.Linear(self.embedding_dim, self.class_size)
        elif self.task_sc == 'CLS_max':
            self.hidden2class = nn.Linear(self.embedding_dim * 2, self.class_size)
        else:
            pass

        #self.init_weights()

    def init_weights(self, initrange=0.2):
        """Initialize weights."""
        if self.extFeats_linear:
            self.extFeats_linear.weight.data.uniform_(-initrange, initrange)
            self.extFeats_linear.bias.data.uniform_(-initrange, initrange)
        self.hidden2tag.weight.data.uniform_(-initrange, initrange)
        self.hidden2tag.bias.data.uniform_(-initrange, initrange)
        if self.task_sc:
            self.hidden2class.weight.data.uniform_(-initrange, initrange)
            self.hidden2class.bias.data.uniform_(-initrange, initrange)
    
    def forward(self, sentences, lengths, extFeats=None, masked_output=None):
        # step 1: word embedding
        tokens, segments, selects, copies, attention_mask = sentences['tokens'], sentences['segments'], sentences['selects'], sentences['copies'], sentences['mask']
        outputs = self.pretrained_model(tokens, token_type_ids=segments, attention_mask=attention_mask)
        if self.pretrained_model_type == 'bert':
            transformer_top_hiddens, transformer_cls_hidden = outputs[0:2]
        else:
            transformer_top_hiddens = outputs[0]
            transformer_cls_hidden = self.sequence_summary(transformer_top_hiddens)
        batch_size, transformer_seq_length, hidden_size = transformer_top_hiddens.size(0), transformer_top_hiddens.size(1), transformer_top_hiddens.size(2)
        chosen_encoder_hiddens = transformer_top_hiddens.view(-1, hidden_size).index_select(0, selects)
        embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
        embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        if type(extFeats) != type(None):
            concat_input = torch.cat((embeds, self.extFeats_linear(extFeats)), 2)
        else:
            concat_input = embeds
        
        # step 2: slot tagger
        concat_input_reshape = concat_input.contiguous().view(concat_input.size(0)*concat_input.size(1), concat_input.size(2))
        tag_space = self.hidden2tag(self.dropout_layer(concat_input_reshape))
        if self.task_st == 'NN':
            tag_scores = F.log_softmax(tag_space, dim=1)
        else:
            tag_scores = tag_space
        tag_scores = tag_scores.view(concat_input.size(0), concat_input.size(1), tag_space.size(1))
        
        # step 3: intent classifier
        if self.task_sc:
            if self.task_sc == 'CLS':
                hidden_for_intent = transformer_cls_hidden
            elif self.task_sc == 'max':
                hidden_for_intent = embeds.max(1)[0]
            else:
                hidden_for_intent = torch.cat((transformer_cls_hidden, embeds.max(1)[0]), dim=1)
            class_space = self.hidden2class(self.dropout_layer(transformer_cls_hidden))
            if self.multi_class:
                class_scores = torch.sigmoid(class_space)
                if type(masked_output) != type(None):
                    class_scores.index_fill_(1, masked_output, 0)
            else:
                class_scores = F.log_softmax(class_space, dim=1)
        else:
            class_scores = None
        
        return tag_scores, class_scores

    def crf_neg_log_likelihood(self, tag_scores, masks, tags):
        return self.crf_layer.neg_log_likelihood_loss(tag_scores, masks, tags)

    def crf_viterbi_decode(self, tag_scores, masks):
        path_score, best_path = self.crf_layer._viterbi_decode(tag_scores, masks)
        return path_score, best_path
    
    def load_model(self, load_dir):
        if self.device.type == 'cuda':
            self.load_state_dict(torch.load(open(load_dir, 'rb')))
        else:
            self.load_state_dict(torch.load(open(load_dir, 'rb'), map_location=lambda storage, loc: storage))

    def save_model(self, save_dir):
        torch.save(self.state_dict(), open(save_dir, 'wb'))

