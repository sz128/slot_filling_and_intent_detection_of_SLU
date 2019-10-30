#!/usr/bin/env python3

'''
@Time   : 2019-10-28 21:57:36
@Author : su.zhu
@Desc   : 
'''

import torch
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel 

import itertools

MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer),
        'xlnet': (XLNetModel, XLNetTokenizer),
        }

def prepare_inputs_for_bert_xlnet(sentences, word_lengths, tokenizer, cls_token_at_end=False, pad_on_left=False, cls_token='[CLS]', sep_token='[SEP]', pad_token=0, sequence_a_segment_id=0, cls_token_segment_id=1, pad_token_segment_id=0, device=None):
    """ Loads a data file into a list of `InputBatch`s
        `cls_token_at_end` define the location of the CLS token:
            - False (Default, BERT/XLM pattern): [CLS] + A + [SEP] + B + [SEP]
            - True (XLNet/GPT pattern): A + [SEP] + B + [SEP] + [CLS]
        `cls_token_segment_id` define the segment id associated to the CLS token (0 for BERT, 2 for XLNet)
    """
    """ output: {
        'tokens': tokens_tensor,        # input_ids
        'segments': segments_tensor,    # token_type_ids
        'mask': input_mask,             # attention_mask
        'selects': selects_tensor,      # original_word_to_token_position
        'copies': copies_tensor         # original_word_position
        }
    """
    ## sentences are sorted by sentence length
    max_length_of_sentences = max(word_lengths)
    tokens = []
    segment_ids = []
    selected_indexes = []
    start_pos = 0
    for ws in sentences:
        selected_index = []
        ts = []
        for w in ws:
            if cls_token_at_end:
                selected_index.append(len(ts))
            else:
                selected_index.append(len(ts) + 1)
            ts += tokenizer.tokenize(w)
        ts += [sep_token]
        si = [sequence_a_segment_id] * len(ts)
        if cls_token_at_end:
            ts = ts + [cls_token]
            si = si + [cls_token_segment_id]
        else:
            ts = [cls_token] + ts
            si = [cls_token_segment_id] + si
        tokens.append(ts)
        segment_ids.append(si)
        selected_indexes.append(selected_index)
    max_length_of_tokens = max([len(tokenized_text) for tokenized_text in tokens])
    #if not cls_token_at_end: # bert
    #    assert max_length_of_tokens <= model_bert.config.max_position_embeddings
    padding_lengths = [max_length_of_tokens - len(tokenized_text) for tokenized_text in tokens]
    if pad_on_left:
        input_mask = [[0] * padding_lengths[idx] + [1] * len(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [[pad_token] * padding_lengths[idx] + tokenizer.convert_tokens_to_ids(tokenized_text) for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [[pad_token_segment_id] * padding_lengths[idx] + si for idx,si in enumerate(segment_ids)]
        selected_indexes = [[padding_lengths[idx] + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    else:
        input_mask = [[1] * len(tokenized_text) + [0] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        indexed_tokens = [tokenizer.convert_tokens_to_ids(tokenized_text) + [pad_token] * padding_lengths[idx] for idx,tokenized_text in enumerate(tokens)]
        segments_ids = [si + [pad_token_segment_id] * padding_lengths[idx] for idx,si in enumerate(segment_ids)]
        selected_indexes = [[0 + i + idx * max_length_of_tokens for i in selected_index] for idx,selected_index in enumerate(selected_indexes)]
    copied_indexes = [[i + idx * max_length_of_sentences for i in range(length)] for idx,length in enumerate(word_lengths)]

    input_mask = torch.tensor(input_mask, dtype=torch.long, device=device)
    tokens_tensor = torch.tensor(indexed_tokens, dtype=torch.long, device=device)
    segments_tensor = torch.tensor(segments_ids, dtype=torch.long, device=device)
    selects_tensor = torch.tensor(list(itertools.chain.from_iterable(selected_indexes)), dtype=torch.long, device=device)
    copies_tensor = torch.tensor(list(itertools.chain.from_iterable(copied_indexes)), dtype=torch.long, device=device)
    return {'tokens': tokens_tensor, 'segments': segments_tensor, 'selects': selects_tensor, 'copies': copies_tensor, 'mask': input_mask}

class bert_embeddings():
    def __init__(self, model_type='bert', model_name='bert-base-cased', device=None):
        self.model_type = model_type
        self.model_name = model_name
        self.device = device

        pretrained_model_class, tokenizer_class = MODEL_CLASSES[model_type]
        # Load pre-trained model tokenizer (vocabulary)
        self.tokenizer = tokenizer_class.from_pretrained(model_name)
        # Load pre-trained model (weights)
        self.model = pretrained_model_class.from_pretrained(model_name)
        self.model.eval()
        if device is not None:
            self.model.to(device)

    def forward(self, words):
        # Predict hidden states features for each layer
        lengths = [len(s) for s in words]
        with torch.no_grad():
            inputs = prepare_inputs_for_bert_xlnet(words, lengths, self.tokenizer, 
                    cls_token_at_end=bool(self.model_type in ['xlnet']),  # xlnet has a cls token at the end
                    cls_token=self.tokenizer.cls_token,
                    sep_token=self.tokenizer.sep_token,
                    cls_token_segment_id=2 if self.model_type in ['xlnet'] else 0,
                    pad_on_left=bool(self.model_type in ['xlnet']), # pad on the left for xlnet
                    pad_token_segment_id=4 if self.model_type in ['xlnet'] else 0,
                    device=self.device)
            tokens, segments, selects, copies, attention_mask = inputs['tokens'], inputs['segments'], inputs['selects'], inputs['copies'], inputs['mask']
            outputs = self.model(tokens, token_type_ids=segments, attention_mask=attention_mask)
            pretrained_top_hiddens = outputs[0]
            batch_size, pretrained_seq_length, hidden_size = pretrained_top_hiddens.size(0), pretrained_top_hiddens.size(1), pretrained_top_hiddens.size(2)
            chosen_encoder_hiddens = pretrained_top_hiddens.view(-1, hidden_size).index_select(0, selects)
            embeds = torch.zeros(len(lengths) * max(lengths), hidden_size, device=self.device)
            embeds = embeds.index_copy_(0, copies, chosen_encoder_hiddens).view(len(lengths), max(lengths), -1)
        return embeds

if __name__ == "__main__":

    to_get_bert_embeddings = bert_embeddings()
    
    # use batch_to_ids to convert sentences to character ids
    sentences = [['first', 'sentence', '.'], ['first', 'sentence', 'aa', '.'], ['Another', '.']]
    #print(character_ids)

    embeddings = to_get_bert_embeddings.forward(sentences)

    print(embeddings)
    print(embeddings.size())
