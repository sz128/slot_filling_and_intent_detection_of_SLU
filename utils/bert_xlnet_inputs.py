#!/usr/bin/env python3

'''
@Time   : 2019-07-18 22:06:40
@Author : su.zhu
@Desc   : 
'''

import torch
import itertools

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
