#!/usr/bin/env python3

'''
@Time   : 2019-06-16 11:34:23
@Author : su.zhu
@Desc   : 
'''

import torch
from transformers import XLNetModel, XLNetTokenizer

# OPTIONAL: if you want to have more information on what's happening, activate the logger as follows
import logging
logging.basicConfig(level=logging.INFO)

# Load pre-trained model tokenizer (vocabulary)
tokenizer = XLNetTokenizer.from_pretrained('xlnet-base-cased')
model = XLNetModel.from_pretrained('xlnet-base-cased')
input_ids = torch.tensor([tokenizer.encode("Here is some text to encode")])
last_hidden_states = model(input_ids)[0]

print(model.config)
print(last_hidden_states.size())

# Tokenized input
text_a = "Who was Jim Henson ?"
text_b = "Jim Henson was a puppeteer"
tokens_a = tokenizer.tokenize(text_a)
tokens_b = tokenizer.tokenize(text_b)
cls_token='[CLS]'
sep_token='[SEP]'
tokens = tokens_a + ['[SEP]']
segment_ids = [0] * len(tokens)
tokens += tokens_b + ['[SEP]', '[CLS]']
segment_ids += [1] * (len(tokens_b) + 1) + [2]

input_ids = torch.tensor([tokenizer.convert_tokens_to_ids(tokens)])
segment_ids = torch.tensor([segment_ids])

last_hidden_states = model(input_ids, token_type_ids=segment_ids)[0]
print(last_hidden_states.size())
