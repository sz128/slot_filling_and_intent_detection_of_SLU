#!/usr/bin/env python3

'''
@Time   : 2020-03-06 17:07:54
@Author : su.zhu
@Desc   : 
'''

import sys
import re
import argparse

import torch
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel

MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer),
        'xlnet': (XLNetModel, XLNetTokenizer),
        }

def load_pretrained_transformer(model_type, model_name):
	pretrained_model_class, tokenizer_class = MODEL_CLASSES[model_type]
	tokenizer = tokenizer_class.from_pretrained(model_name)
	pretrained_model = pretrained_model_class.from_pretrained(model_name)
	print(pretrained_model.config)
	return tokenizer, pretrained_model

def get_bert_token_embeddings(tf_tokenizer, token_embeddings, token):
    tok_id = tf_tokenizer.convert_tokens_to_ids([token])
    return token_embeddings(torch.tensor(tok_id))[0]
def get_bert_token_embeddings_mean(tf_tokenizer, token_embeddings, text):
    tokens = tf_tokenizer.tokenize(text)
    tok_ids = tf_tokenizer.convert_tokens_to_ids(tokens)
    embeds = token_embeddings(torch.tensor(tok_ids))
    return embeds.mean(dim=0)

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_files', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('--output_word2vec', required=True, help='')
    parser.add_argument('--pretrained_tf_type', required=True, help='bert')
    parser.add_argument('--pretrained_tf_name', required=True, help='bert-base-uncased')
    args = parser.parse_args()

    word_vocab = set()
    for f in args.in_files:
        for line in f:
            word_slots, intents = line.strip('\r\n').split(' <=> ')
            for word_slot in word_slots.split(' '):
                tmp = word_slot.split(':')
                word, slot = ':'.join(tmp[:-1]), tmp[-1]
                if 'uncased' in args.pretrained_tf_name:
                    word = word.lower()
                word_vocab.add(word)

    word_vocab = list(word_vocab)
    tf_tokenizer, tf_model = load_pretrained_transformer(args.pretrained_tf_type, args.pretrained_tf_name)
    token_embeddings = tf_model.embeddings.word_embeddings
    print(token_embeddings.weight.shape)
    embed_dim = token_embeddings.weight.shape[1]
    with open(args.output_word2vec, 'w') as out_file:
        out_file.write(str(len(word_vocab)) + ' ' + str(embed_dim) + '\n')
        i = 0
        for word in word_vocab:
            i += 1
            e = get_bert_token_embeddings_mean(tf_tokenizer, token_embeddings, word)
            e = e.data.numpy().tolist()
            string = ' '.join([str(v) for v in e])
            out_file.write(word + ' ' + string + '\n')
            if i % 100 == 0:
                print(i)
