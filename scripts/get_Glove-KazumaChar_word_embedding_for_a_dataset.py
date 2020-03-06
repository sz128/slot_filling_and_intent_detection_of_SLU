#!/usr/bin/env python3

'''
@Time   : 2019-06-09 10:38:57
@Author : su.zhu
@Desc   : 
'''

import sys
import re
import argparse

from embeddings import GloveEmbedding, KazumaCharEmbedding

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument('--in_files', type=argparse.FileType('r'), nargs='+')
    parser.add_argument('--output_word2vec', required=True, help='')
    parser.add_argument('--word_lowercase', action='store_true', help='')
    args = parser.parse_args()

    word_vocab = set()
    for f in args.in_files:
        for line in f:
            word_slots, intents = line.strip('\r\n').split(' <=> ')
            for word_slot in word_slots.split(' '):
                tmp = word_slot.split(':')
                word, slot = ':'.join(tmp[:-1]), tmp[-1]
                if args.word_lowercase:
                    word = word.lower()
                word_vocab.add(word)

    word_vocab = list(word_vocab)
    embeddings = [GloveEmbedding(), KazumaCharEmbedding()]
    embed_dim = embeddings[0].d_emb + embeddings[1].d_emb
    with open(args.output_word2vec, 'w') as out_file:
        out_file.write(str(len(word_vocab)) + ' ' + str(embed_dim) + '\n')
        i = 0
        for word in word_vocab:
            i += 1
            e = []
            for emb in embeddings:
                e += emb.emb(word, default='zero')
            string = ' '.join([str(v) for v in e])
            out_file.write(word + ' ' + string + '\n')
            if i % 100 == 0:
                print(i)
