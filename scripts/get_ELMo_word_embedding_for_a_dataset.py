#!/usr/bin/env python3

'''
@Time   : 2019-06-09 10:38:57
@Author : su.zhu
@Desc   : 
'''

import sys
import re
import argparse

import torch
from allennlp.modules.elmo import Elmo, batch_to_ids, _ElmoBiLm
from allennlp.nn.util import remove_sentence_boundaries

class elmo_embeddings():
    def __init__(self, options_file, weight_file, device=None):
        self._elmo_lstm = _ElmoBiLm(options_file,
                                    weight_file,
                                    requires_grad=False,
                                    vocab_to_cache=None)

        if device is not None:
            self._elmo_lstm = self._elmo_lstm.to(device)

        self.output_dim = self._elmo_lstm.get_output_dim()

    def forward(self, inputs):
        """
        Parameters
        ----------
        inputs: ``torch.Tensor``, required.
        Shape ``(batch_size, timesteps, 50)`` of character ids representing the current batch.
        word_inputs : ``torch.Tensor``, required.
            If you passed a cached vocab, you can in addition pass a tensor of shape
            ``(batch_size, timesteps)``, which represent word ids which have been pre-cached.
        Returns
        -------
        Dict with keys:
        """
        # reshape the input if needed
        original_shape = inputs.size()
        if len(original_shape) > 3:
            timesteps, num_characters = original_shape[-2:]
            reshaped_inputs = inputs.view(-1, timesteps, num_characters)
        else:
            reshaped_inputs = inputs

        # run the biLM
        bilm_output = self._elmo_lstm(reshaped_inputs, None)
        layer_activations = bilm_output['activations']
        mask_with_bos_eos = bilm_output['mask']

        word_embedding_and_hiddens = torch.cat(layer_activations, dim=-1)
        assert self.output_dim * len(layer_activations) == word_embedding_and_hiddens.size(-1)

        # compute the elmo representations
        representation_with_bos_eos = word_embedding_and_hiddens
        representation_without_bos_eos, mask_without_bos_eos = remove_sentence_boundaries(representation_with_bos_eos, mask_with_bos_eos)
        processed_representation = representation_without_bos_eos
        processed_mask = mask_without_bos_eos

        # reshape if necessary
        out_representations = []
        out_representations.append(processed_representation[:, :, :self.output_dim])
        if len(layer_activations) > 1:
            for i in range(1, len(layer_activations)):
                out_representations.append(processed_representation[:, :, self.output_dim * i : self.output_dim * (i + 1)])

        return {'elmo_representations': out_representations, 'mask': processed_mask}

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

    #options_file = "./local/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    #weight_file = "./local/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"
    options_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_options.json"
    weight_file = "https://s3-us-west-2.amazonaws.com/allennlp/models/elmo/2x4096_512_2048cnn_2xhighway/elmo_2x4096_512_2048cnn_2xhighway_weights.hdf5"

    to_get_elmo_embeddings = elmo_embeddings(options_file, weight_file)

    word_vocab = list(word_vocab)
    batch_size = 10
    with open(args.output_word2vec, 'w') as out_file:
        out_file.write(str(len(word_vocab)) + ' ' + str(to_get_elmo_embeddings.output_dim) + '\n')
        for i in range(0, len(word_vocab), batch_size):
            sentences = [[word] for word in word_vocab[i: i + batch_size]]
            character_ids = batch_to_ids(sentences)
            word_embeddings = to_get_elmo_embeddings.forward(character_ids)['elmo_representations'][0]
            for j in range(len(sentences)):
                word_emb = word_embeddings[j][0]
                string = ' '.join([str(float(value.item())) for value in word_emb])
                out_file.write(sentences[j][0] + ' ' + string + '\n')
            if i % 100 == 0:
                print(i)
