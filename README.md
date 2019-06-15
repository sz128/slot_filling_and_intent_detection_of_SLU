# Slot filling and intent detection tasks of spoken language understanding
 * An implementation for "focus" part of the paper "Encoder-decoder with focus-mechanism for sequence labelling based spoken language understanding".
 * An implementation of BLSTM-CRF based on [jiesutd/NCRFpp](https://github.com/jiesutd/NCRFpp/blob/master/model/crf.py)
 * An implementation of joint training of slot filling and intent detection tasks [(Bing Liu and Ian Lane, 2016)](https://arxiv.org/abs/1609.01454).
 * Tutorials on [ATIS](https://github.com/yvchen/JointSLU) and [SNIPS](https://github.com/snipsco/nlu-benchmark/tree/master/2017-06-custom-intent-engines) datasets.

## Setup
 * pytorch 1.0
 * python 3.6.x
 * pip install gpustat     [if gpu is used]

## Tutorials A: Slot filling and intent detection with pretrained word embeddings
 1. Pretrained word embeddings are from CNN-BLSTM language models of [ELMo](https://github.com/allenai/allennlp/blob/master/tutorials/how_to/elmo.md) where word embeddings are modelled by char-CNNs. We extract the pretrained word embeddings from atis and snips datasets by:
   * python3 scripts/get_ELMo_word_embedding_for_a_dataset.py --in_files data/atis-2/{train,valid,test} --output_word2vec local/word_embeddings/elmo_1024_cased_for_atis.txt
   * python3 scripts/get_ELMo_word_embedding_for_a_dataset.py --in_files data/snips/{train,valid,test} --output_word2vec local/word_embeddings/elmo_1024_cased_for_snips.txt

 2. Run scripts of training and evaluation at each epoch.
   * BLSTM model: "bash run/atis_with_pretrained_word_embeddings.sh slot_tagger" or "bash run/snips_with_pretrained_word_embeddings.sh slot_tagger"
   * BLSTM-CRF model: "bash run/atis_with_pretrained_word_embeddings.sh slot_tagger_with_crf" or "bash run/snips_with_pretrained_word_embeddings.sh slot_tagger_with_crf"
   * Enc-dec focus model (BLSTM-LSTM), the same as Encoder-Decoder NN (with aligned inputs)(Liu and Lane, 2016): "bash run/atis_with_pretrained_word_embeddings.sh slot_tagger_with_focus" or "bash run/snips_with_pretrained_word_embeddings.sh slot_tagger_with_focus"
 
 3. Results of ATIS:
    
    | models | intent Acc (%) | slot F1-score (%) |
    |:------:|------|-------|
    | [Atten. enc-dec NN with aligned inputs](Liu and Lane, 2016) | 98.43 | 95.87 |
    | [Atten.-BiRNN](Liu and Lane, 2016) | 98.21 | 95.98 |
    | [Enc-dec focus](Zhu and Yu, 2017) | - | 95.79 |
    | [Slot-Gated](Goo et al., 2018) | 94.1 | 95.2 |
    | [BLSTM-CRF + ELMo](https://arxiv.org/abs/1811.05370) | 97.42 | 95.62 |
    | [Joint BERT](https://arxiv.org/pdf/1902.10909.pdf) | 97.5 | 96.1 |
    | [Joint BERT + CRF](https://arxiv.org/pdf/1902.10909.pdf) | 97.9 | 96.0 |
    | [BLSTM](this implementation) | 98.10 | 95.67 |
    | [BLSTM-CRF](this implementation) | 98.54 | 95.39 |
    | [Enc-dec focus](this implementation) | 98.43 | 95.78 |
 
 4. Results of SNIPS:
    
    | models | intent Acc (%) | slot F1-score (%) |
    |:------:|------|-------|
    | [Slot-Gated](Goo et al., 2018) | 97.0 | 88.8 |
    | [BLSTM-CRF + ELMo](https://arxiv.org/abs/1811.05370) | 99.29 | 93.90 |
    | [Joint BERT](https://arxiv.org/pdf/1902.10909.pdf) | 98.6 | 97.0 |
    | [Joint BERT + CRF](https://arxiv.org/pdf/1902.10909.pdf) | 98.4 | 96.7 |
    | [BLSTM](this implementation) | 99.14 | 95.75 |
    | [BLSTM-CRF](this implementation) | 98.71 | 96.22 |
    | [Enc-dec focus](this implementation) | 99.00 | 96.92 |

## Tutorials B: Slot filling and intent detection with ELMo [Todo]

## Tutorials C: Slot filling and intent detection with BERT [Todo]

## Reference
 * Su Zhu and Kai Yu, "Encoder-decoder with focus-mechanism for sequence labelling based spoken language understanding," in IEEE International Conference on Acoustics, Speech and Signal Processing(ICASSP), 2017, pp. 5675-5679.
