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
 ```sh
   python3 scripts/get_ELMo_word_embedding_for_a_dataset.py \
           --in_files data/atis-2/{train,valid,test} \
           --output_word2vec local/word_embeddings/elmo_1024_cased_for_atis.txt
   python3 scripts/get_ELMo_word_embedding_for_a_dataset.py \
           --in_files data/snips/{train,valid,test} \
           --output_word2vec local/word_embeddings/elmo_1024_cased_for_snips.txt
```

 2. Run scripts of training and evaluation at each epoch.
   * BLSTM model: 
   ```sh
   bash run/atis_with_pretrained_word_embeddings.sh slot_tagger
   bash run/snips_with_pretrained_word_embeddings.sh slot_tagger
   ```
   * BLSTM-CRF model: 
   ```sh
   bash run/atis_with_pretrained_word_embeddings.sh slot_tagger_with_crf
   bash run/snips_with_pretrained_word_embeddings.sh slot_tagger_with_crf
   ```
   * Enc-dec focus model (BLSTM-LSTM), the same as Encoder-Decoder NN (with aligned inputs)(Liu and Lane, 2016): 
   ```sh
   bash run/atis_with_pretrained_word_embeddings.sh slot_tagger_with_focus
   bash run/snips_with_pretrained_word_embeddings.sh slot_tagger_with_focus
   ```

## Tutorials B: Slot filling and intent detection with [ELMo](https://arxiv.org/abs/1802.05365)
 
 1. Run scripts of training and evaluation at each epoch.
   * BLSTM model: 
   ```sh
   bash run/atis_with_elmo.sh slot_tagger
   bash run/snips_with_elmo.sh slot_tagger
   ```
   * BLSTM-CRF model: 
   ```sh
   bash run/atis_with_elmo.sh slot_tagger_with_crf
   bash run/snips_with_elmo.sh slot_tagger_with_crf
   ```
   * Enc-dec focus model (BLSTM-LSTM), the same as Encoder-Decoder NN (with aligned inputs)(Liu and Lane, 2016): 
   ```sh
   bash run/atis_with_elmo.sh slot_tagger_with_focus
   bash run/snips_with_elmo.sh slot_tagger_with_focus
   ```

## Tutorials C: Slot filling and intent detection with [BERT](https://github.com/huggingface/pytorch-pretrained-BERT)

 1. Run scripts of training and evaluation at each epoch.
   * BLSTM model: 
   ```sh
   bash run/atis_with_bert.sh slot_tagger
   bash run/snips_with_bert.sh slot_tagger
   ```
   * BLSTM-CRF model: 
   ```sh
   bash run/atis_with_bert.sh slot_tagger_with_crf
   bash run/snips_with_bert.sh slot_tagger_with_crf
   ```
   * Enc-dec focus model (BLSTM-LSTM), the same as Encoder-Decoder NN (with aligned inputs)(Liu and Lane, 2016): 
   ```sh
   bash run/atis_with_bert.sh slot_tagger_with_focus
   bash run/snips_with_bert.sh slot_tagger_with_focus
   ```

## Tutorials C: Slot filling and intent detection with [XLNET](https://github.com/zihangdai/xlnet) [ToDo]


## Results:

 * For "NLU + BERT" model, hyper-parameters are not tuned carefully.
 
 1. Results of ATIS:
    
    | models | intent Acc (%) | slot F1-score (%) |
    |:------:|------|-------|
    | [Atten. enc-dec NN with aligned inputs](Liu and Lane, 2016) | 98.43 | 95.87 |
    | [Atten.-BiRNN](Liu and Lane, 2016) | 98.21 | 95.98 |
    | [Enc-dec focus](Zhu and Yu, 2017) | - | 95.79 |
    | [Slot-Gated](Goo et al., 2018) | 94.1 | 95.2 |
    | [Intent Gating & self-attention](https://www.aclweb.org/anthology/D18-1417) | 98.77 | 96.52 |
    | [BLSTM-CRF + ELMo](https://arxiv.org/abs/1811.05370) | 97.42 | 95.62 |
    | [Joint BERT](https://arxiv.org/pdf/1902.10909.pdf) | 97.5 | 96.1 |
    | [Joint BERT + CRF](https://arxiv.org/pdf/1902.10909.pdf) | 97.9 | 96.0 |
    | BLSTM (A. Pre-train word emb.) | 98.10 | 95.67 |
    | BLSTM-CRF (A. Pre-train word emb.) | 98.54 | 95.39 |
    | Enc-dec focus (A. Pre-train word emb.) | 98.43 | 95.78 |
    | BLSTM (B. +ELMo) | 98.66 | 95.52 |
    | BLSTM-CRF (B. +ELMo) | 98.32 | 95.62 |
    | Enc-dec focus (B. +ELMo) | 98.66 | 95.70 |
    | BLSTM (C. +BERT) | 99.10 | 95.94 |
 
 2. Results of SNIPS:
    
    | models | intent Acc (%) | slot F1-score (%) |
    |:------:|------|-------|
    | [Slot-Gated](Goo et al., 2018) | 97.0 | 88.8 |
    | [BLSTM-CRF + ELMo](https://arxiv.org/abs/1811.05370) | 99.29 | 93.90 |
    | [Joint BERT](https://arxiv.org/pdf/1902.10909.pdf) | 98.6 | 97.0 |
    | [Joint BERT + CRF](https://arxiv.org/pdf/1902.10909.pdf) | 98.4 | 96.7 |
    | BLSTM (A. Pre-train word emb.) | 99.14 | 95.75 |
    | BLSTM-CRF (A. Pre-train word emb.) | 99.00 | 96.92 |
    | Enc-dec focus (A. Pre-train word emb.) | 98.71 | 96.22 |
    | BLSTM (B. +ELMo) | 98.71 | 96.32 |
    | BLSTM-CRF (B. +ELMo) | 98.57 | 96.61 |
    | Enc-dec focus (B. +ELMo) | 99.14 | 96.69 |
    | BLSTM (C. +BERT) | 99.00 | 96.23 | 

## Reference
 * Su Zhu and Kai Yu, "Encoder-decoder with focus-mechanism for sequence labelling based spoken language understanding," in IEEE International Conference on Acoustics, Speech and Signal Processing(ICASSP), 2017, pp. 5675-5679.
