#!/bin/bash

source ./path.sh

task_slot_filling=slot_tagger_with_focus #slot_tagger, slot_tagger_with_crf, slot_tagger_with_focus
task_intent_detection=none # none, hiddenAttention, hiddenCNN, maxPooling, 2tails
balance_weight=1

#cased_word_vectors=./local/word_embeddings/elmo_1024_cased_for_MIT_corpus.txt
cased_word_vectors=./local/word_embeddings/glove-kazumachar_400_cased_for_MIT_corpus.txt
read_word2vec_inText=${cased_word_vectors}
word_lowercase=false
fix_word2vec_inText=false
word_digit_features=false #false, true

dataroot=data/MIT_corpus/movie_eng #movie_eng, movie_trivia10k13, restaurant
dataset=mit_movie_eng #mit_movie_eng, mit_movie_trivia10k13, mit_restaurant

word_embedding_size=400 #1024
lstm_hidden_size=200
lstm_layers=2
slot_tag_embedding_size=100  ## for slot_tagger_with_focus
batch_size=20

optimizer=adam
learning_rate=0.001
max_norm_of_gradient_clip=5
dropout_rate=0.5

max_epoch=20

device=0
# device=0 means auto-choosing a GPU
# Set deviceId=-1 if you are going to use cpu for training.
experiment_output_path=exp

source ./utils/parse_options.sh

if [[ $word_lowercase != true && $word_lowercase != True ]]; then
  unset word_lowercase
fi
if [[ $fix_word2vec_inText != true && $fix_word2vec_inText != True ]]; then
  unset fix_word2vec_inText
fi
if [[ $word_digit_features != true && $word_digit_features != True ]]; then
  unset word_digit_features
fi

python scripts/slot_tagging_and_intent_detection.py --task_st $task_slot_filling --task_sc $task_intent_detection --dataset $dataset --dataroot $dataroot --bidirectional --lr $learning_rate --dropout $dropout_rate --batchSize $batch_size --optim $optimizer --max_norm $max_norm_of_gradient_clip --experiment $experiment_output_path --deviceId $device --max_epoch $max_epoch --emb_size $word_embedding_size --hidden_size $lstm_hidden_size --num_layers ${lstm_layers} --tag_emb_size $slot_tag_embedding_size --st_weight ${balance_weight} ${word_lowercase:+--word_lowercase} ${read_word2vec_inText:+--read_input_word2vec ${read_word2vec_inText}} ${fix_word2vec_inText:+--fix_input_word2vec} ${word_digit_features:+--word_digit_features}
