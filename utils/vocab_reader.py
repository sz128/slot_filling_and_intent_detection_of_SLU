"""Data utilities."""
#import torch
import operator
#import json

def read_vocab_file(vocab_path, bos_eos=False, no_pad=False, no_unk=False, separator=':'):
    '''file format: "word : idx" '''
    word2id, id2word = {}, {}
    if not no_pad:
        word2id['<pad>'] = len(word2id)
        id2word[len(id2word)] = '<pad>'
    if not no_unk:
        word2id['<unk>'] = len(word2id)
        id2word[len(id2word)] = '<unk>'
    if bos_eos == True:
        word2id['<s>'] = len(word2id)
        id2word[len(id2word)] = '<s>'
        word2id['</s>'] = len(word2id)
        id2word[len(id2word)] = '</s>'
    with open(vocab_path, 'r') as f:
        for line in f:
            if separator in line:
                word, idx = line.strip('\r\n').split(' '+separator+' ')
                idx = int(idx)
            else:
                word = line.strip()
                idx = len(word2id)
            if word not in word2id:
                word2id[word] = idx
                id2word[idx] = word
    return word2id, id2word

def save_vocab(idx2word, vocab_path, separator=':'):
    with open(vocab_path, 'w') as f:
        for idx in range(len(idx2word)):
            f.write(idx2word[idx]+' '+separator+' '+str(idx)+'\n')

def construct_vocab(input_seqs, vocab_config={'mini_word_freq':1, 'bos_eos':False}):
    '''
    @params:
        1. input_seqs: a list of seqs.
        2. vocab_config: 
            mini_word_freq: minimum word frequency
            bos_eos: <s> </s>
    @return:
        1. word2idx
        2. idx2word
    '''
    vocab = {}
    for seq in input_seqs:
        if type(seq) == type([]):
            for word in seq:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1
        else:
            assert type(seq) == str
            if seq not in vocab:
                vocab[seq] = 1
            else:
                vocab[seq] += 1
    
    # Discard start, end, pad and unk tokens if already present
    if '<s>' in vocab:
        del vocab['<s>']
    if '<pad>' in vocab:
        del vocab['<pad>']
    if '</s>' in vocab:
        del vocab['</s>']
    if '<unk>' in vocab:
        del vocab['<unk>']

    if vocab_config['bos_eos'] == True:
        word2id = {'<pad>': 0, '<unk>': 1, '<s>': 2, '</s>': 3}
        id2word = {0: '<pad>', 1: '<unk>', 2: '<s>', 3: '</s>'}
    else:
        word2id = {'<pad>': 0, '<unk>': 1,}
        id2word = {0: '<pad>', 1: '<unk>',}

    sorted_word2id = sorted(
        vocab.items(),
        key=operator.itemgetter(1),
        reverse=True
    )

    sorted_words = [x[0] for x in sorted_word2id if x[1] >= vocab_config['mini_word_freq']]
    
    for word in sorted_words:
        idx = len(word2id)
        word2id[word] = idx
        id2word[idx] = word

    return word2id, id2word

def read_vocab_from_data_file(data_path, vocab_config={'mini_word_freq':1, 'bos_eos':False, 'lowercase':False}, with_tag=True, separator=':'):
    '''
    Read data from files.
    @params:
        1. data_path: file path of data
        2. vocab_config: config of how to build vocab. It is used when in_vocab == None.
    @return:
        1. input vocab
    '''
    print('Reading source data ...')
    input_seqs = []
    with open(data_path, 'r') as f:
        for ind, line in enumerate(f):
            slot_tag_line = line.strip('\n\r').split(' <=> ')[0]
            if slot_tag_line == "":
                continue
            in_seq = []
            for item in slot_tag_line.split(' '):
                if with_tag:
                    tmp = item.split(separator)
                    assert len(tmp) >= 2
                    word, tag = separator.join(tmp[:-1]), tmp[-1]
                else:
                    word = item
                if vocab_config['lowercase']:
                    word = word.lower()
                in_seq.append(word)
            input_seqs.append(in_seq)

    print('Constructing input vocabulary from ', data_path, ' ...')
    word2idx, idx2word = construct_vocab(input_seqs, vocab_config)
    return (word2idx, idx2word)
