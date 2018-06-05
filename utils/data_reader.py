
"""Data utilities."""
import torch
import operator
import json

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
                word, idx = line.strip().split(' '+separator+' ')
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

def read_vocab_from_data_file(data_path, vocab_config={'mini_word_freq':1, 'bos_eos':False}, with_tag=True, separator=':'):
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
                    parts = item.split(separator)
                    word, tag = parts
                else:
                    word = item
                in_seq.append(word.lower())
            input_seqs.append(in_seq)

    print('Constructing input vocabulary from ', data_path, ' ...')
    word2idx, idx2word = construct_vocab(input_seqs, vocab_config)
    return (word2idx, idx2word)

def read_seqtag_data_with_unali_act(data_path, word2idx, slot_tag2idx, separator=':', keep_order=False, raw_word=False):
    '''
    Read data from files.
    @params:
        1. data_path: file path of data
        2. in_vocab: input vocabulary, e.g. {'<unk>':0, '<pad>':1, 'hello':2, ...}
        3. tag_vocab: tag vocabulary, e.g. {'<pad>':0, 'CITY':1, ...}
        4. unali_vocab: unaligned slot tag, e.g. {'request-路况':0, 'inform-操作-退出':1, ...}
        5. act_vocab: sentence classification vocabulary, e.g. {'inform':0, 'deny':1, ...}
        6. keep_order: keep a track of the line number
        7. raw_word: replace '<unk>' with the original word
    @return:
        1. input features
        2. tag labels
        3. unaligned labels
        4. act labels
    '''
    print('Reading source data ...')
    input_seqs = []
    slot_tag_seqs = []
    line_num = -1
    with open(data_path, 'r') as f:
        for ind, line in enumerate(f):
            line_num += 1
            slot_tag_line, intent = line.strip('\n\r').split(' <=> ')
            if slot_tag_line == "":
                continue
            in_seq, slot_tag_seq = [], []
            for item in slot_tag_line.split(' '):
                parts = item.split(separator)
                word, tag = parts

                if raw_word:
                    in_seq.append((word2idx[word.lower()], word) if word.lower() in word2idx else (word2idx['<unk>'], word))
                else:
                    in_seq.append(word2idx[word.lower()] if word.lower() in word2idx else word2idx['<unk>'])
                slot_tag_seq.append(slot_tag2idx[tag] if tag in slot_tag2idx else slot_tag2idx['<unk>'])
            if keep_order:
                in_seq.append(line_num)
            input_seqs.append(in_seq)
            slot_tag_seqs.append(slot_tag_seq)

    input_feats = {'data':input_seqs}
    slot_tag_labels = {'data':slot_tag_seqs}

    return input_feats, slot_tag_labels

def get_minibatch_with_unali_act(input_seqs, slot_tag_seqs, word2idx, slot_tag2idx, train_data_indx, index, batch_size, add_start_end=False, keep_order=False, raw_word=False, enc_dec_focus=False, device=None):
    """Prepare minibatch."""
    input_seqs = [input_seqs[idx] for idx in train_data_indx[index:index + batch_size]]
    slot_tag_seqs = [slot_tag_seqs[idx] for idx in train_data_indx[index:index + batch_size]]
    if add_start_end:
        if raw_word:
            input_seqs = [[(word2idx['<s>'], '<s>')] + line + [(word2idx['</s>'], '</s>')] for line in input_seqs]
        else:
            input_seqs = [[word2idx['<s>']] + line + [word2idx['</s>']] for line in input_seqs]
        slot_tag_seqs = [[slot_tag2idx['O']] + line + [slot_tag2idx['O']] for line in slot_tag_seqs]
    else:
        pass
    
    data_mb = zip(input_seqs, slot_tag_seqs)
    data_mb = sorted(data_mb, key=lambda x: len(x[0]), reverse=True) #sorted for pad setence

    if keep_order:
        line_nums = [seq[-1] for seq,_ in data_mb]
        data_mb = [(seq[:-1], slot_tag) for seq,slot_tag in data_mb]

    if raw_word:
        raw_words = [[word for word_idx, word in seq] for seq,slot_tag in data_mb]
        data_mb = [([word_idx for word_idx, word in seq], slot_tag) for seq,slot_tag in data_mb]

    lens = [len(seq) for seq,_ in data_mb]
    max_len = max(lens)
    input_idxs = [
        seq + [word2idx['<pad>']] * (max_len - len(seq))
        for seq,_ in data_mb
        ]
    input_idxs = torch.tensor(input_idxs, dtype=torch.long, device=device)

    # slot tag
    if not enc_dec_focus:
        slot_tag_idxs = [
            seq + [slot_tag2idx['<pad>']] * (max_len - len(seq))
            for _,seq in data_mb
            ]
    else:
        slot_tag_idxs = [
            [slot_tag2idx['<s>']] + seq + [slot_tag2idx['<pad>']] * (max_len - len(seq))
            for _,seq in data_mb
            ]
    slot_tag_idxs = torch.tensor(slot_tag_idxs, dtype=torch.long, device=device)

    ret = [input_idxs, slot_tag_idxs, lens]
    if keep_order:
        ret.append(line_nums)
    if raw_word:
        ret.append(raw_words)

    return ret
