"""Data utilities."""
import torch
import operator
import json
import random

def read_seqtag_data_with_class(data_path, tag2idx, class2idx, separator=':', multiClass=False, keep_order=False, lowercase=False):
    '''
    Read data from files.
    @params:
        1. data_path: file path of data
        2. in_vocab: input vocabulary, e.g. {'<unk>':0, '<pad>':1, 'hello':2, ...}
        3. tag_vocab: tag vocabulary, e.g. {'<pad>':0, 'CITY':1, ...}
        4. class_vocab: sentence classification vocabulary, e.g. {'inform':0, 'deny':1, ...}
        5. multiClass: multiple classifiers
        6. keep_order: keep a track of the line number
    @return:
        1. input features 
        2. tag labels 
        3. class labels
    '''
    print('Reading source data ...')
    input_seqs = []
    tag_seqs = []
    class_labels = []
    line_num = -1
    with open(data_path, 'r') as f:
        for ind, line in enumerate(f):
            line_num += 1
            slot_tag_line, class_name = line.strip('\n\r').split(' <=> ')
            if slot_tag_line == "":
                continue
            in_seq, tag_seq = [], []
            for item in slot_tag_line.split(' '):
                tmp = item.split(separator)
                assert len(tmp) >= 2
                word, tag = separator.join(tmp[:-1]), tmp[-1]
                if lowercase:
                    word = word.lower()
                in_seq.append(word)
                tag_seq.append(tag2idx[tag] if tag in tag2idx else (tag2idx['<unk>'], tag))
            if keep_order:
                in_seq.append(line_num)
            input_seqs.append(in_seq)
            tag_seqs.append(tag_seq)
            if multiClass:
                if class_name == '':
                    class_labels.append([])
                else:
                    class_labels.append([class2idx[val] for val in class_name.split(';')])
            else:
                if ';' not in class_name:
                    class_labels.append(class2idx[class_name])
                else:
                    class_labels.append((class2idx[class_name.split(';')[0]], class_name.split(';'))) # get the first class for training

    input_feats = {'data':input_seqs}
    tag_labels = {'data':tag_seqs}
    class_labels = {'data':class_labels}

    return input_feats, tag_labels, class_labels

def get_minibatch_with_class(input_seqs, tag_seqs, class_labels, tag2idx, class2idx, train_data_indx, index, batch_size, add_start_end=False, multiClass=False, keep_order=False, enc_dec_focus=False, device=None):
    """Prepare minibatch."""
    input_seqs = [input_seqs[idx] for idx in train_data_indx[index:index + batch_size]]
    tag_seqs = [tag_seqs[idx] for idx in train_data_indx[index:index + batch_size]]
    class_labels = [class_labels[idx] for idx in train_data_indx[index:index + batch_size]]
    if add_start_end:
        input_seqs = [['<s>'] + line + ['</s>'] for line in input_seqs]
        tag_seqs = [[tag2idx['O']] + line + [tag2idx['O']] for line in tag_seqs]
    else:
        pass
    
    data_mb = list(zip(input_seqs, tag_seqs, class_labels))
    data_mb.sort(key=lambda x: len(x[0]), reverse=True)   # sorted for pad setence

    raw_tags = [[item[1] if type(item) in {list, tuple} else item for item in tag] for seq,tag,cls in data_mb]
    data_mb = [(seq, [item[0] if type(item) in {list, tuple} else item for item in tag], cls) for seq,tag,cls in data_mb]
    if keep_order:
        line_nums = [seq[-1] for seq,_,_ in data_mb]
        data_mb = [(seq[:-1], tag, cls) for seq,tag,cls in data_mb]

    lens = [len(seq) for seq,_,_ in data_mb]
    max_len = max(lens)
    input_idxs = [seq for seq,_,_ in data_mb]

    if not enc_dec_focus:
        tag_idxs = [
            seq + [tag2idx['<pad>']] * (max_len - len(seq))
            for _,seq,_ in data_mb
            ]
    else:
        tag_idxs = [
            [tag2idx['<s>']] + seq + [tag2idx['<pad>']] * (max_len - len(seq))
            for _,seq,_ in data_mb
            ]
    tag_idxs = torch.tensor(tag_idxs, dtype=torch.long, device=device)
    
    if multiClass:
        raw_classes = [class_list for _,_,class_list in data_mb]
        class_tensor = torch.zeros(len(data_mb), len(class2idx), dtype=torch.float)
        for idx, (_,_,class_list) in enumerate(data_mb):
            for w in class_list:
                class_tensor[idx][w] = 1
        class_idxs = class_tensor.to(device)
    else:
        raw_classes = [class_label[1] if type(class_label) in {list, tuple} else class_label for _,_,class_label in data_mb]
        class_idxs = [class_label[0] if type(class_label) in {list, tuple} else class_label for _,_,class_label in data_mb]
        class_idxs = torch.tensor(class_idxs, dtype=torch.long, device=device)

    ret = [input_idxs, tag_idxs, raw_tags, class_idxs, raw_classes, lens]
    if keep_order:
        ret.append(line_nums)

    return ret

