import sys
import argparse

def get_chunks(labels):
    chunks = []
    start_idx,end_idx = 0,0
    for idx in range(1,len(labels)-1):
        chunkStart, chunkEnd = False,False
        if labels[idx-1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            prevTag, prevType = labels[idx-1][:1], labels[idx-1][2:]
        else:
            prevTag, prevType = 'O', 'O'
        if labels[idx] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            Tag, Type = labels[idx][:1], labels[idx][2:]
        else:
            Tag, Type = 'O', 'O'
        if labels[idx+1] not in ('O', '<pad>', '<unk>', '<s>', '</s>', '<STOP>', '<START>'):
            nextTag, nextType = labels[idx+1][:1], labels[idx+1][2:]
        else:
            nextTag, nextType = 'O', 'O'

        if (Tag == 'B' and prevTag in ('B', 'I', 'O')) or (prevTag, Tag) in [('O', 'I'), ('E', 'E'), ('E', 'I'), ('O', 'E')]:
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if (Tag in ('B','I') and nextTag in ('B','O')) or (Tag == 'E' and nextTag in ('E', 'I', 'O')):
            chunkEnd = True
        if Tag != 'O' and Type != nextType:
            chunkEnd = True

        if chunkStart:
            start_idx = idx
        if chunkEnd:
            end_idx = idx
            chunks.append((start_idx,end_idx,Type))
            start_idx,end_idx = 0,0
    return chunks

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help='path to dataset')
    parser.add_argument('-l', '--labfile', required=True, help='path to dataset')
    parser.add_argument('-p', '--print_log', action='store_true', help='print log')
    opt = parser.parse_args()

    pred_lines = {}
    idx = 0
    for line in open(opt.infile):
        line = line.strip('\n\r')
        if ' : ' in line:
            idx_str, line = line.split(' : ')
            pred_lines[int(idx_str)] = line
        else:
            pred_lines[idx] = line
        idx += 1
    lab_lines = {}
    idx = 0
    for line in open(opt.labfile):
        line = line.strip('\n\r')
        if ' : ' in line:
            idx_str, line = line.split(' : ')
            lab_lines[int(idx_str)] = line
        else:
            lab_lines[idx] = line
        idx += 1
    assert len(pred_lines) == len(lab_lines)

    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    all_slots = {}
    for idx in pred_lines:
        lab_line, pred_line = lab_lines[idx], pred_lines[idx]
        lab_ali = lab_line.split(' <=> ')[0]
        pred_ali = pred_line.split(' <=> ')[0]

        # slot tag
        words = [item.split(':')[0] for item in lab_ali.split(' ')]
        labels = [item.split(':', 1)[1] for item in lab_ali.split(' ')]
        preds = [item.split(':', 1)[1] for item in pred_ali.split(' ')]
        label_chunks = get_chunks(['O']+labels+['O'])
        pred_chunks = get_chunks(['O']+preds+['O'])
        for pred_chunk in pred_chunks:
            if pred_chunk[-1] not in all_slots:
                all_slots[pred_chunk[-1]] = {'TP':0.0, 'FP':0.0, 'FN':0.0, 'TN':0.0}
            if pred_chunk in label_chunks:
                TP += 1
                all_slots[pred_chunk[-1]]['TP'] += 1
            else:
                FP += 1
                all_slots[pred_chunk[-1]]['FP'] += 1
        for label_chunk in label_chunks:
            if label_chunk[-1] not in all_slots:
                all_slots[label_chunk[-1]] = {'TP':0.0, 'FP':0.0, 'FN':0.0, 'TN':0.0}
            if label_chunk not in pred_chunks:
                FN += 1
                all_slots[label_chunk[-1]]['FN'] += 1
        
    if TP == 0:
        print('all slot', int(TP), int(FP), int(FN), 0, 0, 0)
    else:
        print('all slot', int(TP), int(FP), int(FN), round(100*TP/(TP+FP), 2), round(100*TP/(TP+FN), 2), round(100*2*TP/(2*TP+FN+FP), 2))

    for slot in all_slots:
        TP = all_slots[slot]['TP']
        FP = all_slots[slot]['FP']
        FN = all_slots[slot]['FN']
        if TP == 0:
            print(slot, int(TP), int(FP), int(FN), 0, 0, 0)
        else:
            print(slot, int(TP), int(FP), int(FN), round(100*TP/(TP+FP), 2), round(100*TP/(TP+FN), 2), round(100*2*TP/(2*TP+FN+FP), 2))

