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
    parser.add_argument('-p', '--print_log', action='store_true', help='print log')
    opt = parser.parse_args()

    file = open(opt.infile)

    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0
    right, All = 0.0, 0.0
    all_slots = {}
    for line in file:
        line = line.strip('\n\r')
        if ' : ' in line:
            line_num, line = line.split(' : ')
        tmps = line.split(' <=> ')
        if len(tmps) > 1:
            line, intent_label, intent_pred = tmps
            intent_label_items = intent_label.split(';') if intent_label != '' else []
            intent_pred_items = intent_pred.split(';') if intent_pred != '' else []
            local_right = 0
            for pred_intent in intent_pred_items:
                if pred_intent in intent_label_items:
                    TP2 += 1
                    local_right += 1
                else:
                    FP2 += 1
            for label_intent in intent_label_items:
                if label_intent not in intent_pred_items:
                    FN2 += 1
            right += int(local_right > 0)
            All += 1
        else:
            line = tmps[0]

        words, labels, preds = [], [], []
        items = line.split(' ')
        for item in items:
            parts = item.split(':')
            word, label, pred = ':'.join(parts[:-2]), parts[-2], parts[-1]
            words.append(word)
            labels.append(label)
            preds.append(pred)
        label_chunks = get_chunks(['O']+labels+['O'])
        pred_chunks = get_chunks(['O']+preds+['O'])
        failed = False
        for pred_chunk in pred_chunks:
            if pred_chunk[-1] not in all_slots:
                all_slots[pred_chunk[-1]] = {'TP':0.0, 'FP':0.0, 'FN':0.0, 'TN':0.0}
            if pred_chunk in label_chunks:
                TP += 1
                all_slots[pred_chunk[-1]]['TP'] += 1
            else:
                FP += 1
                all_slots[pred_chunk[-1]]['FP'] += 1
                failed = True
        for label_chunk in label_chunks:
            if label_chunk[-1] not in all_slots:
                all_slots[label_chunk[-1]] = {'TP':0.0, 'FP':0.0, 'FN':0.0, 'TN':0.0}
            if label_chunk not in pred_chunks:
                FN += 1
                all_slots[label_chunk[-1]]['FN'] += 1
                failed = True
        if failed and opt.print_log:
            print(' '.join([word if label == 'O' else word+':'+label for word, label in zip(words, labels)]))
            print(' '.join([word if pred == 'O' else word+':'+pred for word, pred in zip(words, preds)]))
            print('-'*20)

    if TP == 0:
        print('all', int(TP), int(FN), int(FP), 0, 0, 0)
    else:
        print('all', int(TP), int(FN), int(FP), round(100*TP/(TP+FP), 2), round(100*TP/(TP+FN), 2), round(100*2*TP/(2*TP+FN+FP), 2))
    if TP2 != 0:
        print('all intent', int(TP2), int(FN2), int(FP2), round(100*TP2/(TP2+FP2), 2), round(100*TP2/(TP2+FN2), 2), round(100*2*TP2/(2*TP2+FN2+FP2), 2))
        print(right/All)
    
    all_F1 = []
    for slot,_ in sorted(all_slots.items(), key=lambda kv:(kv[1]['FN']+kv[1]['TP'], kv[0]), reverse=True):
        TP = all_slots[slot]['TP']
        FN = all_slots[slot]['FN']
        FP = all_slots[slot]['FP']
        if TP == 0:
            print(slot, int(TP), int(FN), int(FP), 0, 0, 0)
            all_F1.append(0)
        else:
            print(slot, int(TP), int(FN), int(FP), round(100*TP/(TP+FP), 2), round(100*TP/(TP+FN), 2), round(100*2*TP/(2*TP+FN+FP), 2))
            all_F1.append(100*2*TP/(2*TP+FN+FP))
    print("F1 ave. of slot count is", sum(all_F1)/len(all_F1))
