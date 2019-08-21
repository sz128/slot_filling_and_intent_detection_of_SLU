import sys

def get_chunks(labels):
    """
        It supports IOB2 or IOBES tagging scheme.
        You may also want to try https://github.com/sighsmile/conlleval.
    """
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

        if Tag == 'B' or Tag == 'S' or (prevTag, Tag) in {('O', 'I'), ('O', 'E'), ('E', 'I'), ('E', 'E'), ('S', 'I'), ('S', 'E')}:
            chunkStart = True
        if Tag != 'O' and prevType != Type:
            chunkStart = True

        if Tag == 'E' or Tag == 'S' or (Tag, nextTag) in {('B', 'B'), ('B', 'O'), ('B', 'S'), ('I', 'B'), ('I', 'O'), ('I', 'S')}:
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
    import argparse
    import prettytable

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--infile', required=True, help='path to dataset')
    parser.add_argument('-p', '--print_log', action='store_true', help='print log')
    opt = parser.parse_args()

    file = open(opt.infile)

    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0
    correct_sentence_slots, correct_sentence_intents, correct_sentence, sentence_number = 0.0, 0.0, 0.0, 0.0
    all_slots = {}
    for line in file:
        line = line.strip('\n\r')
        if ' : ' in line:
            line_num, line = line.split(' : ')
        tmps = line.split(' <=> ')
        if len(tmps) > 1:
            line, intent_label, intent_pred = tmps
            intent_label_items = set(intent_label.split(';')) if intent_label != '' else set()
            intent_pred_items = set(intent_pred.split(';')) if intent_pred != '' else set()
            for pred_intent in intent_pred_items:
                if pred_intent in intent_label_items:
                    TP2 += 1
                else:
                    FP2 += 1
            for label_intent in intent_label_items:
                if label_intent not in intent_pred_items:
                    FN2 += 1
            correct_sentence_intents += int(intent_label_items == intent_pred_items)
            intent_correct = (intent_label_items == intent_pred_items)
        else:
            line = tmps[0]
            intent_correct = True
        sentence_number += 1

        words, labels, preds = [], [], []
        items = line.split(' ')
        for item in items:
            parts = item.split(':')
            word, label, pred = ':'.join(parts[:-2]), parts[-2], parts[-1]
            words.append(word)
            labels.append(label)
            preds.append(pred)
        label_chunks = set(get_chunks(['O']+labels+['O']))
        pred_chunks = set(get_chunks(['O']+preds+['O']))
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
        correct_sentence_slots += int(label_chunks == pred_chunks)
        if intent_correct and label_chunks == pred_chunks:
            correct_sentence += 1
        if label_chunks != pred_chunks and opt.print_log:
            print(' '.join([word if label == 'O' else word+':'+label for word, label in zip(words, labels)]))
            print(' '.join([word if pred == 'O' else word+':'+pred for word, pred in zip(words, preds)]))
            print('-'*20)

    table = prettytable.PrettyTable(["Metric", "TP", "FN", "FP", "Prec.", "Recall", "F1-score", "Sentence Acc"])
    ## 自定义表格输出样式
    ### 设定左对齐
    table.align = 'l'
    ### 设定数字输出格式
    table.float_format = "2.2"

    if TP == 0:
        table.add_row(('all slots', int(TP), int(FN), int(FP), 0, 0, 0, 100*correct_sentence_slots/sentence_number))
    else:
        table.add_row(('all slots', int(TP), int(FN), int(FP), 100*TP/(TP+FP), 100*TP/(TP+FN), 100*2*TP/(2*TP+FN+FP), 100*correct_sentence_slots/sentence_number))
    if TP2 != 0:
        table.add_row(('all intents', int(TP2), int(FN2), int(FP2), 100*TP2/(TP2+FP2), 100*TP2/(TP2+FN2), 100*2*TP2/(2*TP2+FN2+FP2), 100*correct_sentence_intents/sentence_number))
        table.add_row(('all slots+intents', '-', '-', '-', '-', '-', '-', 100*correct_sentence/sentence_number))
    table.add_row(('-', '-', '-', '-', '-', '-', '-', '-'))
    all_F1 = []
    for slot,_ in sorted(all_slots.items(), key=lambda kv:(kv[1]['FN']+kv[1]['TP'], kv[0]), reverse=True):
        TP = all_slots[slot]['TP']
        FN = all_slots[slot]['FN']
        FP = all_slots[slot]['FP']
        if TP == 0:
            table.add_row((slot, int(TP), int(FN), int(FP), 0, 0, 0, '-'))
            all_F1.append(0)
        else:
            table.add_row((slot, int(TP), int(FN), int(FP), 100*TP/(TP+FP), 100*TP/(TP+FN), 100*2*TP/(2*TP+FN+FP), '-'))
            all_F1.append(100*2*TP/(2*TP+FN+FP))
    table.add_row(("Macro-average of slots", '-', '-', '-', '-', '-', sum(all_F1)/len(all_F1), '-'))
    print(table)
