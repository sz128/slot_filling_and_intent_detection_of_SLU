
import argparse
import random
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import os, sys, time
import logging
import gc

install_path = os.path.abspath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(install_path)

#from pytorch_pretrained_bert import BertTokenizer, BertModel
#from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule
from transformers import BertTokenizer, BertModel, XLNetTokenizer, XLNetModel 
from transformers import AdamW, get_linear_schedule_with_warmup
from models.optimization import BertAdam
from utils.bert_xlnet_inputs import prepare_inputs_for_bert_xlnet

from allennlp.modules.elmo import Elmo, batch_to_ids

import models.slot_tagger as slot_tagger
import models.slot_tagger_with_focus as slot_tagger_with_focus
import models.slot_tagger_crf as slot_tagger_with_crf
import models.snt_classifier as snt_classifier

import utils.vocab_reader as vocab_reader
import utils.data_reader_for_elmo as data_reader
import utils.read_wordEmb as read_wordEmb
import utils.util as util
import utils.acc as acc

MODEL_CLASSES = {
        'bert': (BertModel, BertTokenizer),
        'xlnet': (XLNetModel, XLNetTokenizer),
        }

parser = argparse.ArgumentParser()
parser.add_argument('--task_st', required=True, help='slot filling task: slot_tagger | slot_tagger_with_focus | slot_tagger_with_crf')
parser.add_argument('--task_sc', required=True, help='intent detection task: none | 2tails | maxPooling | hiddenCNN | hiddenAttention')
parser.add_argument('--sc_type', default='single_cls_CE', help='single_cls_CE | multi_cls_BCE')
parser.add_argument('--st_weight', type=float, default=0.5, help='loss weight for slot tagging task, ranging from 0 to 1.')

parser.add_argument('--dataset', required=True, help='atis-2 | snips')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--save_model', default='model', help='save model to this file')
#parser.add_argument('--mini_word_freq', type=int, default=2, help='mini_word_freq in the training data')
#parser.add_argument('--word_lowercase', action='store_true', help='word lowercase')
parser.add_argument('--bos_eos', action='store_true', help='Whether to add <s> and </s> to the input sentence (default is not)')
parser.add_argument('--save_vocab', default='vocab', help='save vocab to this file')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')

parser.add_argument('--testing', action='store_true', help='Only test your model (default is training && testing)')
parser.add_argument('--read_model', required=False, help='Online test: read model from this file')
parser.add_argument('--read_vocab', required=False, help='Online test: read input vocab from this file')
parser.add_argument('--out_path', required=False, help='Online test: out_path')

parser.add_argument('--pretrained_model_type', required=True, help='bert, xlnet')
parser.add_argument('--pretrained_model_name', required=True, help='bert-base-uncased, bert-base-cased, bert-large-uncased, bert-large-cased, bert-base-multilingual-cased, bert-base-chinese; xlnet-base-cased, xlnet-large-cased')
parser.add_argument('--elmo_json', required=True, help='')
parser.add_argument('--elmo_weight', required=True, help='')

#parser.add_argument('--emb_size', type=int, default=100, help='word embedding dimension')
parser.add_argument('--tag_emb_size', type=int, default=100, help='tag embedding dimension')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer dimension')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional RNN (default is unidirectional)')

#parser.add_argument('--ngpu'  , type=int, default=1, help='number of GPUs to use')
parser.add_argument('--deviceId', type=int, default=-1, help='train model on ith gpu. -1:cpu, 0:auto_select')
parser.add_argument('--random_seed', type=int, default=999, help='set initial random seed')

parser.add_argument('--lr', type=float, default=0.01, help='learning rate')
parser.add_argument('--dropout', type=float, default=0., help='dropout rate at each non-recurrent layer')
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--test_batchSize', type=int, default=0, help='input batch size in decoding')
parser.add_argument('--init_weight', type=float, default=0.2, help='all weights will be set to [-init_weight, init_weight] during initialization')
parser.add_argument('--max_norm', type=float, default=5, help="threshold of gradient clipping (2-norm)")
parser.add_argument('--max_epoch', type=int, default=50, help='max number of epochs to train for')
parser.add_argument('--experiment', default='exp', help='Where to store samples and models')
parser.add_argument('--optim', default='bertadam', help='choose an optimizer')
parser.add_argument('--warmup_proportion', type=float, default=0.1, help='Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10%% of training.')

opt = parser.parse_args()

assert opt.testing == bool(opt.out_path) == bool(opt.read_model) ==  bool(opt.read_vocab)

if opt.test_batchSize == 0:
    opt.test_batchSize = opt.batchSize

assert opt.task_st in {'slot_tagger', 'slot_tagger_with_focus', 'slot_tagger_with_crf'}
assert opt.task_sc in {'none', '2tails', 'maxPooling', 'hiddenCNN', 'hiddenAttention'}
assert opt.sc_type in {'single_cls_CE', 'multi_cls_BCE'}
if opt.sc_type == 'multi_cls_BCE':
    opt.multiClass = True
else:
    opt.multiClass = False
if opt.task_st == 'slot_tagger_with_focus':
    opt.enc_dec, opt.crf = True, False
elif opt.task_st == 'slot_tagger_with_crf':
    opt.enc_dec, opt.crf = False, True
else:
    opt.enc_dec, opt.crf = False, False

assert 0 < opt.st_weight <= 1
if opt.st_weight == 1 or opt.task_sc == 'none':
    opt.task_sc = None

if not opt.testing:
    if opt.task_sc:
        opt.task = opt.task_st + '__and__' + opt.task_sc + '__and__' + opt.sc_type + '__with_elmo_and_bert'
    else:
        opt.task = opt.task_st + '__with_elmo_and_bert'
    exp_path = util.hyperparam_string(opt)
    exp_path = os.path.join(opt.experiment, '', exp_path)
    exp_path += '__tes_%s' % (opt.tag_emb_size)
    if opt.task_sc:
        exp_path += '__alpha_%s' % (opt.st_weight)
    exp_path += '__%s_%s' % (opt.pretrained_model_type, opt.pretrained_model_name)
else:
    exp_path = opt.out_path
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

try:
    import absl.logging
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
except Exception as e:
    print("Failed to fix absl logging bug", e)
    pass
logFormatter = logging.Formatter('%(message)s') #('%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger('mylogger')
logger.setLevel(logging.DEBUG)
if opt.testing:
    fileHandler = logging.FileHandler('%s/log_test.txt' % (exp_path), mode='w')
else:
    fileHandler = logging.FileHandler('%s/log_train.txt' % (exp_path), mode='w')
fileHandler.setFormatter(logFormatter)
logger.addHandler(fileHandler)
if not opt.noStdout:
    consoleHandler = logging.StreamHandler(sys.stdout)
    consoleHandler.setFormatter(logFormatter)
    logger.addHandler(consoleHandler)
logger.info(opt)
logger.info("Experiment path: %s" % (exp_path))
logger.info(time.asctime(time.localtime(time.time())))

if opt.deviceId >= 0:
    import utils.gpu_selection as gpu_selection
    if opt.deviceId > 0:
        opt.deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu(assigned_gpu_id=opt.deviceId - 1)
    elif opt.deviceId == 0:
        opt.deviceId, gpu_name, valid_gpus = gpu_selection.auto_select_gpu()
    logger.info("Valid GPU list: %s ; GPU %d (%s) is auto selected." % (valid_gpus, opt.deviceId, gpu_name))
    torch.cuda.set_device(opt.deviceId)
    opt.device = torch.device("cuda") # is equivalent to torch.device('cuda:X') where X is the result of torch.cuda.current_device()
else:
    logger.info("CPU is used.")
    opt.device = torch.device("cpu")

random.seed(opt.random_seed)
torch.manual_seed(opt.random_seed)
if torch.cuda.is_available():
    if opt.device.type != 'cuda':
        logger.info("WARNING: You have a CUDA device, so you should probably run with --deviceId [1|2|3]")
    else:
        torch.cuda.manual_seed(opt.random_seed)
np.random.seed(opt.random_seed)


dataroot = opt.dataroot

tag_vocab_dir = dataroot + '/vocab.slot' 
class_vocab_dir = dataroot + '/vocab.intent'
train_data_dir = dataroot + '/train'
valid_data_dir = dataroot + '/valid'
test_data_dir = dataroot + '/test'

if not opt.testing:
    tag_to_idx, idx_to_tag = vocab_reader.read_vocab_file(tag_vocab_dir, bos_eos=opt.enc_dec)
    class_to_idx, idx_to_class = vocab_reader.read_vocab_file(class_vocab_dir, bos_eos=False)
else:
    tag_to_idx, idx_to_tag = vocab_reader.read_vocab_file(opt.read_vocab+'.tag', bos_eos=False, no_pad=True, no_unk=True)
    class_to_idx, idx_to_class = vocab_reader.read_vocab_file(opt.read_vocab+'.class', bos_eos=False, no_pad=True, no_unk=True)

logger.info("Vocab size: %s %s" % (len(tag_to_idx), len(class_to_idx)))
if not opt.testing:
    vocab_reader.save_vocab(idx_to_tag, os.path.join(exp_path, opt.save_vocab+'.tag'))
    vocab_reader.save_vocab(idx_to_class, os.path.join(exp_path, opt.save_vocab+'.class'))

opt.word_lowercase = False
if not opt.testing:
    train_feats, train_tags, train_class = data_reader.read_seqtag_data_with_class(train_data_dir, tag_to_idx, class_to_idx, multiClass=opt.multiClass, lowercase=opt.word_lowercase)
    valid_feats, valid_tags, valid_class = data_reader.read_seqtag_data_with_class(valid_data_dir, tag_to_idx, class_to_idx, multiClass=opt.multiClass, keep_order=opt.testing, lowercase=opt.word_lowercase)
    test_feats, test_tags, test_class = data_reader.read_seqtag_data_with_class(test_data_dir, tag_to_idx, class_to_idx, multiClass=opt.multiClass, keep_order=opt.testing, lowercase=opt.word_lowercase)
else:
    valid_feats, valid_tags, valid_class = data_reader.read_seqtag_data_with_class(valid_data_dir, tag_to_idx, class_to_idx, multiClass=opt.multiClass, keep_order=opt.testing, lowercase=opt.word_lowercase)
    test_feats, test_tags, test_class = data_reader.read_seqtag_data_with_class(test_data_dir, tag_to_idx, class_to_idx, multiClass=opt.multiClass, keep_order=opt.testing, lowercase=opt.word_lowercase)

pretrained_model_class, tokenizer_class = MODEL_CLASSES[opt.pretrained_model_type]
tokenizer = tokenizer_class.from_pretrained(opt.pretrained_model_name)
pretrained_model = pretrained_model_class.from_pretrained(opt.pretrained_model_name)
print(pretrained_model.config)
model_elmo = Elmo(opt.elmo_json, opt.elmo_weight, 1, dropout=0)
opt.emb_size = None
if opt.task_st == 'slot_tagger':
    model_tag = slot_tagger.LSTMTagger(opt.emb_size, opt.hidden_size, None, len(tag_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, pretrained_model=pretrained_model, pretrained_model_type=opt.pretrained_model_type, elmo_model=model_elmo)
elif opt.task_st == 'slot_tagger_with_focus':
    model_tag = slot_tagger_with_focus.LSTMTagger_focus(opt.emb_size, opt.tag_emb_size, opt.hidden_size, None, len(tag_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, pretrained_model=pretrained_model, pretrained_model_type=opt.pretrained_model_type, elmo_model=model_elmo)
elif opt.task_st == 'slot_tagger_with_crf':
    model_tag = slot_tagger_with_crf.LSTMTagger_CRF(opt.emb_size, opt.hidden_size, None, len(tag_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, pretrained_model=pretrained_model, pretrained_model_type=opt.pretrained_model_type, elmo_model=model_elmo)
else:
    exit()

if opt.task_sc == '2tails':
    model_class = snt_classifier.sntClassifier_2tails(opt.hidden_size, len(class_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, multi_class=opt.multiClass)
    encoder_info_filter = lambda info: info[0]
elif opt.task_sc == 'maxPooling':
    model_class = snt_classifier.sntClassifier_hiddenPooling(opt.hidden_size, len(class_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, multi_class=opt.multiClass, pooling='max')
    encoder_info_filter = lambda info: (info[1], info[2])
elif opt.task_sc == 'hiddenCNN':
    model_class = snt_classifier.sntClassifier_hiddenCNN(opt.hidden_size, len(class_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, multi_class=opt.multiClass)
    encoder_info_filter = lambda info: (info[1], info[2])
elif opt.task_sc == 'hiddenAttention':
    model_class = snt_classifier.sntClassifier_hiddenAttention(opt.hidden_size, len(class_to_idx), bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device, multi_class=opt.multiClass)
    encoder_info_filter = lambda info: info
else:
    pass

model_tag = model_tag.to(opt.device)
if opt.task_sc:
    model_class = model_class.to(opt.device)

# read pretrained model
if opt.read_model:
    model_tag.load_model(opt.read_model+'.tag')
    if opt.task_sc:
        model_class.load_model(opt.read_model+'.class')
else:
    assert not opt.testing
    #custom init (needed maybe) ...
    model_tag.init_weights(opt.init_weight)
    if opt.task_sc:
        model_class.init_weights(opt.init_weight)

# loss function
weight_mask = torch.ones(len(tag_to_idx), device=opt.device)
weight_mask[tag_to_idx['<pad>']] = 0
tag_loss_function = nn.NLLLoss(weight=weight_mask, size_average=False)
if opt.task_sc:
    if opt.multiClass:
        class_loss_function = nn.BCELoss(size_average=False)
    else:
        class_loss_function = nn.NLLLoss(size_average=False)

# optimizer
if opt.optim.lower() == 'bertadam':
    named_params = []
    named_params += list(model_tag.named_parameters())
    if opt.task_sc:
        named_params += list(model_class.named_parameters())
    named_params = list(filter(lambda p: p[1].requires_grad, named_params))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_feats['data']) // opt.batchSize * opt.max_epoch
    optimizer = BertAdam(optimizer_grouped_parameters, lr=opt.lr, warmup=opt.warmup_proportion, t_total=num_train_optimization_steps)
elif opt.optim.lower() == 'adamw':
    params = []
    params += list(model_tag.parameters())
    if opt.task_sc:
        params += list(model_class.parameters())
    params = list(filter(lambda p: p.requires_grad, params))
    named_params = []
    named_params += list(model_tag.named_parameters())
    if opt.task_sc:
        named_params += list(model_class.named_parameters())
    named_params = list(filter(lambda p: p[1].requires_grad, named_params))
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
    num_train_optimization_steps = len(train_feats['data']) // opt.batchSize * opt.max_epoch
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.lr, correct_bias=False)  # To reproduce BertAdam specific behavior set correct_bias=False
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(opt.warmup_proportion * num_train_optimization_steps), num_training_steps=num_train_optimization_steps)  # PyTorch scheduler

# prepare_inputs_for_bert(sentences, word_lengths)

def decode(data_feats, data_tags, data_class, output_path):
    data_index = np.arange(len(data_feats))
    losses = []
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    TP2, FP2, FN2, TN2 = 0.0, 0.0, 0.0, 0.0
    with open(output_path, 'w') as f:
        for j in range(0, len(data_index), opt.test_batchSize):
            if opt.testing:
                words, tags, raw_tags, classes, raw_classes, lens, line_nums = data_reader.get_minibatch_with_class(data_feats, data_tags, data_class, tag_to_idx, class_to_idx, data_index, j, opt.test_batchSize, add_start_end=opt.bos_eos, multiClass=opt.multiClass, keep_order=opt.testing, enc_dec_focus=opt.enc_dec, device=opt.device)
            else:
                words, tags, raw_tags, classes, raw_classes, lens = data_reader.get_minibatch_with_class(data_feats, data_tags, data_class, tag_to_idx, class_to_idx, data_index, j, opt.test_batchSize, add_start_end=opt.bos_eos, multiClass=opt.multiClass, keep_order=opt.testing, enc_dec_focus=opt.enc_dec, device=opt.device)
            inputs = {}
            inputs['transformer'] = prepare_inputs_for_bert_xlnet(words, lens, tokenizer, 
                    cls_token_at_end=bool(opt.pretrained_model_type in ['xlnet']),  # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    sep_token=tokenizer.sep_token,
                    cls_token_segment_id=2 if opt.pretrained_model_type in ['xlnet'] else 0,
                    pad_on_left=bool(opt.pretrained_model_type in ['xlnet']), # pad on the left for xlnet
                    pad_token_segment_id=4 if opt.pretrained_model_type in ['xlnet'] else 0,
                    device=opt.device)
            inputs['elmo'] = batch_to_ids(words).to(opt.device)

            if opt.enc_dec:
                opt.greed_decoding = True
                if opt.greed_decoding:
                    tag_scores_1best, outputs_1best, encoder_info = model_tag.decode_greed(inputs, tags[:, 0:1], lens, with_snt_classifier=True)
                    tag_loss = tag_loss_function(tag_scores_1best.contiguous().view(-1, len(tag_to_idx)), tags[:, 1:].contiguous().view(-1))
                    top_pred_slots = outputs_1best.cpu().numpy()
                else:
                    beam_size = 2
                    beam_scores_1best, top_path_slots, encoder_info = model_tag.decode_beam_search(inputs, lens, beam_size, tag_to_idx, with_snt_classifier=True)
                    top_pred_slots = [[item[0].item() for item in seq] for seq in top_path_slots]
                    ppl = beam_scores_1best.cpu() / torch.tensor(lens, dtype=torch.float)
                    tag_loss = ppl.exp().sum()
                #tags = tags[:, 1:].data.cpu().numpy()
            elif opt.crf:
                max_len = max(lens)
                masks = [([1] * l) + ([0] * (max_len - l)) for l in lens]
                masks = torch.tensor(masks, dtype=torch.bool, device=opt.device)
                crf_feats, encoder_info = model_tag._get_lstm_features(inputs, lens, with_snt_classifier=True)
                tag_path_scores, tag_path = model_tag.forward(crf_feats, masks)
                tag_loss = model_tag.neg_log_likelihood(crf_feats, masks, tags)
                top_pred_slots = tag_path.data.cpu().numpy()
            else:
                tag_scores, encoder_info = model_tag(inputs, lens, with_snt_classifier=True)
                tag_loss = tag_loss_function(tag_scores.contiguous().view(-1, len(tag_to_idx)), tags.view(-1))
                top_pred_slots = tag_scores.data.cpu().numpy().argmax(axis=-1)
                #tags = tags.data.cpu().numpy()
            if opt.task_sc:
                class_scores = model_class(encoder_info_filter(encoder_info))
                class_loss = class_loss_function(class_scores, classes)
                if opt.multiClass:
                    snt_probs = class_scores.data.cpu().numpy()
                else:
                    snt_probs = class_scores.data.cpu().numpy().argmax(axis=-1)
                losses.append([tag_loss.item()/sum(lens), class_loss.item()/len(lens)])
            else:
                losses.append([tag_loss.item()/sum(lens), 0])

            #classes = classes.data.cpu().numpy()
            for idx, pred_line in enumerate(top_pred_slots):
                length = lens[idx]
                pred_seq = [idx_to_tag[tag] for tag in pred_line][:length]
                lab_seq = [idx_to_tag[tag] if type(tag) == int else tag for tag in raw_tags[idx]]
                pred_chunks = acc.get_chunks(['O']+pred_seq+['O'])
                label_chunks = acc.get_chunks(['O']+lab_seq+['O'])
                for pred_chunk in pred_chunks:
                    if pred_chunk in label_chunks:
                        TP += 1
                    else:
                        FP += 1
                for label_chunk in label_chunks:
                    if label_chunk not in pred_chunks:
                        FN += 1

                input_line = words[idx]
                word_tag_line = [input_line[_idx]+':'+lab_seq[_idx]+':'+pred_seq[_idx] for _idx in range(len(input_line))]
                
                if opt.task_sc:
                    if opt.multiClass:
                        pred_classes = [idx_to_class[i] for i,p in enumerate(snt_probs[idx]) if p > 0.5]
                        gold_classes = [idx_to_class[i] for i in raw_classes[idx]]
                        for pred_class in pred_classes:
                            if pred_class in gold_classes:
                                TP2 += 1
                            else:
                                FP2 += 1
                        for gold_class in gold_classes:
                            if gold_class not in pred_classes:
                                FN2 += 1
                        gold_class_str = ';'.join(gold_classes)
                        pred_class_str = ';'.join(pred_classes)
                    else:
                        pred_class = idx_to_class[snt_probs[idx]]
                        if type(raw_classes[idx]) == int:
                            gold_classes = {idx_to_class[raw_classes[idx]]}
                        else:
                            gold_classes = set(raw_classes[idx])
                        if pred_class in gold_classes:
                            TP2 += 1
                        else:
                            FP2 += 1
                            FN2 += 1
                        gold_class_str = ';'.join(list(gold_classes))
                        pred_class_str = pred_class
                else:
                    gold_class_str = ''
                    pred_class_str = ''

                if opt.testing:
                    f.write(str(line_nums[idx])+' : '+' '.join(word_tag_line)+' <=> '+gold_class_str+' <=> '+pred_class_str+'\n')
                else:
                    f.write(' '.join(word_tag_line)+' <=> '+gold_class_str+' <=> '+pred_class_str+'\n')

    if TP == 0:
        p, r, f = 0, 0, 0
    else:
        p, r, f = 100*TP/(TP+FP), 100*TP/(TP+FN), 100*2*TP/(2*TP+FN+FP)
    
    mean_losses = np.mean(losses, axis=0)
    return mean_losses, p, r, f, 0 if 2*TP2+FN2+FP2 == 0 else 100*2*TP2/(2*TP2+FN2+FP2)

if not opt.testing:
    logger.info("Training starts at %s" % (time.asctime(time.localtime(time.time()))))
    train_data_index = np.arange(len(train_feats['data']))
    best_f1, best_result = -1, {}
    for i in range(opt.max_epoch):
        start_time = time.time()
        losses = []
        # training data shuffle
        np.random.shuffle(train_data_index)
        model_tag.train()
        if opt.task_sc:
            model_class.train()
        
        nsentences = len(train_data_index)
        piece_sentences = opt.batchSize if int(nsentences * 0.1 / opt.batchSize) == 0 else int(nsentences * 0.1 / opt.batchSize) * opt.batchSize
        for j in range(0, nsentences, opt.batchSize):
            words, tags, raw_tags, classes, raw_classes, lens = data_reader.get_minibatch_with_class(train_feats['data'], train_tags['data'], train_class['data'], tag_to_idx, class_to_idx, train_data_index, j, opt.batchSize, add_start_end=opt.bos_eos, multiClass=opt.multiClass, enc_dec_focus=opt.enc_dec, device=opt.device)
            inputs = {}
            inputs['transformer'] = prepare_inputs_for_bert_xlnet(words, lens, tokenizer, 
                    cls_token_at_end=bool(opt.pretrained_model_type in ['xlnet']),  # xlnet has a cls token at the end
                    cls_token=tokenizer.cls_token,
                    sep_token=tokenizer.sep_token,
                    cls_token_segment_id=2 if opt.pretrained_model_type in ['xlnet'] else 0,
                    pad_on_left=bool(opt.pretrained_model_type in ['xlnet']), # pad on the left for xlnet
                    pad_token_segment_id=4 if opt.pretrained_model_type in ['xlnet'] else 0,
                    device=opt.device)
            inputs['elmo'] = batch_to_ids(words).to(opt.device)
            optimizer.zero_grad()
            if opt.enc_dec:
                tag_scores, encoder_info = model_tag(inputs, tags[:, :-1], lens, with_snt_classifier=True)
                tag_loss = tag_loss_function(tag_scores.contiguous().view(-1, len(tag_to_idx)), tags[:, 1:].contiguous().view(-1))
            elif opt.crf:
                max_len = max(lens)
                masks = [([1] * l) + ([0] * (max_len - l)) for l in lens]
                masks = torch.tensor(masks, dtype=torch.bool, device=opt.device)
                crf_feats, encoder_info = model_tag._get_lstm_features(inputs, lens, with_snt_classifier=True)
                tag_loss = model_tag.neg_log_likelihood(crf_feats, masks, tags)
            else:
                tag_scores, encoder_info = model_tag(inputs, lens, with_snt_classifier=True)
                tag_loss = tag_loss_function(tag_scores.contiguous().view(-1, len(tag_to_idx)), tags.view(-1))
            if opt.task_sc:
                class_scores = model_class(encoder_info_filter(encoder_info))
                class_loss = class_loss_function(class_scores, classes)
                losses.append([tag_loss.item()/sum(lens), class_loss.item()/len(lens)])
                total_loss = opt.st_weight * tag_loss + (1 - opt.st_weight) * class_loss
            else:
                losses.append([tag_loss.item()/sum(lens), 0])
                total_loss = tag_loss
            total_loss.backward()
            
            # Clips gradient norm of an iterable of parameters.
            if opt.optim.lower() != 'bertadam' and opt.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, opt.max_norm)
            
            if opt.optim.lower() == 'adamw':
                scheduler.step()
            optimizer.step()

            if j % piece_sentences == 0:
                print('[learning] epoch %i >> %2.2f%%'%(i,(j+opt.batchSize)*100./nsentences),'completed in %.2f (sec) <<\r'%(time.time()-start_time), end='')
                sys.stdout.flush()
        print('')
        
        mean_loss = np.mean(losses, axis=0)
        logger.info('Training:\tEpoch : %d\tTime : %.4fs\tLoss of tag : %.2f\tLoss of class : %.2f ' % (i, time.time() - start_time, mean_loss[0], mean_loss[1]))
        gc.collect()

        model_tag.eval()
        if opt.task_sc:
            model_class.eval()
        # Evaluation
        start_time = time.time()
        loss_val, p_val, r_val, f_val, cf_val = decode(valid_feats['data'], valid_tags['data'], valid_class['data'], os.path.join(exp_path, 'valid.iter'+str(i)))
        logger.info('Validation:\tEpoch : %d\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f ' % (i, time.time() - start_time, loss_val[0], loss_val[1], f_val, cf_val))
        start_time = time.time()
        loss_te, p_te, r_te, f_te, cf_te = decode(test_feats['data'], test_tags['data'], test_class['data'], os.path.join(exp_path, 'test.iter'+str(i)))
        logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f ' % (i, time.time() - start_time, loss_te[0], loss_te[1], f_te, cf_te))

        if opt.task_sc:
            val_f1_score = (opt.st_weight * f_val + (1 - opt.st_weight) * cf_val)
        else:
            val_f1_score = f_val
        if best_f1 < val_f1_score:
            model_tag.save_model(os.path.join(exp_path, opt.save_model+'.tag'))
            if opt.task_sc:
                model_class.save_model(os.path.join(exp_path, opt.save_model+'.class'))
            best_f1 = val_f1_score
            logger.info('NEW BEST:\tEpoch : %d\tbest valid F1 : %.2f, cls-F1 : %.2f;\ttest F1 : %.2f, cls-F1 : %.2f' % (i, f_val, cf_val, f_te, cf_te))
            best_result['iter'] = i
            best_result['vf1'], best_result['vcf1'], best_result['vce'] = f_val, cf_val, loss_val
            best_result['tf1'], best_result['tcf1'], best_result['tce'] = f_te, cf_te, loss_te
    logger.info('BEST RESULT: \tEpoch : %d\tbest valid (Loss: (%.2f, %.2f) F1 : %.2f cls-F1 : %.2f)\tbest test (Loss: (%.2f, %.2f) F1 : %.2f cls-F1 : %.2f) ' % (best_result['iter'], best_result['vce'][0], best_result['vce'][1], best_result['vf1'], best_result['vcf1'], best_result['tce'][0], best_result['tce'][1], best_result['tf1'], best_result['tcf1']))
else:    
    logger.info("Testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    model_tag.eval()
    if opt.task_sc:
        model_class.eval()
    start_time = time.time()
    loss_val, p_val, r_val, f_val, cf_val = decode(valid_feats['data'], valid_tags['data'], valid_class['data'], os.path.join(exp_path, 'valid.eval'))
    logger.info('Validation:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f ' % (time.time() - start_time, loss_val[0], loss_val[1], f_val, cf_val))
    start_time = time.time()
    loss_te, p_te, r_te, f_te, cf_te = decode(test_feats['data'], test_tags['data'], test_class['data'], os.path.join(exp_path, 'test.eval'))
    logger.info('Evaluation:\tTime : %.4fs\tLoss : (%.2f, %.2f)\tFscore : %.2f\tcls-F1 : %.2f ' % (time.time() - start_time, loss_te[0], loss_te[1], f_te, cf_te))
