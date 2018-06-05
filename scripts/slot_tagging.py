
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

import utils.data_reader as data_reader
import utils.read_wordEmb as read_wordEmb
import utils.util as util
import utils.acc as acc

parser = argparse.ArgumentParser()
parser.add_argument('--task', required=True, help='slot_tagger | slot_tagger_with_focus | slot_tagger_with_crf')
parser.add_argument('--dataset', required=True, help='map')
parser.add_argument('--dataroot', required=True, help='dataroot contains: train, valid, vocab.slot_tag, vocab.act_tag, vocab.unali, vocab.act')
parser.add_argument('--save_model', default='model', help='save model to this file')
parser.add_argument('--mini_word_freq', type=int, default=2, help='mini_word_freq in the training data')
parser.add_argument('--bos_eos', action='store_true', help='Whether to add <s> and </s> to the input sentence (default is not)')
parser.add_argument('--save_vocab', default='vocab', help='save vocab to this file')
parser.add_argument('--noStdout', action='store_true', help='Only log to a file; no stdout')

parser.add_argument('--testing', action='store_true', help='Online test: only test your model (default is training && testing)')
parser.add_argument('--test_file_name', required=False, help='${test_file_name}.lab, ${test_file_name}.rec')
parser.add_argument('--read_model', required=False, help='Online test: read model from this file')
parser.add_argument('--read_vocab', required=False, help='Online test: read input vocab from this file')
parser.add_argument('--out_path', required=False, help='Online test: out_path')
parser.add_argument('--add_pred_rule', action='store_true', help='Whether to consider BIO rule when predicting tags (default is not)')
parser.add_argument('--save_model_to_cpu', action='store_true', help='Save model to cpu (default is not)')

parser.add_argument('--read_input_word2vec', required=False, help='read word embedding from word2vec file')

parser.add_argument('--emb_size', type=int, default=100, help='word embedding dimension')
parser.add_argument('--hidden_size', type=int, default=100, help='hidden layer dimension')
parser.add_argument('--num_layers', type=int, default=1, help='number of hidden layers')
parser.add_argument('--bidirectional', action='store_true', help='Whether to use bidirectional RNN (default is unidirectional)')
parser.add_argument('--decoder_tied', action='store_true', help='To tie the output layer and input embedding in decoder')

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
parser.add_argument('--optim', default='sgd', help='choose an optimizer')
parser.add_argument('--st_weight', type=float, default=1, help='loss weight for slot tagging task, ranging from 1 to the max.')

opt = parser.parse_args()

assert opt.testing == bool(opt.out_path) == bool(opt.read_model) ==  bool(opt.read_vocab) == bool(opt.test_file_name)

if opt.test_batchSize == 0:
    opt.test_batchSize = opt.batchSize

assert opt.task in ('slot_tagger', 'slot_tagger_with_focus', 'slot_tagger_with_crf')
if opt.task == 'slot_tagger_with_focus':
    opt.enc_dec, opt.crf = True, False
elif opt.task == 'slot_tagger_with_crf':
    opt.enc_dec, opt.crf = False, True
else:
    opt.enc_dec, opt.crf = False, False

if not opt.testing:
    exp_path = util.hyperparam_string(opt)
    exp_path = os.path.join(opt.experiment, exp_path) + '__alpha_%s' % (opt.st_weight)
    if opt.mini_word_freq != 2:
        exp_path += '__mwf_%s' % (opt.mini_word_freq)
    if opt.read_input_word2vec:
        exp_path += '__preEmb_in'
    if opt.enc_dec and opt.decoder_tied:
        exp_path += '__decTied'
else:
    exp_path = opt.out_path
if not os.path.exists(exp_path):
    os.makedirs(exp_path)

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


vocab_config = {'mini_word_freq':opt.mini_word_freq, 'bos_eos':opt.bos_eos}
dataroot = opt.dataroot

slot_tag_vocab_dir = dataroot+'/lab'

train_data_dir = dataroot+'/train'
valid_data_dir = dataroot+'/valid'
test_data_dir = dataroot+'/test'
if opt.testing:
    print("[testing=True] is not ready!")
    exit()
    lab_test_name = opt.test_file_name+'.lab'
    rec_test_name = opt.test_file_name+'.rec'
    lab_test_data_dir = dataroot+'/'+lab_test_name
    rec_test_data_dir = dataroot+'/'+rec_test_name

if not opt.testing:
    slot_tag_to_idx, idx_to_slot_tag = data_reader.read_vocab_file(slot_tag_vocab_dir, bos_eos=opt.enc_dec)
    word_to_idx, idx_to_word = data_reader.read_vocab_from_data_file(train_data_dir, vocab_config=vocab_config)
else:
    slot_tag_to_idx, idx_to_slot_tag = data_reader.read_vocab_file(opt.read_vocab+'.slot_tag', bos_eos=False, no_pad=True, no_unk=True)
    word_to_idx, idx_to_word = data_reader.read_vocab_file(opt.read_vocab+'.in', bos_eos=False, no_pad=True, no_unk=True)

if not opt.testing and opt.read_input_word2vec:
    # pretrained-embedding initialization for training
    ext_word_to_idx, ext_word_emb = read_wordEmb.read_word2vec_inText(opt.read_input_word2vec, opt.device)
    for word in ext_word_to_idx:
        if word not in word_to_idx:
            idx = len(word_to_idx)
            word_to_idx[word] = idx
            idx_to_word[idx] = word

logger.info("Vocab size: %s %s" % (len(word_to_idx), len(slot_tag_to_idx)))
if not opt.testing:
    data_reader.save_vocab(idx_to_word, os.path.join(exp_path, opt.save_vocab+'.in'))
    data_reader.save_vocab(idx_to_slot_tag, os.path.join(exp_path, opt.save_vocab+'.slot_tag'))

if not opt.testing:
    train_feats, train_slot_tags = data_reader.read_seqtag_data_with_unali_act(train_data_dir, word_to_idx, slot_tag_to_idx)
    valid_feats, valid_slot_tags = data_reader.read_seqtag_data_with_unali_act(valid_data_dir, word_to_idx, slot_tag_to_idx, keep_order=opt.testing)
    test_feats, test_slot_tags = data_reader.read_seqtag_data_with_unali_act(test_data_dir, word_to_idx, slot_tag_to_idx, keep_order=opt.testing)
else:
    lab_test_feats, lab_test_slot_tags = data_reader.read_seqtag_data_with_unali_act(lab_test_data_dir, word_to_idx, slot_tag_to_idx, keep_order=opt.testing, raw_word=True)
    rec_test_feats, rec_test_slot_tags = data_reader.read_seqtag_data_with_unali_act(rec_test_data_dir, word_to_idx, slot_tag_to_idx, keep_order=opt.testing, raw_word=True)

if opt.task == 'slot_tagger':
    import models.slot_tagger as slot_tagger
    model_tag = slot_tagger.LSTMTagger(opt.emb_size, opt.hidden_size, len(word_to_idx), len(slot_tag_to_idx), word_to_idx['<pad>'], bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device)
elif opt.task == 'slot_tagger_with_focus':
    import models.enc_dec.slot_tagger_with_focus as slot_tagger_with_focus
    model_tag = slot_tagger_with_focus.LSTMTagger_focus(opt.emb_size, opt.hidden_size, len(word_to_idx), len(slot_tag_to_idx), word_to_idx['<pad>'], slot_tag_to_idx['<pad>'], bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device)
elif opt.task == 'slot_tagger_with_crf':
    import models.slot_tagger_crf as slot_tagger_crf
    model_tag = slot_tagger_crf.LSTMTagger_CRF(opt.emb_size, opt.hidden_size, len(word_to_idx), len(slot_tag_to_idx), word_to_idx['<pad>'], bidirectional=opt.bidirectional, num_layers=opt.num_layers, dropout=opt.dropout, device=opt.device)

model_tag = model_tag.to(opt.device)

if not opt.testing:
    #custom init (needed maybe) ...
    model_tag.init_weights(opt.init_weight)

    # pretrained-embedding initialization for training
    if opt.read_input_word2vec:
        for word in word_to_idx:
            if word in ext_word_to_idx:
                model_tag.word_embeddings.weight.data[word_to_idx[word]] = ext_word_emb[ext_word_to_idx[word]]

# read pretrained model
if opt.read_model:
    model_tag.load_model(opt.read_model+'.tag')

# output embedding tied for enc_dec
if not opt.testing and opt.enc_dec and opt.decoder_tied:
    model.hidden2tag.weight = model.tag_embeddings.weight

# loss function
weight_mask = torch.ones(len(slot_tag_to_idx), device=opt.device)
weight_mask[slot_tag_to_idx['<pad>']] = 0
slot_tag_loss_function = nn.NLLLoss(weight=weight_mask, size_average=False)

# optimizer
params = list(model_tag.parameters())
if opt.optim.lower() == 'sgd':
    optimizer = optim.SGD(params, lr=opt.lr)
elif opt.optim.lower() == 'adam':
    optimizer = optim.Adam(params, lr=opt.lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0) # (beta1, beta2)
elif opt.optim.lower() == 'adadelta':
    optimizer = optim.Adadelta(params, rho=0.95, lr=1.0)
elif opt.optim.lower() == 'rmsprop':
    optimizer = optim.RMSprop(params, lr=opt.lr)

def decode(data_feats, data_slot_tags, output_path):
    data_index = np.arange(len(data_feats))
    losses = []
    TP, FP, FN, TN = 0.0, 0.0, 0.0, 0.0
    with open(output_path, 'w') as f:
        for j in range(0, len(data_index), opt.test_batchSize):
            if opt.testing:
                inputs, slot_tags, lens, line_nums, raw_words = data_reader.get_minibatch_with_unali_act(data_feats, data_slot_tags, word_to_idx, slot_tag_to_idx, data_index, j, opt.test_batchSize, add_start_end=opt.bos_eos, keep_order=opt.testing, raw_word=True, enc_dec_focus=opt.enc_dec, device=opt.device)
            else:
                inputs, slot_tags, lens = data_reader.get_minibatch_with_unali_act(data_feats, data_slot_tags, word_to_idx, slot_tag_to_idx, data_index, j, opt.test_batchSize, add_start_end=opt.bos_eos, keep_order=opt.testing, raw_word=False, enc_dec_focus=opt.enc_dec, device=opt.device)

            # slot tag
            if opt.enc_dec:
                opt.greed_decoding = False #True
                if opt.greed_decoding:
                    slot_tag_scores_1best, pred_slot_tag_1best, h_t_c_t = model_tag.decode_greed(inputs, slot_tags[:, 0:1], lens)
                    slot_tag_loss = slot_tag_loss_function(slot_tag_scores_1best.contiguous().view(-1, len(slot_tag_to_idx)), slot_tags[:, 1:].contiguous().view(-1))
                    pred_slot_tag_1best = pred_slot_tag_1best.cpu().numpy()
                else:
                    beam_size = 2
                    beam_tag_scores_1best, pred_slot_tag_1best, _ = model_tag.decode_beam_search(inputs, lens, beam_size, slot_tag_to_idx)
                    ppl = beam_tag_scores_1best.cpu() / torch.tensor(lens, dtype=torch.float)
                    slot_tag_loss = ppl.exp().sum()
                    pred_slot_tag_1best = [[word[0].item() for word in line] for line in pred_slot_tag_1best]
                slot_tags = slot_tags[:, 1:].data.cpu().numpy()
            elif opt.crf:
                max_len = max(lens)
                masks = [([1] * l) + ([0] * (max_len - l)) for l in lens]
                masks = torch.tensor(masks, dtype=torch.uint8, device=opt.device)
                crf_feats, h_t_c_t = model_tag._get_lstm_features(inputs, lens)
                slot_tag_path_scores, slot_tag_path = model_tag.forward(crf_feats, masks)
                slot_tag_loss = model_tag.neg_log_likelihood(crf_feats, masks, slot_tags)
                pred_slot_tag_1best = slot_tag_path.data.cpu().numpy()
                slot_tags = slot_tags.data.cpu().numpy()
            else:
                slot_tag_scores, h_t_c_t = model_tag(inputs, lens)
                slot_tag_loss = slot_tag_loss_function(slot_tag_scores.contiguous().view(-1, len(slot_tag_to_idx)), slot_tags.view(-1))
                pred_slot_tag_1best = slot_tag_scores.data.cpu().numpy().argmax(axis=-1)
                slot_tags = slot_tags.data.cpu().numpy()
            
            losses.append(slot_tag_loss.item()/sum(lens))

            inputs = inputs.data.cpu().numpy()
            for idx, pred_line in enumerate(pred_slot_tag_1best):
                length = lens[idx]
                # slot tag
                pred_seq = []
                for slot_tag in pred_line[:length]:
                    slot_tag = idx_to_slot_tag[slot_tag]
                    pred_seq.append(slot_tag)
                lab_seq = []
                for slot_tag in slot_tags[idx][:length]:
                    slot_tag = idx_to_slot_tag[slot_tag]
                    lab_seq.append(slot_tag)
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
                if opt.testing:
                    input_line = raw_words[idx]
                else:
                    input_line = [idx_to_word[word] for word in inputs[idx]][:length]
                word_tag_line = [input_line[_idx]+':'+pred_seq[_idx] for _idx in range(len(input_line))]

                if opt.testing:
                    f.write(str(line_nums[idx])+' : '+' '.join(word_tag_line)+'\n')
                else:
                    f.write(' '.join(word_tag_line)+'\n')

    if TP == 0:
        p, r, f = 0, 0, 0
    else:
        p, r, f = 100*TP/(TP+FP), 100*TP/(TP+FN), 100*2*TP/(2*TP+FN+FP)
    
    mean_losses = np.mean(losses, axis=0)
    return mean_losses, p, r, f

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
        
        for j in range(0, len(train_data_index), opt.batchSize):
            inputs, slot_tags, lens = data_reader.get_minibatch_with_unali_act(train_feats['data'], train_slot_tags['data'], word_to_idx, slot_tag_to_idx, train_data_index, j, opt.batchSize, add_start_end=opt.bos_eos, enc_dec_focus=opt.enc_dec, device=opt.device)
            optimizer.zero_grad()
            # slot tag
            if opt.enc_dec:
                slot_tag_scores, h_t_c_t = model_tag(inputs, slot_tags[:, :-1], lens)
                slot_tag_loss = slot_tag_loss_function(slot_tag_scores.contiguous().view(-1, len(slot_tag_to_idx)), slot_tags[:, 1:].contiguous().view(-1))
            elif opt.crf:
                max_len = max(lens)
                masks = [([1] * l) + ([0] * (max_len - l)) for l in lens]
                masks = torch.tensor(masks, dtype=torch.uint8, device=opt.device)
                crf_feats, h_t_c_t = model_tag._get_lstm_features(inputs, lens)
                slot_tag_loss = model_tag.neg_log_likelihood(crf_feats, masks, slot_tags)
            else:
                slot_tag_scores, h_t_c_t = model_tag(inputs, lens)
                slot_tag_loss = slot_tag_loss_function(slot_tag_scores.contiguous().view(-1, len(slot_tag_to_idx)), slot_tags.view(-1))
            losses.append(slot_tag_loss.item()/sum(lens))
            
            total_loss = slot_tag_loss
            total_loss.backward()
            
            # Clips gradient norm of an iterable of parameters.
            if opt.max_norm > 0:
                torch.nn.utils.clip_grad_norm_(params, opt.max_norm)
            
            optimizer.step()
        
        mean_loss = np.mean(losses, axis=0)
        logger.info('Training:\tEpoch : %d\tTime : %.4fs\tLoss of slot tag : %.5f' % (i, time.time() - start_time, mean_loss))
        gc.collect()

        # Evaluation
        model_tag.eval()
        start_time = time.time()
        loss_val, p_val, r_val, f_val = decode(valid_feats['data'], valid_slot_tags['data'], os.path.join(exp_path, 'valid.iter'+str(i)))
        logger.info('Validation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f' % (i, time.time() - start_time, loss_val, f_val))
        start_time = time.time()
        loss_te, p_te, r_te, f_te = decode(test_feats['data'], test_slot_tags['data'], os.path.join(exp_path, 'test.iter'+str(i)))
        logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f' % (i, time.time() - start_time, loss_te, f_te))

        if best_f1 < f_val:
            model_tag.save_model(os.path.join(exp_path, opt.save_model+'.tag'))
            best_f1 = f_val
            logger.info('NEW BEST:\tEpoch : %d\tbest valid F1 : %.5f\ttest F1 : %.5f ' % (i, f_val, f_te))
            best_result['iter'] = i
            best_result['vf1'], best_result['vce'] = f_val, loss_val
            best_result['tf1'], best_result['tce'] = f_te, loss_te
    logger.info('BEST RESULT: \tEpoch : %d\tbest valid (Loss: %.5f F1 : %.5f)\tbest test (Loss: %.5f F1 : %.5f) ' % (best_result['iter'], best_result['vce'], best_result['vf1'], best_result['tce'], best_result['tf1']))
else:    
    logger.info("Online testing starts at %s" % (time.asctime(time.localtime(time.time()))))
    model_tag.eval()
    start_time = time.time()
    loss_te, p_te, r_te, f_te = decode(lab_test_feats['data'], lab_test_slot_tags['data'], os.path.join(exp_path, 'test.lab'))
    logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f' % (0, time.time() - start_time, loss_te, f_te))
    start_time = time.time()
    loss_te, p_te, r_te, f_te = decode(rec_test_feats['data'], rec_test_slot_tags['data'], os.path.join(exp_path, 'test.rec'))
    logger.info('Evaluation:\tEpoch : %d\tTime : %.4fs\tLoss : %.5f\tFscore : %.5f' % (0, time.time() - start_time, loss_te, f_te))
    
    if opt.save_model_to_cpu:
        torch.save(model_tag.cpu(), os.path.join(exp_path, opt.save_model+'.cpu.tag'))

