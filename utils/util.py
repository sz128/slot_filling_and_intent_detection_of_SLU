import os

def hyperparam_string(options):
    """Hyerparam string."""
    task_path = 'model_%s' % (options.task)
    dataset_path = 'data_%s' % (options.dataset)
    
    exp_name = ''
    exp_name += 'bidir_%s__' % (options.bidirectional)
    if options.__contains__("emb_size"):
        exp_name += 'emb_dim_%s__' % (options.emb_size)
    exp_name += 'hid_dim_%s_x_%s__' % (options.hidden_size, options.num_layers)
    exp_name += 'bs_%s__' % (options.batchSize)
    exp_name += 'dropout_%s__' % (options.dropout)
    if options.__contains__("optim"):
        exp_name += 'optimizer_%s__' % (options.optim)
    exp_name += 'lr_%s__' % (options.lr)
    if options.__contains__("max_norm"):
        exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'me_%s' % (options.max_epoch)

    return os.path.join(task_path, dataset_path, exp_name)
