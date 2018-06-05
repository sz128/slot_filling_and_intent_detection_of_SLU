import os

def hyperparam_string(options, tf=False):
    """Hyerparam string."""
    task_path = 'model_%s' % (options.task)
    dataset_path = 'data_%s' % (options.dataset)
    
    exp_name = ''
    exp_name += 'bidir_%s__' % (options.bidirectional)
    exp_name += 'emb_dim_%s__' % (options.emb_size)
    exp_name += 'hid_dim_%s_x_%s__' % (options.hidden_size, options.num_layers)
    exp_name += 'bs_%s__' % (options.batchSize)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'optimizer_%s__' % (options.optim)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'me_%s' % (options.max_epoch)

    if tf:
        return os.path.join('tf', task_path, dataset_path, exp_name)
    else:
        return os.path.join(task_path, dataset_path, exp_name)

def hyperparam_string_with_vc(options, tf=False):
    """Hyerparam string."""
    task_path = 'model_%s' % (options.task)
    vc_task_path = 'vc_model_%s' % (options.vc_task)
    dataset_path = 'data_%s' % (options.dataset)
    
    exp_name = ''
    exp_name += 'bidir_%s__' % (options.bidirectional)
    exp_name += 'emb_dim_%s__' % (options.emb_size)
    exp_name += 'hid_dim_%s_x_%s__' % (options.hidden_size, options.num_layers)
    exp_name += 'bs_%s__' % (options.batchSize)
    exp_name += 'dropout_%s__' % (options.dropout)
    exp_name += 'optimizer_%s__' % (options.optim)
    exp_name += 'lr_%s__' % (options.lr)
    exp_name += 'mn_%s__' % (options.max_norm)
    exp_name += 'me_%s' % (options.max_epoch)

    if tf:
        return os.path.join('tf', task_path, vc_task_path, dataset_path, exp_name)
    else:
        return os.path.join(task_path, vc_task_path, dataset_path, exp_name)

