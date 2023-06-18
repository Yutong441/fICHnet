import os
import re
import json
from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
from . import survival as surv

# --------------------summarise config information--------------------


def get_yvar(cf):
    ''' Obtain the y variable in the training set '''
    label_dir = cf['label_dir'][0]
    if re.search('.csv$', label_dir) is not None:
        label_dir = os.path.dirname(label_dir)
    label_path = cf['root']+'/'+label_dir+'/train.csv'
    df = pd.read_csv(label_path)
    yname = cf['outcome_col']
    return np.array(df[~pd.isna(df[yname])][yname])


def obtain_class_num(cf):
    ''' Output sample number in each class as a list '''
    if 'balance' in cf['loss_type']:
        yvar = get_yvar(cf)
        vec = np.arange(cf['predict_class']).reshape([1, -1])
        yvar_hot = yvar.reshape([-1, 1]) == vec
        return [int(i) for i in yvar_hot.sum(0)]


def get_weighted_sampler(ylabel, pos_label, beta=0.9999):
    '''Generate pytorch weighted random sampler for class imbalance issue'''
    n_pos = sum(ylabel >= pos_label)
    n_neg = sum(ylabel < pos_label)
    n_per_class = np.array([n_neg, n_pos])
    eff_num = 1 - beta**n_per_class
    eff_num = (1 - beta)/eff_num
    eff_weight = eff_num/np.sum(eff_num)
    weights = [eff_weight[(i >= pos_label).astype(int)] for i in ylabel]
    return torch.utils.data.WeightedRandomSampler(weights, len(weights),
                                                  replacement=True)


def postprocess(cf):
    ''' Generate secondarily derived config attributes from other existing
    attributes. It is essential to run this function before using the config
    '''
    cf['n_per_class'] = obtain_class_num(cf)
    cf["label_dir"] = [cf["outcome_col"]+"/"+i for i in [
        "val.csv", "test.csv"]]
    if cf['random_sampler']:
        yvar = get_yvar(cf)
        cf['sampler'] = get_weighted_sampler(yvar, cf['pos_label'])
        cf['shuffle'] = False  # random sampler does not work with shuffling
    else:
        cf['sampler'] = None
        cf['shuffle'] = True
    if not torch.cuda.is_available():
        cf['device'] = 'cpu'

    survival_losses = ['logistic_hazard', 'pmf', 'deephit', 'coxph', 'coxcc']
    cf['survival'] = True if cf['loss_type'] in survival_losses else False
    cf = surv.get_labtrans(cf)
    return cf

# --------------------Prepare arguments to dataloader--------------------


def get_data_args(config):
    assert len(config['data_folder']) == len(config['label_dir'])
    chan_depth = config['depth_as_channel']
    chan_depth = chan_depth or '2d' in config['model_type']

    loader_arg = [{
            'data_folder': config['data_folder'][i],
            'select_channels': config['select_channels'],
            'select_depths': config['select_depths'],
            'select_num': config['select_num'],
            'outcome_col': config ['outcome_col'],
            'index_col': config ['index_col'],
            'common_shape': config ['common_shape'],
            'output_features': config ['output_features'],
            'label_dir': config['label_dir'][i],
            'depth_as_channel': chan_depth,
            'survival': config['survival']
    } for i in range(len(config['data_folder']))]

    data_arg = {
            'num_workers': config ['num_workers'],
            'pin_memory': config ['pin_memory'],
            'batch_size': config ['batch_size'],
            'shuffle': False
    }
    train_data_arg = data_arg.copy()
    train_data_arg ['sampler']= config ['sampler']
    train_data_arg ['shuffle']= config ['shuffle']
    return loader_arg, [train_data_arg, data_arg]


def get_model_args(config):
    if 'cum_link' in config ['loss_type']: n_class = 1
    else:
        if 'ordinal' in config ['loss_type']: 
            n_class = config ['predict_class'] - 1
        else: n_class = config ['predict_class'] 
    if config['labtrans'] is not None:
        n_class = config['labtrans'].out_features

    n_decode = 0 if config['output_features'] is None else len(
            config['output_features'])
    resblock = True if config ['model_type'] == 'resunet' else False
    n_conv = int(re.sub ('conv$', '',config ['decoder'].split ('_')[0]))
    n_decoder = int(re.sub ('lin$', '',config ['decoder'].split ('_')[1]))
    if len (config ['decoder'].split ('_')) >1:
        n_pool = int(re.sub ('pool$', '',config ['decoder'].split ('_')[2]))

    in_chan = config['input_channels']
    if config['depth_as_channel'] == 'True' or '2d' in config['model_type']:
        in_chan *= config['select_depths']

    return {
        'in_channels': in_chan,
        'n_classes': n_class,
        'model_type': config['model_type'],
        'add_sigmoid': config['add_sigmoid'],
        'times_max': config['times_max'],
        'extra_features': n_decode,
        'n_conv': n_conv,
        'n_decoder': n_decoder,
        'dropout': config['dropout'],
        'step': config['step_linear'],
        'resblock': resblock,
        'n_pool': n_pool,
        'add_STN': config['add_STN'],
        'init_chan': config['init_chan'],
        'img_shape': [config['select_depths']] + config['common_shape'],
        'two_stage': config['loss_type'] == 'two_stage'
        }

# --------------------Command line interface--------------------
def trim_ws (xx):
    out = re.sub (' +$', '', xx)
    out = re.sub ('^ +', '', out)
    return out

def convert (xx):
    out = trim_ws (xx)
    try: 
        num_out = float (out)
        if re.search ('\.', out) is None: out = int (out)
        else: out = num_out
    except:
        front = re.search('^(\[|\{)', out)
        back = re.search('(\]|\})$', out)

        # convert to list or tuple
        if front is not None and back is not None:
            out = re.sub ('(\[|\{|\}|\])', '', out)
            out = out.split (',')
            try: out = [float(i) for i in out]
            except: out = out
        elif xx == 'None': out = None
        elif xx == 'False': out = False
        elif xx == 'True': out = True
        else: out = xx
    return out


def str_to_dict(dict_str, item_sep=';', key_sep='='):
    dict_list = dict_str.split(item_sep)
    odict = OrderedDict()
    for i in dict_list:
        odict[trim_ws(i.split(key_sep)[0])] = convert(i.split(key_sep)[1])
    return odict


def sub_dict(old_dict, dict_str, *args, **kwargs):
    '''
    Replace certain config elements with infor specified in a string format
    Args:
        `old_dict`: original dictionary
        `dict_str`: format in terms of 'key1=val1; key2=val2'. White space is
        permitted, double semicolon, i.e., ';;' is not. Trailing or leading
        semicolons are not permitted
    '''
    new_dict = str_to_dict(dict_str, *args, **kwargs)
    for key, val in new_dict.items():
        old_dict[key] = val
    return old_dict

# --------------------Write config logs--------------------


def log_config(filename, cfg):
    with open(filename, 'a') as f:
        f.write(print_config(cfg))
        f.write("\n================================================\n")

    with open(filename+'.json', 'w') as f: 
        cfg_write = cfg.copy()
        # only write json serializable items
        for key, val in cfg_write.items():
            if type(val) not in [int, float, list, tuple, str, dict, bool]:
                cfg_write [key] = str(type(val))
        json.dump (cfg_write, f)

def add_entry (explain, val):
    '''
    Add entry to printing config
    Args:
        `explain`: explanatory text accompanying a parameter, must contain '{}'
        somewhere
        `val`: the parameter value
    '''
    if type (val) == np.ndarray or type (val) == list:
        txt = ', '.join ([str(i) for i in val])
    else: txt = val
    return explain.format (txt)

def print_config (cf, join_str=True):
    str_list = []
    str_list.append (add_entry ('where the data is stored: {}', cf['root']))
    str_list.append (add_entry ('where the results are saved: {}',
        cf['save_prefix']))

    str_list.append ('')
    str_list.append ('# learning rate')
    str_list.append (add_entry ('number of training epochs: {}',cf ['num_epochs']))
    str_list.append (add_entry ('initial learning rate: {}',cf['lr']))
    str_list.append (add_entry ('L2 weight decay: {}',cf['L2']))
    str_list.append (add_entry ('L1 weight decay: {}',cf['L1']))
    str_list.append ('multiply the learning rate by {} every {} steps'.format
            (cf ['gamma'], cf ['step_size']))
    str_list.append (add_entry ('whether to resume training from the last saved model weights: {}',
        cf ['resume_training']))
    str_list.append (add_entry ('load previous model: {}',cf ['old_save_path']))

    str_list.append ('')
    str_list.append ('# transfer learning')
    str_list.append (add_entry ('whether to initialise with pretrained resnet18: {}',
        cf ['pretrained']))
    str_list.append (add_entry ('whether to train the pretrained weights: {}',
        cf ['train_weight']))

    str_list.append ('')
    str_list.append ('# data')
    str_list.append (add_entry ('Imaging data are stored in: {}', 
        [cf ['root'] +'/' + i for i in cf ['data_folder']] ))
    str_list.append (add_entry ('Labels are stored in: {}', 
        [cf ['root'] +'/' + i for i in cf ['label_dir']] ))
    str_list.append (add_entry ('number of channels selected from the original image: {}', 
        cf ['select_channels'] ))
    str_list.append (add_entry ('the central {} slices selected', 
        cf ['select_depths']))
    str_list.append (add_entry ('number of images selected for training and testing: {}', 
        cf['select_num']))
    str_list.append (add_entry ('number of workers to prepare images in CPU: {}',
        cf ['num_workers']))
    str_list.append (add_entry ('batch size: {}', cf ['batch_size']))
    str_list.append (add_entry ('the model is trained on: {}',cf ['device']))
    str_list.append (add_entry ('Images are padded to: {}',cf['common_shape']))
    str_list.append (add_entry ('Images are downsampled to: {}',cf['downsize']))
    str_list.append (add_entry ('Random sampling: {}',cf['random_sampler']))

    str_list.append ('')
    str_list.append ('# model')
    str_list.append (add_entry ('number of input channels to CNN: {}',
        cf['input_channels']))
    str_list.append (add_entry ('number of predicted classes: {}',
        cf ['predict_class']))
    str_list.append (add_entry ('type of model: {}', cf ['model_type']))
    str_list.append (add_entry ('type of loss: {}', cf ['loss_type']))
    if cf ['loss_type'] == 'KL':
        str_list.append (add_entry ('sigma for KL is: {}', cf ['sigma']))
    if cf ['loss_type'] == 'cum_link':
        str_list.append (add_entry ('probability distribution is: {}', 
            cf ['prob_fun']))
    str_list.append (add_entry ('whether to add sigmoid: {}',
        cf['add_sigmoid']))
    str_list.append (add_entry ('sigmoid output multiplied by: {}',
        cf['times_max']))
    str_list.append (add_entry ('organisation of the decoding layers: {}',
        cf['decoder']))
    str_list.append (add_entry ('dropout among linear layers: {}',
        cf['dropout']))
    str_list.append (add_entry ('Reducing feature number by {} in linear layers',
        cf['step_linear']))
    str_list.append (add_entry ('STN layer added: {}',
        cf['add_STN']))
    str_list.append (add_entry ('initial channel after the first layer: {}',
        cf['init_chan']))

    str_list.append ('')
    str_list.append ('# metrics')
    str_list.append (add_entry ('the target outcome is: {}', 
        cf ['outcome_col']))
    str_list.append (add_entry ('filename is located in: {}', 
        cf ['index_col']))
    str_list.append (add_entry ('the tested metrics: {}', cf ['all_metrics']))
    str_list.append (add_entry ('evaluate the metrics every {} epoch', 
        cf ['eval_every']))
    str_list.append (add_entry ('expected initial value for the metric of interest: {}',
        cf ['initial_metric']))
    str_list.append (add_entry ('on which metric to determine model improvement: {}',
        cf ['metric']))
    str_list.append (add_entry ('after {} epochs of failure of improvement the training will stop',
        cf ['tolerance']))
    str_list.append (add_entry ('whether a higher value in the metric of interest is better: {}',
        cf ['better']== 'pos'))
    str_list.append (add_entry ('predicted value equal to or above {} is classified as positive',
        cf ['pos_label']))

    str_list.append ('')
    str_list.append ('# visualise')
    str_list.append (add_entry ('the level of imaging plane in display: {}',
        cf ['level']))
    str_list.append (add_entry ('number of images to show: {}',cf ['show_num']))
    str_list.append (add_entry ('the layer from which activation map is visualised: {}',
        cf ['gcam']))
    str_list.append (add_entry ('the non-imaging features for DR are: {}',
        cf ['x_features']))
    str_list.append (add_entry ('the non-imaging features for visualisation are: {}',
        cf ['y_features']))
    str_list.append (add_entry ('the non-imaging features for neural net are: {}', 
        cf['output_features']))
    if join_str: return '\n'.join (str_list)
    else: return str_list
