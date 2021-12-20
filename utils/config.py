import os
from collections import OrderedDict
import json
from datetime import datetime

def get_timestamp():
    return datetime.now().strftime('%y%m%d_%H%M%S')

def parse(args):
    opt_path = args.config
    # remove comments starting with '//'
    json_str = ''
    with open(opt_path, 'r') as f:
        for line in f:
            line = line.split('//')[0] + '\n'
            json_str += line
    opt = json.loads(json_str, object_pairs_hook=OrderedDict)

    # set log directory
    if args.debug:
        opt['name'] = 'debug_{}'.format(opt['name'])
    experiments_root = os.path.join(
        'experiments', '{}_{}'.format(opt['name'], get_timestamp()))
    opt['path']['experiments_root'] = experiments_root
    for key, path in opt['path'].items():
        if 'resume' not in key and 'experiments' not in key:
            opt['path'][key] = os.path.join(experiments_root, path)
            os.makedirs(opt['path'][key])


    # export CUDA_VISIBLE_DEVICES
    if opt['gpu_ids'] is not None:
        gpu_list = ','.join(str(x) for x in opt['gpu_ids'])
    else:
        gpu_list = ''

    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    print('export CUDA_VISIBLE_DEVICES=' + gpu_list)
    if len(gpu_list) > 1:
        opt['distributed'] = True
    else:
        opt['distributed'] = False

    # debug
    if args.debug:
        opt['train']['n_iter'] = 8
        opt['train']['val_freq'] = 4
        opt['train']['print_freq'] = 2
        opt['train']['save_checkpoint_freq'] = 4
        opt['datasets']['train']['batch_size'] = 4
        opt['datasets']['val']['batch_size'] = 4
        opt['model']['beta_schedule']['train']['n_timestep'] = 10
        opt['model']['beta_schedule']['val']['n_timestep'] = 10
        opt['datasets']['train']['data_len'] = 8
        opt['datasets']['val']['data_len'] = 4

    return opt

def dict2str(opt, indent_l=1):
    '''dict to string for logger'''
    msg = ''
    for k, v in opt.items():
        if isinstance(v, dict):
            msg += ' ' * (indent_l * 2) + k + ':[\n'
            msg += dict2str(v, indent_l + 1)
            msg += ' ' * (indent_l * 2) + ']\n'
        else:
            msg += ' ' * (indent_l * 2) + k + ': ' + str(v) + '\n'
    return msg


