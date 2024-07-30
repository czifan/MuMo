import json
import numpy as np
from utils.print_utils import *
import argparse
import glob 
import torch 
import random
import time 
import logging
import matplotlib.pyplot as plt 
import os 

def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)

    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'

    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.manual_seed(seed)

    torch.use_deterministic_algorithms(True)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = False
    torch.backends.cudnn.benchmark = False

def build_logging(filename):
    logging.basicConfig(level=logging.DEBUG,
                        format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s',
                        datefmt='%m-%d %H:%M',
                        filename=filename,
                        filemode='w')
    console = logging.StreamHandler()
    console.setLevel(logging.INFO)
    formatter = logging.Formatter('%(name)-12s: %(levelname)-8s %(message)s')
    console.setFormatter(formatter)
    logging.getLogger('PIL').setLevel(logging.INFO)
    logging.getLogger('').addHandler(console)
    return logging

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NumpyEncoder, self).default(obj)

class ColorEncoder(object):
    def __init__(self):
        super().__init__()

    def get_colors(self, dataset_name):
        if dataset_name == 'bingli':
            class_colors = [
                (228/ 255.0, 26/ 255.0, 28/ 255.0),
                (55/ 255.0, 126/ 255.0, 184/ 255.0),
                #(77/ 255.0, 175/ 255.0, 74/ 255.0),
                #(152/ 255.0, 78/ 255.0, 163/ 255.0)
            ]

            class_linestyle = ['solid', 'solid']

            return class_colors, class_linestyle
        else:
            raise NotImplementedError

class DictWriter(object):
    def __init__(self, file_name, format='csv'):
        super().__init__()
        assert format in ['csv', 'json', 'txt']

        self.file_name = '{}.{}'.format(file_name, format)
        self.format = format

    def write(self, data_dict: dict):
        if self.format == 'csv':
            import csv
            with open(self.file_name, 'w', newline="") as csv_file:
                writer = csv.writer(csv_file)
                for key, value in data_dict.items():
                    writer.writerow([key, value])
        elif self.format == 'json':
            import json
            with open(self.file_name, 'w') as fp:
                json.dump(data_dict, fp, indent=4, sort_keys=True)
        else:
            with open(self.file_name, 'w') as txt_file:
                for key, value in data_dict.items():
                    line = '{} : {}\n'.format(key, value)
                    txt_file.write(line)

def save_checkpoint(epoch, model_state, optimizer_state, best_perf, save_dir, is_best, metric, keep_best_k_models=-1, printer=print):
    best_perf = round(best_perf, 3)
    checkpoint = {
        'epoch': epoch,
        'state_dict': model_state,
        'optim_dict': optimizer_state,
        'best_perf': best_perf
    }
    # overwrite last checkpoint everytime
    ckpt_fname = '{}/checkpoint_last.pth'.format(save_dir)
    torch.save(checkpoint, ckpt_fname)

#     if epoch % 10 == 0:
    if metric >= 0.8:
        # # write checkpoint for every epoch
        ep_ckpt_fname = '{}/model_{:03d}.pth'.format(save_dir, epoch)
        torch.save(checkpoint['state_dict'], ep_ckpt_fname)

    if keep_best_k_models > 0:
        checkpoint_files = glob.glob('{}/model_best_*')
        n_best_chkpts = len(checkpoint_files)
        if n_best_chkpts >= keep_best_k_models:
            # Extract accuracy of existing best checkpoints
            perf_tie = dict()
            for f_name in checkpoint_files:
                # first split on directory
                # second split on _
                # 3rd split on pth
                perf = float(f_name.split('/')[-1].split('_')[-1].split('.pth')[0])
                # in case multiple models have the same perf value
                if perf not in perf_tie:
                    perf_tie[perf] = [f_name]
                else:
                    perf_tie[perf].append(f_name)

            min_perf_k_checks = min(list(perf_tie.keys()))

            if best_perf >= min_perf_k_checks:
                best_ckpt_fname = '{}/model_best_{}_{}.pth'.format(save_dir, epoch, best_perf)
                torch.save(checkpoint['state_dict'], best_ckpt_fname)

                min_check_loc = perf_tie[min_auc][0]
                if os.path.isfile(min_check_loc):
                    os.remove(min_check_loc)
        else:
            best_ckpt_fname = '{}/model_best_{}_{}.pth'.format(save_dir, epoch, best_perf)
            torch.save(checkpoint['state_dict'], best_ckpt_fname)

    # save the best checkpoint
    if is_best:
        best_model_fname = '{}/model_best.pth'.format(save_dir)
        torch.save(model_state, best_model_fname)
        print_info_message('Checkpoint saved at: {}'.format(best_model_fname), printer)

    #print_info_message('Checkpoint saved at: {}'.format(ep_ckpt_fname), printer)


def load_checkpoint(ckpt_fname, device='cpu'):
    #ckpt_fname = '{}/checkpoint_last.pth'.format(checkpoint_dir)
    model_state = torch.load(ckpt_fname, map_location=device)
    return model_state

    # epoch = checkpoint['epoch']
    # model_state = checkpoint['state_dict']
    # optim_state = checkpoint['optim_dict']
    # best_perf = checkpoint['best_perf']
    # return epoch, model_state, optim_state, best_perf

def save_arguments(args, save_loc, json_file_name='arguments.json', printer=print):
    argparse_dict = vars(args)
    arg_fname = '{}/{}'.format(save_loc, json_file_name)
    writer = DictWriter(file_name=arg_fname, format='json')
    writer.write(argparse_dict)
    print_log_message('Arguments are dumped here: {}'.format(arg_fname), printer)


def load_arguments(parser, dumped_arg_loc, json_file_name='arguments.json'):
    arg_fname = '{}/{}'.format(dumped_arg_loc, json_file_name)
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)

        updated_args = parser.parse_args()

    return updated_args


def load_arguments_file(parser, arg_fname):
    parser = argparse.ArgumentParser(parents=[parser], add_help=False)
    with open(arg_fname, 'r') as fp:
        json_dict = json.load(fp)
        parser.set_defaults(**json_dict)
        updated_args = parser.parse_args()

    return updated_args

def plot_results(res_dict, plot_file):
    N = len(list(res_dict.keys()))
    _, axarr = plt.subplots(1, N, figsize=(5*N, 5))
    for i, (key, value) in enumerate(res_dict.items()):
        axarr[i].plot(range(len(value)), value, label=key)
        axarr[i].legend()
    plt.savefig(plot_file)
    plt.close()