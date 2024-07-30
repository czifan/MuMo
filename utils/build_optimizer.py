import torch
from torch import optim
from utils.print_utils import *
import os 

def build_optimizer(opts, model, printer=print):
    optimizer = None

    if opts.finetune:
        params = [
            {"params": [p for n, p in model.named_parameters() if "classifier." in n], "lr": opts.lr * 10.0},
            {"params": [p for n, p in model.named_parameters() if "classifier." not in n], "lr": opts.lr},
        ]
        # params = [
        #     {"params": [p for n, p in model.named_parameters() if not(("bl_model." in n) or ("yx_model." in n))], "lr": opts.lr * 100.0},
        #     {"params": [p for n, p in model.named_parameters() if "bl_model." in n], "lr": opts.lr},
        #     {"params": [p for n, p in model.named_parameters() if "yx_model." in n], "lr": opts.lr},
        # ]
    else:
        params = [p for p in model.parameters() if p.requires_grad]

    if opts.optim == 'sgd':
        print_info_message('Using SGD optimizer', printer)
        optimizer = optim.SGD(params, lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay)
    elif opts.optim == 'adam':
        print_info_message('Using ADAM optimizer', printer)
        beta1 = 0.9 if opts.adam_beta1 is None else opts.adam_beta1
        beta2 = 0.999 if opts.adam_beta2 is None else opts.adam_beta2
        optimizer = optim.Adam(
            params,
            lr=opts.lr,
            betas=(beta1, beta2),
            weight_decay=opts.weight_decay,
            eps=1e-9)
    elif opts.optim == "adamw":
        print_info_message('Using ADAMW optimizer', printer)
        beta1 = 0.9 if opts.adam_beta1 is None else opts.adam_beta1
        beta2 = 0.999 if opts.adam_beta2 is None else opts.adam_beta2
        optimizer = optim.AdamW(
            params,
            lr=opts.lr,
            betas=(beta1, beta2),
            weight_decay=opts.weight_decay,
            eps=1e-9)
    else:
        print_error_message('{} optimizer not yet supported'.format(opts.optim), printer)

    # sanity check to ensure that everything is fine
    if optimizer is None:
        print_error_message('Optimizer cannot be None. Please check', printer)

    return optimizer

def update_optimizer(optimizer, lr_value):
    optimizer.param_groups[0]['lr'] = lr_value
    return optimizer

def read_lr_from_optimzier(optimizer):
    return optimizer.param_groups[0]['lr']

def get_optimizer_opts(parser):
    'Loss function details'
    group = parser.add_argument_group('Optimizer options')
    group.add_argument('--optim', default='sgd', type=str, help='Optimizer')
    group.add_argument('--momentum', default=0.8,  type=float, help='Momentum for SGD')
    group.add_argument('--adam-beta1', default=0.9, type=float, help='Beta1 for ADAM')
    group.add_argument('--adam-beta2', default=0.999,  type=float, help='Beta2 for ADAM')
    group.add_argument('--lr', default=0.0005, type=float, help='Initial learning rate for the optimizer')
    group.add_argument('--weight-decay', default=4e-6, type=float, help='Weight decay')

    group =  parser.add_argument_group('Optimizer accumulation options')
    group.add_argument('--accum-count', type=int, default=1, help='After how many iterations shall we update the weights')

    return parser

