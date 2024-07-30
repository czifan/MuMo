import torch
import torch.nn as nn
from utils.print_utils import *
from utils.criterions.smoothing_loss import *
from utils.criterions.blyx_loss import *

def build_criterion(opts, class_weights, printer=print):
    criterion = None
    if opts.loss_fn == 'ce':
        if opts.label_smoothing:
            criterion = CrossEntropyWithLabelSmoothing(ls_eps=opts.label_smoothing_eps)
            print_log_message('Using label smoothing value of : \n\t{}'.format(opts.label_smoothing_eps), printer)
        else:
            if opts.loss_weight:
                criterion = nn.CrossEntropyLoss(weight=class_weights)
                class_wts_str = '\n\t'.join(['{} --> {:.3f}'.format(cl_id, class_weights[cl_id]) for cl_id in range(class_weights.size(0))])
                print_log_message('Using class-weights: \n\t{}'.format(class_wts_str), printer)
            else:
                criterion = nn.CrossEntropyLoss()
    elif opts.loss_fn == 'bce':
        criterion = nn.BCEWithLogitsLoss(pos_weight=class_weights)
        class_wts_str = '\n\t'.join(
            ['{} --> {:.3f}'.format(cl_id, class_weights[cl_id]) for cl_id in range(class_weights.size(0))])
        print_log_message('Using class-weights: \n\t{}'.format(class_wts_str), printer)
    elif "BLYX" in opts.loss_fn:
        criterion = eval(opts.loss_fn)()
    elif "YX" in opts.loss_fn:
        criterion = eval(opts.loss_fn)()
    elif "BL" in opts.loss_fn:
        criterion = eval(opts.loss_fn)()
    else:
        print_error_message('{} critiria not yet supported')

    if criterion is None:
        print_error_message('Criteria function cannot be None. Please check', printer)

    return criterion

def get_criterion_opts(parser):
    group = parser.add_argument_group("Criterion options")
    group.add_argument("--loss-fn", default="ce", help="Loss function")
    group.add_argument("--loss-weight", action="store_true", default=False, help="Weighted loss or not")
    group.add_argument("--label-smoothing", action="store_true", default=False, help="Smooth labels or not")
    group.add_argument("--label-smoothing-eps", default=0.1, type=float, help="Epsilon for label smoothing")
    return parser