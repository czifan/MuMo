from model.bl_model import *
from model.yx_model import *
from model.model import *
from utils.print_utils import *

def _load_from_checkpoint(opts, model, checkpoint, printer):
    if os.path.isfile(checkpoint):
        state_dict = torch.load(checkpoint, map_location='cpu')
        model_state_dict = model.state_dict()
        suc = 0
        freezn_keys = []
        for key, value in state_dict.items():
            if opts.finetune:
                if "classifier" in key:
                    continue
            if key in model_state_dict and model_state_dict[key].shape == value.shape:
                model_state_dict[key] = value
                freezn_keys.append(key)
                suc += 1
        res = model.load_state_dict(model_state_dict)
        print_info_message('Load from {} ({}) {}/{}'.format(checkpoint, res, suc, len(model_state_dict)), printer)
    return model

def build_model(opts, printer=print):
    model = None
    if opts.model == 'bingli':
        model = eval(opts.bl_model)(opts)
        if os.path.isfile(opts.bl_pretrained):
            printer(f'Loaded pretrained weight from {opts.bl_pretrained}')
            state_dict = torch.load(opts.bl_pretrained)
            model_state_dict = model.state_dict()
            suc = 0
            freezn_keys = []
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
                    freezn_keys.append(key)
                    suc += 1
            model.load_state_dict(model_state_dict)
            # for key, param in model.named_parameters():
            #     if key in freezn_keys:
            #         param.requires_grad = False
            printer(f'Loaded {suc}/{len(list(state_dict.keys()))} keys')  
        model = _load_from_checkpoint(opts, model, opts.bl_checkpoint, printer)
    elif opts.model == 'yingxiang':
        model = eval(opts.yx_model)(opts)
        if os.path.isfile(opts.yx_pretrained):
            printer(f'Loaded pretrained weight from {opts.yx_pretrained}')
            state_dict = torch.load(opts.yx_pretrained)
            model_state_dict = model.state_dict()
            suc = 0
            freezn_keys = []
            for key, value in state_dict.items():
                if key in model_state_dict:
                    model_state_dict[key] = value
                    freezn_keys.append(key)
                    suc += 1
            model.load_state_dict(model_state_dict)
            # for key, param in model.named_parameters():
            #     if key in freezn_keys:
            #         param.requires_grad = False
            printer(f'Loaded {suc}/{len(list(state_dict.keys()))} keys')
        model = _load_from_checkpoint(opts, model, opts.yx_checkpoint, printer)
        # for key, param in model.cnn.named_parameters():
        #     param.requires_grad = False
    elif opts.model == 'bingliyingxiang':
        base_model = opts.model
        opts.model = 'bingli'
        bl_model = build_model(opts, printer=printer)

        opts.model = 'yingxiang'
        yx_model = build_model(opts, printer=printer)
        
        opts.model = base_model
        model = eval(opts.blyx_model)(opts=opts, bl_model=bl_model, yx_model=yx_model)
        model = _load_from_checkpoint(opts, model, opts.blyx_checkpoint, printer)
    else:
        print_error_message('Model for this dataset ({}) not yet supported'.format('self.opts.dataset'), printer)

    # sanity check to ensure that everything is fine
    if model is None:
        print_error_message('Model cannot be None. Please check', printer)

    return model

def get_model_opts(parser):
    group = parser.add_argument_group('Medical Imaging Model Details')

    group.add_argument('--model', default="bingli", type=str, help='Name of model')
    group.add_argument('--feat-fusion-mode', default="parallel", type=str)

    group.add_argument('--yx-model', default="YXModelv1", type=str, help='Name of YingXiangModel')
    group.add_argument('--yx-pretrained', default='', type=str)
    group.add_argument('--yx-cnn-name', default="resnet18", type=str, help='Name of backbone')
    group.add_argument('--yx-cnn-pretrained', action='store_true', default=False)
    group.add_argument('--yx-cnn-features', type=int, default=512,
                       help='Number of cnn features extracted by the backbone')
    group.add_argument('--yx-out-features', type=int, default=128,
                       help='Number of output features after merging bags and words')
    group.add_argument('--yx-attn-heads', default=2, type=int, help='Number of attention heads')
    group.add_argument('--yx-dropout', default=0.4, type=float, help='Dropout value')
    group.add_argument('--yx-attn-dropout', default=0.2, type=float, help='Dropout value for attention')
    group.add_argument('--yx-attn-fn', type=str, default='softmax', choices=['tanh', 'sigmoid', 'softmax'],
                        help='Proability to drop bag and word attention weights')
    group.add_argument('--yx-num-way', type=int, default=4)

    group.add_argument('--bl-model', default="BLModelV1", type=str, help='Name of BingLiModel')
    group.add_argument('--bl-pretrained', default='', type=str)
    group.add_argument('--bl-cnn-name', default="resnet18", type=str, help='Name of backbone')
    group.add_argument('--bl-cnn-s', type=float, default=2.0,
                       help='Factor by which channels will be scaled. Default is 2.0 for espnetv2')
    group.add_argument('--bl-cnn-pretrained', action='store_true', default=False)
    group.add_argument('--bl-cnn-weight', default=False, type=str, help='Pretrained model')
    group.add_argument('--bl-cnn-features', type=int, default=512,
                       help='Number of cnn features extracted by the backbone')
    group.add_argument('--bl-out-features', type=int, default=128,
                       help='Number of output features after merging bags and words')
    group.add_argument('--bl-attn-heads', default=2, type=int, help='Number of attention heads')
    group.add_argument('--bl-dropout', default=0.4, type=float, help='Dropout value')
    group.add_argument('--bl-max-bsz-cnn-gpu0', type=int, default=100, help='Max. batch size on GPU0')
    group.add_argument('--bl-attn-dropout', type=float, default=0.2, help='Proability to drop bag and word attention weights')
    group.add_argument('--bl-attn-fn', type=str, default='softmax', choices=['tanh', 'sigmoid', 'softmax'],
                       help='Proability to drop bag and word attention weights')
    group.add_argument('--keep-best-k-models', type=int, default=-1)
    group.add_argument('--bl-num-way', type=int, default=6)

    group.add_argument('--n-classes', default=2, type=int, help='Number of classes')
    group.add_argument('--blyx-model', default="BLYXModelv1", type=str, help='Name of BingLiYingXiangModel')
    group.add_argument('--blyx-out-features', type=int, default=128,
                       help='Number of output features after merging bags and words')
    group.add_argument('--blyx-dropout', default=0.4, type=float, help='Dropout value')


    group.add_argument('--resume', action='store_true', default=False)
    group.add_argument('--blyx-checkpoint', default="", type=str, help='Checkpoint for resuming')
    group.add_argument('--bl-checkpoint', default="", type=str, help='Checkpoint for resuming')
    group.add_argument('--yx-checkpoint', default="", type=str, help='Checkpoint for resuming')
    return parser