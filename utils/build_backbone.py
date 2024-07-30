import torch
import math
from utils.print_utils import *
import os

class BaseFeatureExtractor(torch.nn.Module):
    '''
    This class calls different base feature extractors
    '''
    def __init__(self, opts, printer=print):
        '''
        :param opts: Argument list
        '''
        super(BaseFeatureExtractor, self).__init__()

        print(opts.base_extractor, 'resnet' in opts.base_extractor)

        self.printer = printer
        if opts.base_extractor == 'espnetv2':
            from model.feature_extractors.espnetv2 import EESPNet
            self.base_model = EESPNet(opts)
            self.initialize_base_model(opts.weights)
            output_feature_sz = self.base_model.classifier.in_features
            del self.base_model.classifier
        elif opts.base_extractor == 'mobilenetv2':
            from model.feature_extractors.mobilenetv2 import MobileNetV2
            self.base_model = MobileNetV2()
            self.initialize_base_model(opts.weights)
            output_feature_sz = self.base_model.last_channel
            del self.base_model.classifier
        elif opts.base_extractor == 'mnasnet':
            from model.feature_extractors.mnasnet import MNASNet
            assert opts.s == 1.0, 'We are currently supporting models with scale = 1.0. If you are interested in ' \
                                  'exploring more models, download those from PyTorch repo and use it after uncommenting ' \
                                  'this assertion. '
            self.base_model = MNASNet(alpha=opts.s)
            self.initialize_base_model(opts.weights)
            output_feature_sz = self.base_model.last_channel
            del self.base_model.classifier
        elif 'resnet' in opts.base_extractor:
            from torchvision.models import resnet
            self.base_model = eval(f'resnet.{opts.base_extractor}')(pretrained=True)
            output_feature_sz = self.base_model.fc.in_features
            del self.base_model.fc
        else:
            print_error_message('{} model not yet supported'.format(opts.base_extractor), self.printer)

        self.output_feature_sz = output_feature_sz

    def initialize_base_model(self, wts_loc):
        '''
        This function initializes the base model

        :param wts_loc: Location of the weights file
        '''
        # initialize CNN model
        if not os.path.isfile(wts_loc):
            print_error_message('No file exists here: {}'.format(wts_loc), self.printer)

        print_log_message('Loading Imagenet trained weights', self.printer)
        pretrained_dict = torch.load(wts_loc, map_location=torch.device('cpu'))
        self.base_model.load_state_dict(pretrained_dict)
        print_log_message('Loading over', self.printer)

    def forward(self, words):
        '''
        :param words: Word tensor of shape (N_w x C x w x h)
        :return: Features vector for words (N_w x F)
        '''
        assert words.dim() == 4, 'Input should be 4 dimensional tensor (B x 3 X H x W)'
        words = self.base_model(words)
        return words


def get_backbone_opts(parser):
    '''Base feature extractor CNN Model details'''
    group = parser.add_argument_group('CNN Model Details')
    group.add_argument('--base-extractor', default='espnetv2', help='Which CNN model? Default is espnetv2')
    group.add_argument('--s', type=float, default=2.0,
                       help='Factor by which channels will be scaled. Default is 2.0 for espnetv2')
    group.add_argument('--weights', type=str, default='model/model_zoo/espnetv2/espnetv2_s_2.0_imagenet_224x224.pth',
                       help='Location of imagenet pretrained weights')
    group.add_argument('--num_classes', type=int, default=1000, help='Number of classes in the base feature extractor.'
                                                                     ' Default is 1000 for the ImageNet pretrained model')
    group.add_argument('--channels', type=int, default=3,
                       help='Number of input image channesl. Default is 3 for RGB image')

    return parser
