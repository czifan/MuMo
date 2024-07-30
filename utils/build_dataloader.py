import torch
import numpy as np
from utils.print_utils import *
from torch.utils.data import DataLoader
from data_loader import *

def worker_init_fn(worked_id):
    worker_seed = torch.initial_seed() % 2 ** 32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def build_data_loader(opts, printer=print):
    train_loader, val_loader = None, None 
    diag_classes = 0
    if opts.dataset == "her2":
        train_ds = BLYXDataset(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = BLYXDataset(opts, split="val", split_file=opts.val_file, printer=print, cohort="her2")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    elif opts.dataset == "her2_test":
        train_ds = BLYXDatasetV1(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = BLYXDatasetV1(opts, split="val", split_file=opts.val_file, printer=print, cohort="her2")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    elif opts.dataset == "ci":
        train_ds = BLYXDataset(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = BLYXDataset(opts, split="val", split_file=opts.val_file, printer=print, cohort="ci")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    elif opts.dataset == "her2_yx":
        train_ds = YXDataset(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = YXDataset(opts, split="val", split_file=opts.val_file, printer=print, cohort="her2")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    elif opts.dataset == "ci_yx":
        train_ds = YXDataset(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = YXDataset(opts, split="val", split_file=opts.val_file, printer=print, cohort="ci")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    elif opts.dataset == "her2_bl":
        train_ds = BLDataset(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = BLDataset(opts, split="val", split_file=opts.val_file, printer=print, cohort="her2")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    elif opts.dataset == "ci_bl":
        train_ds = BLDataset(opts, split="train", split_file=opts.train_file, printer=print, cohort=opts.train_cohort)
        val_ds = BLDataset(opts, split="val", split_file=opts.val_file, printer=print, cohort="ci")

        diag_classes = train_ds.n_classes
        #assert diag_classes == opts.n_classes, (diag_classes, opts.n_classes)
        diag_labels = train_ds.diag_labels

        train_dl = DataLoader(train_ds, batch_size=opts.batch_size, shuffle=True,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn)
        val_dl = DataLoader(val_ds, batch_size=1, shuffle=False,
                            pin_memory=True, num_workers=opts.data_workers,
                            worker_init_fn=worker_init_fn) 
    else:
        print_error_message('{} dataset not supported yet'.format(opts.dataset), printer)
    
    # compute class-weights for balancing dataset
    if opts.class_weights:
        class_weights = np.histogram(diag_labels, bins=diag_classes)[0]
        class_weights = np.array(class_weights) / sum(class_weights)
        for i in range(diag_classes):
            class_weights[i] = round(np.log(1.0 / class_weights[i]), 5)
    else:
        class_weights = np.ones(diag_classes, dtype=np.float)
    
    return train_dl, val_dl, diag_classes, class_weights

def get_dataset_opts(parser):
    group = parser.add_argument_group('Dataset general details')
    group.add_argument('--dataset', type=str, default='bingli', help='Dataset name')
    group.add_argument('--label-type', type=str, default='response')
    group.add_argument('--train-cohort', type=str, default=None)
    group.add_argument('--train-file', type=str, default=None)
    group.add_argument('--val-file', type=str, default=None)
    group.add_argument('--bl-img-dir', type=str, default='/public/share/chenzifan/Journal22-GuoZhong/GuoZhongBingLiData/', help='Dataset location')
    group.add_argument('--bl-rad-dir', type=str, default='../Data/BL_radiomics', help='Dataset location')
    group.add_argument('--bl-img-extn', type=str, default='jpg', help='Extension of WSIs. Default is tiff')
    group.add_argument('--bl-num-bags', type=int, default=10, help='Number of bags for running')
    group.add_argument('--bl-bag-size', type=int, default=2048, help='Bag size.')
    group.add_argument('--bl-word-size', type=int, default=256, help='Word size.')
    group.add_argument('--yx-img-dir', type=str, default='/public/share/chenzifan/Journal22-GuoZhong/YingXiangCropData/', help='Dataset location')
    group.add_argument('--yx-rad-dir', type=str, default='../Data/YX_radiomics', help='Dataset location')
    group.add_argument('--yx-img-extn', type=str, default='jpg', help='Extension of WSIs. Default is tiff')
    group.add_argument('--yx-num-lesions', type=int, default=4, help='Number of lesions for running')
    group.add_argument('--yx-lesion-size', type=int, default=224, help='Lesion size.')
    group.add_argument('--split-file', type=str, default='../Data/SplitData.xlsx',
                        help='Text file with training image ids and labels')
    group.add_argument('--batch-size', type=int, default=1, help='Batch size')
    group.add_argument('--data-workers', type=int, default=1, help='Number of workers for data loading')
    group.add_argument('--class-weights', action='store_true', default=False,
                        help='Compute normalized to address class-imbalance')
    return parser