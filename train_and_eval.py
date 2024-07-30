import torch
import numpy as np
from utils.print_utils import *
import os
from utils.lr_scheduler import get_lr_scheduler
from utils.metric_utils import *
import gc
from utils.utils import *
from utils.build_dataloader import build_data_loader
from utils.build_model import build_model
from utils.build_optimizer import build_optimizer, update_optimizer, read_lr_from_optimzier
from utils.build_criterion import build_criterion
from utils.build_backbone import BaseFeatureExtractor
import numpy as np
import math
import json
from sklearn.metrics import roc_auc_score
from lifelines.utils import concordance_index
import pandas as pd
from copy import deepcopy
import torch
# torch.autograd.set_detect_anomaly(True)

def _concordance_index(T, P, E):
    P = [p for i, p in enumerate(P) if not np.isnan(T[i])]
    E = [e for i, e in enumerate(E) if not np.isnan(T[i])]
    T = [t for i, t in enumerate(T) if not np.isnan(T[i])]
    return concordance_index(T, P, E)

def my_concordance_index(data):
    try:
        os_cindex = _concordance_index(data['OS'].values, data['pred'].values, data['OSCensor'].values)
        pfs_cindex = _concordance_index(data['PFS'].values, data['pred'].values, data['PFSCensor'].values)
    except:
        os_cindex = -1
        pfs_cindex = -1
    return os_cindex, pfs_cindex

def my_roc_auc_score(data):
    label = data['label'].values
    pred = data['pred'].values
    ind = np.where(label != -1)
    try:
        auc = roc_auc_score(label[ind], pred[ind])
    except:
        auc = -1
    return auc

def compute_metric(output, target, is_show=False):
    with torch.no_grad():
        y_pred = output.detach().cpu().numpy()
        y_true = target.detach().cpu().numpy()
        ind = np.where(y_true != -1)
        y_pred = y_pred[ind]
        y_true = y_true[ind]

        if is_show:
            y_pred_true = []
            for i in range(y_pred.shape[0]):
                y_pred_true.append((y_pred[i], y_true[i]))
            y_pred_true = sorted(y_pred_true, key=lambda x: x[0])
            for yp, yt in y_pred_true:
                print(yp, yt)
        try:
            return roc_auc_score(y_true, y_pred)
        except:
            return -1.0

class Trainer(object):
    '''This class implemetns the training and validation functionality for training ML model for medical imaging'''

    def __init__(self, opts, printer):
        super().__init__()
        self.opts = opts
        self.best_auc = 0
        self.start_epoch = 1
        self.printer = printer
        self.global_setter()

    def global_setter(self):
        # self.setup_logger()
        self.setup_device()
        self.setup_dataloader()
        self.setup_model_optimizer_lossfn()
        self.setup_lr_scheduler()

    def setup_device(self):
        num_gpus = torch.cuda.device_count()
        self.num_gpus = num_gpus
        if num_gpus > 0:
            print_log_message('Using {} GPUs'.format(num_gpus), self.printer)
        else:
            print_log_message('Using CPU', self.printer)

        self.device = torch.device("cuda:0" if num_gpus > 0 else "cpu")
        self.use_multi_gpu = True if num_gpus > 1 else False

        if torch.backends.cudnn.is_available():
            import torch.backends.cudnn as cudnn
            cudnn.benchmark = True
            cudnn.deterministic = True

    def setup_lr_scheduler(self):
        # fetch learning rate scheduler
        #self.lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=0.98, verbose=True)
        self.lr_scheduler = get_lr_scheduler(self.opts, printer=self.printer)

    def setup_dataloader(self):
        train_loader, val_loader, diag_classes, class_weights = build_data_loader(opts=self.opts, printer=self.printer)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.diag_classes = diag_classes
        self.class_weights = torch.from_numpy(class_weights)

    def setup_model_optimizer_lossfn(self):
        # Build Model
        mi_model = build_model(opts=self.opts, printer=self.printer)

        mi_model = mi_model.to(device=self.device)
        if self.use_multi_gpu:
            mi_model = torch.nn.DataParallel(mi_model)
        self.mi_model = mi_model

        # Build Loss function
        criterion = build_criterion(opts=self.opts, class_weights=self.class_weights.float(), printer=self.printer)
        self.criterion = criterion.to(device=self.device)

        # Build optimizer
        self.optimizer = build_optimizer(model=self.mi_model, opts=self.opts, printer=self.printer)

    def training(self, epoch, epochs, lr, *args, **kwargs):
        train_stats = Statistics(printer=self.printer)

        self.mi_model.train()
        self.optimizer.zero_grad()

        num_samples = len(self.train_loader)
        epoch_start_time = time.time()

        pred_diag_labels_lst, true_diag_labels_lst, loss_lst = [], [], []
        #P_risk_lst, T_risk_lst, E_risk_lst = [], [], []
        for batch_id, batch in enumerate(self.train_loader):
            for key in batch:
                if not isinstance(batch[key][0], str):
                    batch[key] = batch[key].float().to(device=self.device)
                    # if epoch == 6:
                    #     print_log_message(f"{key} {batch[key].min().item()} {batch[key].max().item()}", printer=self.printer)

            true_diag_labels = batch['label']
            results = self.mi_model(batch, opts=self.opts)
            pred_diag_labels = results['pred']
            #print_log_message(f"{results['pred'].min().item()},{results['pred'].max().item()}", printer=self.printer)
            batch['epoch'] = epoch
            batch['epochs'] = epochs

            loss = self.criterion(batch, results)
            if loss is not None:
                #with torch.autograd.detect_anomaly():
                (loss / self.opts.log_interval).backward()
                torch.nn.utils.clip_grad_norm_(self.mi_model.parameters(), max_norm=20, norm_type=2)
                loss_lst.append(loss.item())
            pred_diag_labels_lst.append(torch.softmax(pred_diag_labels, dim=1)[:, 1])
            true_diag_labels_lst.append(true_diag_labels)

            if (batch_id+1) % self.opts.log_interval == 0 or (batch_id+1) == len(self.train_loader): 
                self.optimizer.step()
                self.optimizer.zero_grad()
                auc = compute_metric(torch.cat(pred_diag_labels_lst, dim=0), torch.cat(true_diag_labels_lst, dim=0))
                train_stats.update(loss=np.mean(loss_lst), auc=auc)
                train_stats.output(epoch=epoch, batch=batch_id+1, n_batches=num_samples, start=epoch_start_time, lr=lr)

        return train_stats.avg_auc(), train_stats.avg_loss()

    def validation(self, epoch, lr, *args, **kwargs):
        val_stats = Statistics(printer=self.printer)
        self.mi_model.eval()
        num_samples = len(self.val_loader)

        black_lst = ["feat_words", "diss_words", "diss_bags", "lesions", "diss_lesions"]
        pred_save_dir = os.path.join(self.opts.save_dir, str(self.opts.seed), "pred")
        os.makedirs(pred_save_dir, exist_ok=True)
        pred_diag_labels_lst, true_diag_labels_lst, loss_lst = [], [], []
        info_lst = {}
        with torch.no_grad():
            epoch_start_time = time.time()
            for batch_id, batch in enumerate(self.val_loader):
                for key in batch:
                    if not isinstance(batch[key][0], str):
                        batch[key] = batch[key].float().to(device=self.device)

                true_diag_labels = batch['label']
                results = self.mi_model(batch, opts=self.opts)
                pred_diag_labels = results['pred']

                loss = self.criterion(batch, results)
                if loss is not None:
                    loss_lst.append(loss.item())

                pred_diag_labels_lst.append(torch.softmax(pred_diag_labels, dim=1)[:, 1])
                true_diag_labels_lst.append(true_diag_labels)
                for key, value in batch.items():
                    if key in black_lst:
                        continue
                    if key not in info_lst:
                        info_lst[key] = []
                    if isinstance(value[0], str):
                        info_lst[key].append(value[0])
                    else:
                        info_lst[key].append(value.detach().cpu().numpy()[0])

                #print(batch_id, batch["id"], pred_diag_labels_lst[-1][0].item(), true_diag_labels[0].item())
                
                torch.cuda.empty_cache()
                gc.collect()

        f = open(os.path.join(os.path.join(pred_save_dir, f'{epoch:03d}.csv')), 'w')
        f.write("id,name,blid,yxid,liaoxiao,xianshu,lianhe,PFS,PFSCensor,OS,OSCensor,label,pred\n")
        pred_diag_labels_lst_ = torch.cat(pred_diag_labels_lst, dim=0).detach().cpu().numpy()
        true_diag_labels_lst_ = torch.cat(true_diag_labels_lst, dim=0).detach().cpu().numpy()
        for key in info_lst:
            info_lst[key] = np.asarray(info_lst[key])
        for i in range(len(pred_diag_labels_lst)):
            xianshu = -1
            lianhe = -1
            if "id" in info_lst:
                f.write(f'{info_lst["id"][i]},{info_lst["name"][i]},{info_lst["bl_pid"][i]},{info_lst["yx_pid"][i]},{info_lst["liaoxiao"][i]},{xianshu},{lianhe},{info_lst["pfs"][i]},{info_lst["pfs_censor"][i]},{info_lst["os"][i]},{info_lst["os_censor"][i]},{true_diag_labels_lst_[i]},{pred_diag_labels_lst_[i]}\n')
            elif "yxid" in info_lst:
                f.write(f'None,{info_lst["name"][i]},None,{info_lst["yx_pid"][i]},{info_lst["liaoxiao"][i]},{xianshu},{lianhe},{info_lst["pfs"][i]},{info_lst["pfs_censor"][i]},{info_lst["os"][i]},{info_lst["os_censor"][i]},{true_diag_labels_lst_[i]},{pred_diag_labels_lst_[i]}\n')
            else:
                f.write(f'None,{info_lst["name"][i]},{info_lst["bl_pid"][i]},None,{info_lst["liaoxiao"][i]},{xianshu},{lianhe},{info_lst["pfs"][i]},{info_lst["pfs_censor"][i]},{info_lst["os"][i]},{info_lst["os_censor"][i]},{true_diag_labels_lst_[i]},{pred_diag_labels_lst_[i]}\n')
        f.close()

        tmp_data = pd.read_csv(os.path.join(os.path.join(pred_save_dir, f'{epoch:03d}.csv')))
        os_cindex, pfs_cindex = my_concordance_index(tmp_data)
        auc = my_roc_auc_score(tmp_data)
        #auc = roc_auc_score(tmp_data['label'].values, tmp_data['pred'].values)
        avg_loss = np.mean(loss_lst)

        print_log_message('* Validation Stats', printer=self.printer)
        print_log_message('* Loss: {:.3f}, AUC: {:3.3f}, C-index(OS): {:.3f}, C-index(PFS): {:.3f}'.format(
                        avg_loss, auc, os_cindex, pfs_cindex), printer=self.printer)
        print_log_message('Minv: {:.3f}, Maxv: {:.3f}'.format(torch.cat(pred_diag_labels_lst, dim=0).detach().cpu().numpy().min(), 
                        torch.cat(pred_diag_labels_lst, dim=0).detach().cpu().numpy().max()), printer=self.printer,)

        return auc, avg_loss

    def run(self, *args, **kwargs):
        kwargs['need_attn'] = False

        # if self.opts.warm_up:
        #     self.warm_up(args=args, kwargs=kwargs)

        eval_stats_dict = dict()
        res_dict = {
            "TrainingLoss": [],
            "TrainingAUC": [],
            "ValidationLoss": [],
            "ValidationAUC": [],
        }

        self.validation(epoch=-1, lr=self.opts.lr, args=args, kwargs=kwargs)
        for epoch in range(self.start_epoch, self.opts.epochs+1):
            epoch_lr = self.lr_scheduler.step(epoch)

            self.optimizer = update_optimizer(optimizer=self.optimizer, lr_value=epoch_lr)

            # Uncomment this line if you want to check the optimizer's LR is updated correctly
            # assert read_lr_from_optimzier(self.optimizer) == epoch_lr

            train_auc, train_loss = self.training(epoch=epoch, lr=epoch_lr, epochs=self.opts.epochs, args=args, kwargs=kwargs)
            val_auc, val_loss = self.validation(epoch=epoch, lr=epoch_lr, args=args, kwargs=kwargs)
            eval_stats_dict[epoch] = val_auc
            gc.collect()

            # remember best accuracy and save checkpoint for best model
            is_best = val_auc >= self.best_auc
            self.best_auc = max(val_auc, self.best_auc)

            model_state = self.mi_model.module.state_dict() if isinstance(self.mi_model, torch.nn.DataParallel) \
                else self.mi_model.state_dict()

            optimizer_state = self.optimizer.state_dict()

            save_checkpoint(epoch=epoch,
                            model_state=model_state,
                            optimizer_state=optimizer_state,
                            best_perf=self.best_auc,
                            save_dir=self.opts.save_dir,
                            is_best=is_best,
                            keep_best_k_models=self.opts.keep_best_k_models,
                            printer=self.printer,
                            metric=val_auc,
                            )
            
#             if epoch % 10 == 0:
#                 save_checkpoint(epoch=epoch,
#                                 model_state=model_state,
#                                 optimizer_state=optimizer_state,
#                                 best_perf=self.best_auc,
#                                 save_dir=self.opts.save_dir,
#                                 is_best=is_best,
#                                 keep_best_k_models=self.opts.keep_best_k_models,
#                                 printer=self.printer,
#                                 )
            
            res_dict["TrainingLoss"].append(train_loss)
            res_dict["TrainingAUC"].append(train_auc)
            res_dict["ValidationLoss"].append(val_loss)
            res_dict["ValidationAUC"].append(val_auc)
            plot_results(res_dict, os.path.join(self.opts.save_dir, "plot.jpg"))
