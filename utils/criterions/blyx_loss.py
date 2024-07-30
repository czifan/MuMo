import torch.nn as nn
import torch 
from torch.nn import functional as F
import numpy as np
from utils.criterions.focal_loss import *
from utils.criterions.survival_loss import *

def sigmoid_rampup(current, rampup_length):
    if rampup_length == 0:
        return 1.0
    else:
        current = np.clip(current, 0.0, rampup_length)
        phase = 1.0 - current / rampup_length
        return float(np.exp(-5.0 * phase * phase))

def get_current_consistency_weight(epoch, consistency=1., consistency_rampup=40):
    return consistency * sigmoid_rampup(epoch, consistency_rampup)

class BLYXLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_criterion = FocalLoss(num_class=2)
        self.surv_criterion = DeepSurvLoss()
        self.align_criterion = FocalLoss(num_class=2)

        self.lesion_criterion = FocalLoss(num_class=5)
        self.bag_criterion = FocalLoss(num_class=4)
        self.word_criterion = FocalLoss(num_class=4)

    def forward(self, batch, results):
        ind = torch.where(batch['label'] != -1)
        loss = self.ce_criterion(results['pred'][ind], batch['label'][ind].long())

        loss1 = loss.item()

        if results['pred'].shape[0] == 1:
            return loss

        P_risk = torch.softmax(results['pred'], dim=1)[:, 0]
        T = batch['os']
        E = batch['os_censor']
        # T = torch.round(batch['pfs'] / 30.5).float()
        # E = batch['pfs_censor']
        loss2 = self.surv_criterion(P_risk, T, E)
        if not np.isnan(loss2.item()):
            loss += loss2

        blyx_flag = results["blyx_flag"].squeeze(dim=1) # (B,)
        bl_com_feat = results["bl_com_feat"][torch.where(blyx_flag)] # (B', C)
        yx_com_feat = results["yx_com_feat"][torch.where(blyx_flag)] # (B', C)
        cos_pred = torch.cosine_similarity(bl_com_feat.unsqueeze(dim=1), yx_com_feat.unsqueeze(dim=0), dim=-1).view(-1) # (B'*B')
        cos_pred = torch.stack([1.0-cos_pred, cos_pred], dim=1) # (B'*B', 2)
        cos_label = torch.eye(bl_com_feat.shape[0]).long().view(-1).to(cos_pred.device) # (B'*B')
        loss3 = self.align_criterion(cos_pred, cos_label)
        if not np.isnan(loss3.item()):
            loss += loss3

        cls_weight = 0.1 * (1.0 - get_current_consistency_weight(batch["epoch"], consistency=1.0, consistency_rampup=batch["epochs"]))

        if results["lesions_pred"] is not None:
            ind = torch.where(batch["lesions_label"].view(-1) != -1)
            lesions_pred = results["lesions_pred"].view(-1, results["lesions_pred"].shape[-1])[ind]
            lesions_label = batch["lesions_label"].view(-1)[ind]
            loss4 = self.lesion_criterion(lesions_pred, lesions_label)
            loss += loss4 * cls_weight

        if results["mask_bags_pred"] is not None:
            ind = torch.where(batch["mask_bags_label"].view(-1) != -1)
            mask_bags_pred = results["mask_bags_pred"].view(-1, results["mask_bags_pred"].shape[-1])[ind]
            mask_bags_label = batch["mask_bags_label"].view(-1)[ind]
            loss5 = self.bag_criterion(mask_bags_pred, mask_bags_label)
            loss += loss5 * cls_weight

        if results["mask_words_pred"] is not None:
            ind = torch.where(batch["mask_words_label"].view(-1) != -1)
            mask_words_pred = results["mask_words_pred"].view(-1, results["mask_words_pred"].shape[-1])[ind]
            mask_words_label = batch["mask_words_label"].view(-1)[ind]
            loss6 = self.word_criterion(mask_words_pred, mask_words_label)
            loss += loss6 * cls_weight

        return loss


class BLYXLossFN(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_criterion = FocalLoss(num_class=2)
        self.surv_criterion = DeepSurvLoss()

    def forward(self, batch, results):
        ind = torch.where(batch['label'] != -1)
        loss = self.ce_criterion(results['pred'][ind], batch['label'][ind].long())

        if results['pred'].shape[0] == 1:
            return loss

        P_risk = torch.softmax(results['pred'], dim=1)[:, 0]
        T = batch["pfs"]
        E = batch['pfs_censor']
        loss2 = self.surv_criterion(P_risk, T, E)
        if not np.isnan(loss2.item()):
            loss += loss2

        return loss
    
class BLLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_criterion = FocalLoss(num_class=2)
        self.surv_criterion = DeepSurvLoss()
        self.bag_criterion = FocalLoss(num_class=4)
        self.word_criterion = FocalLoss(num_class=4)

    def forward(self, batch, results):
        ind = torch.where(batch['label'] != -1)
        loss = self.ce_criterion(results['pred'][ind], batch['label'][ind].long())

        loss1 = loss.item()

        if results['pred'].shape[0] == 1:
            return loss

        P_risk = torch.softmax(results['pred'], dim=1)[:, 0]
        T = batch['pfs']
        E = batch['pfs_censor']
        # T = torch.round(batch['pfs'] / 30.5).float()
        # E = batch['pfs_censor']
        loss2 = self.surv_criterion(P_risk, T, E)
        if not np.isnan(loss2.item()):
            loss += loss2

        cls_weight = 0.1 * (1.0 - get_current_consistency_weight(batch["epoch"], consistency=1.0, consistency_rampup=batch["epochs"]))

        if results["mask_bags_pred"] is not None:
            ind = torch.where(batch["mask_bags_label"].view(-1) != -1)
            mask_bags_pred = results["mask_bags_pred"].view(-1, results["mask_bags_pred"].shape[-1])[ind]
            mask_bags_label = batch["mask_bags_label"].view(-1)[ind]
            loss5 = self.bag_criterion(mask_bags_pred, mask_bags_label)
            loss += loss5 * cls_weight

        if results["mask_words_pred"] is not None:
            ind = torch.where(batch["mask_words_label"].view(-1) != -1)
            mask_words_pred = results["mask_words_pred"].view(-1, results["mask_words_pred"].shape[-1])[ind]
            mask_words_label = batch["mask_words_label"].view(-1)[ind]
            loss6 = self.word_criterion(mask_words_pred, mask_words_label)
            loss += loss6 * cls_weight

        return loss
    
class YXLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.ce_criterion = FocalLoss(num_class=2)
        self.surv_criterion = DeepSurvLoss()

        self.lesion_criterion = FocalLoss(num_class=5)

    def forward(self, batch, results):
        ind = torch.where(batch['label'] != -1)
        loss = self.ce_criterion(results['pred'][ind], batch['label'][ind].long())

        loss1 = loss.item()

        if results['pred'].shape[0] == 1:
            return loss

        P_risk = torch.softmax(results['pred'], dim=1)[:, 0]
        T = batch['os']
        E = batch['os_censor']
        # T = torch.round(batch['pfs'] / 30.5).float()
        # E = batch['pfs_censor']
        loss2 = self.surv_criterion(P_risk, T, E)
        if not np.isnan(loss2.item()):
            loss += loss2

        cls_weight = 0.1 * (1.0 - get_current_consistency_weight(batch["epoch"], consistency=1.0, consistency_rampup=batch["epochs"]))

        if results["lesions_pred"] is not None:
            ind = torch.where(batch["lesions_label"].view(-1) != -1)
            lesions_pred = results["lesions_pred"].view(-1, results["lesions_pred"].shape[-1])[ind]
            lesions_label = batch["lesions_label"].view(-1)[ind]
            loss4 = self.lesion_criterion(lesions_pred, lesions_label)
            loss += loss4 * cls_weight

        return loss