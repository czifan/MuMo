import torch
import torch.nn as nn
from torch.nn import init
from model.nn_layers.ffn import FFN
from model.nn_layers.attn_layers import *
from typing import NamedTuple, Optional
from torch import Tensor
from torchvision import models
import torch.nn.functional as F
from copy import deepcopy
from model.feature_extractors.mnasnet import MNASNet
from model.nn_layers.transformer import *

class YXModel(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts

        if opts.yx_cnn_name == "mnasnet":
            s = 1.0
            weight = 'checkpoints/mnasnet_s_1.0_imagenet_224x224.pth'
            backbone = MNASNet(alpha=s)
            pretrained_dict = torch.load(weight, map_location=torch.device('cpu'))
            backbone.load_state_dict(pretrained_dict)
            del backbone.classifier
            self.cnn = backbone
        else:
            backbone = eval(f"models.{opts.yx_cnn_name}")(pretrained=opts.yx_cnn_pretrained)
            # if opts.yx_cnn_name == "mnasnet1_0":
            if "mnasnet" in opts.yx_cnn_name:
                self.cnn = nn.Sequential(*[*list(backbone.children())[:-1], nn.AdaptiveAvgPool2d(1)])
            else:
                self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.attn_layer = nn.Conv2d(opts.yx_cnn_features+3, 1, kernel_size=1, padding=0, stride=1)

        self.cnn_project = nn.Linear(opts.yx_cnn_features, opts.yx_out_features)

        self.attn_over_lesions = nn.MultiheadAttention(embed_dim=opts.yx_out_features,
                                num_heads=1, dropout=opts.yx_dropout, batch_first=True)
        self.ffn_attn_l2p = FFN(input_dim=opts.yx_out_features, scale=2, p=opts.yx_dropout)
        self.lesions_weight_fn = nn.Linear(opts.yx_out_features, 1, bias=False)
        self.attn_dropout = nn.Dropout(p=opts.yx_attn_dropout)
        self.attn_fn = nn.Softmax(dim=-1)

        self.lesions_classifier = nn.Linear(opts.yx_out_features, 5)

        # 临床特征融入
        self.stomach_fc = nn.Linear(2, opts.yx_out_features)
        # self.ln_dis_fc = nn.Linear(9, opts.yx_out_features)
        self.ln_num_fc = nn.Linear(7, opts.yx_out_features)
        self.zl_ln_fc = nn.Linear(3, opts.yx_out_features)
        self.zl_ln_pos_fc = nn.Linear(5, opts.yx_out_features)
        self.zl_multi_fc = nn.Linear(4, opts.yx_out_features)
        self.zl_per_fc = nn.Linear(2, opts.yx_out_features)
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.yx_out_features, num_heads=1, dropout=opts.yx_dropout, batch_first=True)

        # 组学特征融入
        self.radiomics_fc = nn.Linear(736, opts.yx_out_features)
        self.radiomics_image_attn = nn.MultiheadAttention(embed_dim=opts.yx_out_features, num_heads=1, dropout=opts.yx_dropout, batch_first=True)

        #self.yx_classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

    def energy_function(self, x, weight_fn, need_attn=False):
        # x: (B, N, C)
        x = weight_fn(x).squeeze(dim=-1) # (B, N)
        energy: Tensor[Optional] = None
        if need_attn:
            energy = x
        x = self.attn_fn(x)
        x = self.attn_dropout(x)
        x = x.unsqueeze(dim=-1) # (B, N, 1)
        return x, energy

    def parallel_radiomics_clinical(self, patient_from_lesions, radiomics_feat, clinical_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=radiomics_feat)
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=clinical_feat)
        patient_from_lesions = patient_from_lesions \
                                + clinical_image_feat.squeeze(dim=1) \
                                + radiomics_image_feat.squeeze(dim=1)

        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return patient_from_lesions

    def series_radiomics_clinical(self, patient_from_lesions, radiomics_feat, clinical_feat):
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat.squeeze(dim=1) 
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat.squeeze(dim=1)
        return patient_from_lesions

    def series_clinical_radiomics(self, patient_from_lesions, radiomics_feat, clinical_feat):
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=clinical_feat)
        patient_from_lesions = patient_from_lesions + clinical_image_feat.squeeze(dim=1)
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=patient_from_lesions.unsqueeze(dim=1), value=radiomics_feat)
        patient_from_lesions = patient_from_lesions + radiomics_image_feat.squeeze(dim=1) 
        return patient_from_lesions

    def forward(self, batch, *args, **kwargs):
        lesions = batch["lesions"] # (B, N_l, 3, H, W)
        yx_flag = batch["clinical_yx_flag"] # (B, 3)

        B, N_l, C, H, W = lesions.shape
        # (B, N_l, 3, H, W) --> (B, N_l, C)
        lesions_cnn = self.cnn(lesions.view(B*N_l, C, H, W))
        yx_flag = yx_flag.view(B, 1, 3, 1, 1).repeat(1, N_l, 1, *lesions_cnn.shape[-2:])
        yx_flag = yx_flag.view(B*N_l, 3, *lesions_cnn.shape[-2:])
        attn_lesions_cnn = torch.cat([lesions_cnn, yx_flag], dim=1)
        attn_lesions_cnn = torch.sigmoid(self.attn_layer(attn_lesions_cnn))
        lesions_cnn = lesions_cnn * attn_lesions_cnn
        lesions_cnn = lesions_cnn.view(B, N_l, -1)
        lesions_cnn = self.cnn_project(lesions_cnn)

        self.info_dict = {
            "id": batch["id"][0],
            "lesions_label": batch["lesions_label"][0].detach().cpu().numpy(),
        }

        lesions_attn, lesions_attnmap = self.attn_over_lesions(key=lesions_cnn, query=lesions_cnn, value=lesions_cnn)
        self.info_dict["lesions_attnmap"] = lesions_attnmap[0].detach().cpu().numpy()
        lesions_attn_energy, lesions_attn_energy_unnorm = self.energy_function(lesions_attn, self.lesions_weight_fn)
        # (B, N_l, C) x (B, N_l, 1) --> (B, C)
        self.info_dict["lesions_weight"] = lesions_attn_energy[0, ..., 0].detach().cpu().numpy()
        patient_from_lesions = torch.sum(lesions_attn * lesions_attn_energy, dim=1)
        patient_from_lesions = self.ffn_attn_l2p(patient_from_lesions)

        lesions_pred = self.lesions_classifier(lesions_attn)

        self.lesions_attn_energy_unnorm = lesions_attn_energy_unnorm

        clinical_feat = torch.stack([
            self.stomach_fc(batch["clinical_yx_stomach"]),
            # self.ln_dis_fc(batch["clinical_yx_ln_dis"]),
            self.ln_num_fc(batch["clinical_yx_ln_num"]),
            self.zl_ln_fc(batch["clinical_yx_zl_ln"]),
            self.zl_ln_pos_fc(batch["clinical_yx_zl_ln_pos"]),
            self.zl_multi_fc(batch["clinical_yx_zl_multi"]),
            self.zl_per_fc(batch["clinical_yx_zl_per"]),
        ], dim=1) # (B, 7, C)
        
        radiomics_feat = self.radiomics_fc(batch["yx_radiomics_feat"]) # (B, N_l, 736) --> (B, N_l, C)
        
        if self.opts.feat_fusion_mode == "parallel":
            patient_from_lesions = self.parallel_radiomics_clinical(patient_from_lesions, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_rc":
            patient_from_lesions = self.series_radiomics_clinical(patient_from_lesions, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_cr":
            patient_from_lesions = self.series_clinical_radiomics(patient_from_lesions, radiomics_feat, clinical_feat)
        else:
            raise NotImplementedError

        if self.opts.attnmap_weight_dir not in [None, "None"]:
            npz_dir = os.path.join(self.opts.attnmap_weight_dir, "YX")
            os.makedirs(npz_dir, exist_ok=True)
            if batch["yx_flag"][0].item():
                np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                        lesions_label=self.info_dict["lesions_label"],
                        lesions_attnmap=self.info_dict["lesions_attnmap"],
                        lesions_weight=self.info_dict["lesions_weight"],
                        clinical_weight=self.info_dict["clinical_weight"])
        
        return {
            "feat": patient_from_lesions,
            "lesions_pred": lesions_pred,
            #"yx_pred": self.yx_classifier(patient_from_lesions)
        }

class YXModelOnly(YXModel):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        self.classifier = nn.Linear(opts.yx_out_features, opts.n_classes)

    def forward(self, batch, *args, **kwargs):
        yx_results = super().forward(batch, *args, **kwargs)
        yx_pred = self.classifier(yx_results["feat"])
        return {
            "pred": yx_pred,
            "feat": yx_results["feat"],
            "lesions_pred": yx_results["lesions_pred"],
        }

class YXModelNoneClinicalRadiomics(YXModel):
    def forward(self, batch, *args, **kwargs):
        lesions = batch["lesions"] # (B, N_l, 3, H, W)
        yx_flag = batch["clinical_yx_flag"] # (B, 3)

        B, N_l, C, H, W = lesions.shape
        # (B, N_l, 3, H, W) --> (B, N_l, C)
        lesions_cnn = self.cnn(lesions.view(B*N_l, C, H, W))
        yx_flag = yx_flag.view(B, 1, 3, 1, 1).repeat(1, N_l, 1, *lesions_cnn.shape[-2:])
        yx_flag = yx_flag.view(B*N_l, 3, *lesions_cnn.shape[-2:])
        attn_lesions_cnn = torch.cat([lesions_cnn, yx_flag], dim=1)
        attn_lesions_cnn = torch.sigmoid(self.attn_layer(attn_lesions_cnn))
        lesions_cnn = lesions_cnn * attn_lesions_cnn
        lesions_cnn = lesions_cnn.view(B, N_l, -1)
        lesions_cnn = self.cnn_project(lesions_cnn)

        lesions_attn, lesions_attnmap = self.attn_over_lesions(key=lesions_cnn, query=lesions_cnn, value=lesions_cnn)
        lesions_attn_energy, lesions_attn_energy_unnorm = self.energy_function(lesions_attn, self.lesions_weight_fn)
        # (B, N_l, C) x (B, N_l, 1) --> (B, C)
        patient_from_lesions = torch.sum(lesions_attn * lesions_attn_energy, dim=1)
        patient_from_lesions = self.ffn_attn_l2p(patient_from_lesions)

        lesions_pred = self.lesions_classifier(lesions_attn)

        self.lesions_attn_energy_unnorm = lesions_attn_energy_unnorm
        
        return {
            "feat": patient_from_lesions,
            "lesions_pred": lesions_pred,
            #"yx_pred": self.yx_classifier(patient_from_lesions)
        }