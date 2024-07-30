import torch
import torch.nn as nn
from torch.nn import init
from model.nn_layers.ffn import FFN
from model.nn_layers.attn_layers import *
from typing import NamedTuple, Optional
from torch import Tensor
import math
from utils.print_utils import *
import os
import numpy as np
import torch.nn.functional as F
import random
from torchvision import models
from model.feature_extractors.mnasnet import MNASNet
from model.nn_layers.transformer import *
from tqdm import tqdm
import os 

class BLModel(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts

        if opts.bl_cnn_name == "mnasnet":
            s = 1.0
            weight = 'checkpoints/mnasnet_s_1.0_imagenet_224x224.pth'
            backbone = MNASNet(alpha=s)
            pretrained_dict = torch.load(weight, map_location=torch.device('cpu'))
            backbone.load_state_dict(pretrained_dict)
            del backbone.classifier
            self.cnn = backbone
        else:
            backbone = eval(f"models.{opts.bl_cnn_name}")(pretrained=opts.bl_cnn_pretrained)
            # if opts.bl_cnn_name == "mnasnet1_0":
            if "mnasnet" in opts.bl_cnn_name:
                self.cnn = nn.Sequential(*[*list(backbone.children())[:-1], nn.AdaptiveAvgPool2d(1)])
            else:
                self.cnn = nn.Sequential(*list(backbone.children())[:-1])

        self.attn_layer = nn.Conv2d(opts.bl_cnn_features+3, 1, kernel_size=1, padding=0, stride=1)

        self.project_words = nn.Linear(opts.bl_cnn_features, opts.bl_out_features)
        
        self.attn_over_words = nn.MultiheadAttention(embed_dim=opts.bl_out_features,
                                num_heads=1, dropout=opts.bl_dropout, batch_first=True)
        self.attn_over_bags = nn.MultiheadAttention(embed_dim=opts.bl_out_features,
                                num_heads=1, dropout=opts.bl_dropout, batch_first=True)
        self.attn_dropout = nn.Dropout(p=opts.bl_attn_dropout)

        self.bags_classifier = nn.Linear(opts.bl_out_features, 4)
        self.words_classifier = nn.Linear(opts.bl_out_features, 4)
        
        self.ffn_attn_w2b_lst = FFN(input_dim=opts.bl_out_features, scale=2, p=opts.bl_dropout)
        self.ffn_b2i = FFN(input_dim=opts.bl_out_features, scale=2, p=opts.bl_dropout)

        self.attn_fn = nn.Softmax(dim=-1)

        self.words_weight_fn = nn.Linear(opts.bl_out_features, 1, bias=False)
        self.bags_weight_fn = nn.Linear(opts.bl_out_features, 1, bias=False)

        # 临床特征融入
        # self.fenxing_fc = nn.Linear(3, opts.bl_out_features)
        # self.fenhua_fc = nn.Linear(4, opts.bl_out_features)
        self.tils_fc = nn.Linear(10, opts.bl_out_features)
        self.her2_fc = nn.Linear(4, opts.bl_out_features)
        self.tumor_fc = nn.Linear(10, opts.bl_out_features)
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.bl_out_features, num_heads=1, dropout=opts.bl_dropout, batch_first=True)

        # 组学特征融入
        self.radiomics_fc = nn.Linear(736, opts.bl_out_features)
        self.radiomics_image_attn = nn.MultiheadAttention(embed_dim=opts.bl_out_features, num_heads=1, dropout=opts.bl_dropout, batch_first=True)

        #self.bl_classifier = nn.Linear(opts.bl_out_features, opts.n_classes)

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

    def parallel_radiomics_clinical(self, image_from_bags, radiomics_feat, clinical_feat):
        radiomics_image_feat, radiomics_attnmap = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=radiomics_feat)
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags \
                                + clinical_image_feat.squeeze(dim=1) \
                                + radiomics_image_feat.squeeze(dim=1)
        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        return image_from_bags

    def series_radiomics_clinical(self, image_from_bags, radiomics_feat, clinical_feat):
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=radiomics_feat)
        image_from_bags = image_from_bags + radiomics_image_feat.squeeze(dim=1) 
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags + clinical_image_feat.squeeze(dim=1)
        return image_from_bags

    def series_clinical_radiomics(self, image_from_bags, radiomics_feat, clinical_feat):
        clinical_image_feat, _ = self.clinical_image_attn(key=clinical_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=clinical_feat)
        image_from_bags = image_from_bags + clinical_image_feat.squeeze(dim=1)
        radiomics_image_feat, _ = self.radiomics_image_attn(key=radiomics_feat,
                        query=image_from_bags.unsqueeze(dim=1), value=radiomics_feat)
        image_from_bags = image_from_bags + radiomics_image_feat.squeeze(dim=1) 
        return image_from_bags

    def forward(self, batch, *args, **kwargs):
        # if not self.training:
        # return self.incremental_inference(batch)
        words = batch['feat_words']
        bl_flag = batch["clinical_bl_flag"] # (B, 3,)

        # STEP1: Project CNN encoded words
        # (B, N_b, N_w, C, H_w, W_w) --> (B, N_b, N_w, d)
        B, N_b, N_w, C, H, W = words.shape        
        words_cnn = self.cnn(words.view(B*N_b*N_w, C, H, W))
        bl_flag = bl_flag.view(B, 1, 1, 3, 1, 1).repeat(1, N_b, N_w, 1, *words_cnn.shape[-2:])
        bl_flag = bl_flag.view(B*N_b*N_w, 3, *words_cnn.shape[-2:])
        attn_words_cnn = torch.cat([words_cnn, bl_flag], dim=1)
        attn_words_cnn = torch.sigmoid(self.attn_layer(attn_words_cnn))
        words_cnn = words_cnn * attn_words_cnn
        words_cnn = words_cnn.view(B, N_b, N_w, -1)
        words_cnn = self.project_words(words_cnn)

        self.info_dict = {
            "id": batch["id"][0],
            "bags_label": batch["mask_bags_label"][0].detach().cpu().numpy(),
            "words_label": batch["mask_words_label"][0].detach().cpu().numpy(),
        }

        # STEP2: Words to Bags (Attn words | CNN words)
        words_cnn = words_cnn.view(B*N_b, N_w, -1)

        # import os 
        # feat_dir = "../Analysis/BLFeat"
        # os.makedirs(feat_dir, exist_ok=True)
        # np.save(os.path.join(feat_dir, self.info_dict["id"]+".npz"), words_cnn.view(B, N_b, N_w, -1).detach().cpu().numpy())
        
        words_attn, words_attnmap = self.attn_over_words(key=words_cnn, query=words_cnn, value=words_cnn)
        words_attn = words_attn.view(B, N_b, N_w, -1)
        self.info_dict["words_attnmap"] = words_attnmap[0].detach().cpu().numpy()
        words_attn_energy, words_attn_energy_unnorm = self.energy_function(words_attn, self.words_weight_fn)
        # (B, N_B, N_W, C) * (B, N_B, N_W, 1) --> (B, N_B, C)
        self.info_dict["words_weight"] = words_attn_energy[0, ..., 0].detach().cpu().numpy()
        bags_from_attn_words = torch.sum(words_attn * words_attn_energy, dim=-2) 
        bags_from_attn_words = self.ffn_attn_w2b_lst(bags_from_attn_words)

        mask_words_pred = self.words_classifier(words_attn) # (B, N_B, N_W, 6)

        # STEP3: Bags to Image
        bags_attn, bags_attnmap = self.attn_over_bags(key=bags_from_attn_words, query=bags_from_attn_words, value=bags_from_attn_words)
        self.info_dict["bags_attnmap"] = bags_attnmap[0].detach().cpu().numpy()
        bags_energy, bags_energy_unnorm = self.energy_function(bags_attn, self.bags_weight_fn)
        self.info_dict["bags_weight"] = bags_energy[0, ..., 0].detach().cpu().numpy()
        image_from_bags = torch.sum(bags_attn * bags_energy, dim=-2)
        image_from_bags = self.ffn_b2i(image_from_bags)

        mask_bags_pred = self.bags_classifier(bags_attn) # (B, N_B, 6)

        clinical_feat = torch.stack([
            self.tils_fc(batch["clinical_bl_tils"]),
            self.her2_fc(batch["clinical_bl_her2"]),
            self.tumor_fc(batch["clinical_bl_tumor"]),
        ], dim=1) # (B, 5, C)

        radiomics_feat = self.radiomics_fc(batch["bl_radiomics_feat"]) # (B, M, C)

        if self.opts.feat_fusion_mode == "parallel":
            image_from_bags = self.parallel_radiomics_clinical(image_from_bags, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_rc":
            image_from_bags = self.series_radiomics_clinical(image_from_bags, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_cr":
            image_from_bags = self.series_clinical_radiomics(image_from_bags, radiomics_feat, clinical_feat)
        else:
            raise NotImplementedError

        if self.opts.attnmap_weight_dir not in [None, "None"]:
            npz_dir = os.path.join(self.opts.attnmap_weight_dir, "BL")
            os.makedirs(npz_dir, exist_ok=True)
            if batch["bl_flag"][0].item():
                np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                        bags_attnmap=self.info_dict["bags_attnmap"],
                        bags_weight=self.info_dict["bags_weight"],
                        words_attnmap=self.info_dict["words_attnmap"],
                        words_weight=self.info_dict["words_weight"],
                        clinical_weight=self.info_dict["clinical_weight"],
                        bags_label=self.info_dict["bags_label"],
                        words_label=self.info_dict["words_label"])
        
        return {
            "feat": image_from_bags,
            "mask_words_pred": mask_words_pred,
            "mask_bags_pred": mask_bags_pred,
            #"bl_pred": self.bl_classifier(image_from_bags)
        }

    def incremental_inference(self, batch, max_bags_gpu0=64, *args, **kwargs):
        words = batch['feat_words']

        # STEP1: Project CNN encoded words
        # (B, N_b, N_w, F) --> (B, N_b, N_w, d)
        B, N_b, N_w, C, H, W = words.shape
        words = words.view(B*N_b*N_w, C, H, W)
        words_cnn = []
        N = words.shape[0]
        indexes = np.arange(0, N, max_bags_gpu0)
        for i in range(len(indexes)):
            start = indexes[i]
            if i < len(indexes) - 1:
                end = indexes[i+1]
                words_batch = words[start:end]
            else:
                words_batch = words[start:] 
            words_cnn.append(self.cnn(words_batch))
        words_cnn = torch.cat(words_cnn, dim=0)
        words_cnn = words_cnn.view(B, N_b, N_w, -1)
        words_cnn = self.project_words(words_cnn)

        self.info_dict = {
            "id": batch["id"][0],
            "bags_label": batch["mask_bags_label"][0].detach().cpu().numpy(),
            "words_label": batch["mask_words_label"][0].detach().cpu().numpy(),
        }

        # STEP2: Words to Bags (Attn words | CNN words)
        words_cnn = words_cnn.view(B*N_b, N_w, -1)
        words_attn = []
        words_attnmap = []
        N = words_cnn.shape[0]
        indexes = np.arange(0, N, max_bags_gpu0)
        for i in range(len(indexes)):
            start = indexes[i]
            if i < len(indexes) - 1:
                end = indexes[i+1]
                words_cnn_batch = words_cnn[start:end]
            else:
                words_cnn_batch = words_cnn[start:]
            
            words_attn_batch, words_attnmap_batch = self.attn_over_words(key=words_cnn_batch, query=words_cnn_batch, value=words_cnn_batch)
            words_attn.append(words_attn_batch)
            words_attnmap.append(words_attnmap_batch)
        words_attn = torch.cat(words_attn, dim=0).view(B, N_b, N_w, -1)
        words_attnmap = torch.cat(words_attnmap, dim=0).view(B, N_b, N_w, N_w)
        self.info_dict["words_attnmap"] = words_attnmap[0].detach().cpu().numpy()
        words_attn_energy, words_attn_energy_unnorm = self.energy_function(words_attn, self.words_weight_fn)
        # (B, N_B, N_W, C) * (B, N_B, N_W, 1) --> (B, N_B, C)
        self.info_dict["words_weight"] = words_attn_energy[0, ..., 0].detach().cpu().numpy()
        bags_from_attn_words = torch.sum(words_attn * words_attn_energy, dim=-2) 
        bags_from_attn_words = self.ffn_attn_w2b_lst(bags_from_attn_words)

        mask_words_pred = self.words_classifier(words_attn) # (B, N_B, N_W, 6)
        
        # STEP3: Bags to Image
        bags_attn, bags_attnmap = self.attn_over_bags(key=bags_from_attn_words, query=bags_from_attn_words, value=bags_from_attn_words)
        self.info_dict["bags_attnmap"] = bags_attnmap[0].detach().cpu().numpy()
        bags_energy, bags_energy_unnorm = self.energy_function(bags_attn, self.bags_weight_fn)
        self.info_dict["bags_weight"] = bags_energy[0, ..., 0].detach().cpu().numpy()
        image_from_bags = torch.sum(bags_attn * bags_energy, dim=-2)
        image_from_bags = self.ffn_b2i(image_from_bags)

        mask_bags_pred = self.bags_classifier(bags_attn) # (B, N_B, 6)

        clinical_feat = torch.stack([
            self.tils_fc(batch["clinical_bl_tils"]),
            self.her2_fc(batch["clinical_bl_her2"]),
            self.tumor_fc(batch["clinical_bl_tumor"]),
        ], dim=1) # (B, 5, C)

        radiomics_feat = self.radiomics_fc(batch["bl_radiomics_feat"]) # (B, M, C)

        if self.opts.feat_fusion_mode == "parallel":
            image_from_bags = self.parallel_radiomics_clinical(image_from_bags, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_rc":
            image_from_bags = self.series_radiomics_clinical(image_from_bags, radiomics_feat, clinical_feat)
        elif self.opts.feat_fusion_mode == "series_cr":
            image_from_bags = self.series_clinical_radiomics(image_from_bags, radiomics_feat, clinical_feat)
        else:
            raise NotImplementedError

        
        npz_dir = "../Analysis/Attnmap_weight/BL"
        os.makedirs(npz_dir, exist_ok=True)
        np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                bags_attnmap=self.info_dict["bags_attnmap"],
                bags_weight=self.info_dict["bags_weight"],
                bags_label=self.info_dict["bags_label"],
                words_attnmap=self.info_dict["words_attnmap"],
                words_weight=self.info_dict["words_weight"],
                words_label=self.info_dict["words_label"],
                clinical_weight=self.info_dict["clinical_weight"])
        
        return {
            "feat": image_from_bags,
            "mask_words_pred": mask_words_pred,
            "mask_bags_pred": mask_bags_pred,
            #"bl_pred": self.bl_classifier(image_from_bags)
        }


class BLModel_OnlyHer2(nn.Module):
    def __init__(self, opts, *args, **kwargs):
        super().__init__()
        self.opts = opts
        self.her2_fc = nn.Linear(4, opts.bl_out_features)

    def forward(self, batch, *args, **kwargs):
        return {
            "feat": self.her2_fc(batch["clinical_bl_her2"]),
            "mask_words_pred": None,
            "mask_bags_pred": None,
        }


class BLModelOnly(BLModel):
    def __init__(self, opts, *args, **kwargs):
        super().__init__(opts, *args, **kwargs)
        self.classifier = nn.Linear(opts.bl_out_features, opts.n_classes)

    def forward(self, batch, *args, **kwargs):
        bl_results = super().forward(batch, *args, **kwargs)
        bl_pred = self.classifier(bl_results["feat"])
        return {
            "pred": bl_pred,
            "feat": bl_results["feat"],
            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }


class BLModelNoneClinicalRadiomics(BLModel):
    def forward(self, batch, *args, **kwargs):
        # if not self.training:
        # return self.incremental_inference(batch)
        words = batch['feat_words']
        bl_flag = batch["clinical_bl_flag"] # (B, 3,)

        # STEP1: Project CNN encoded words
        # (B, N_b, N_w, C, H_w, W_w) --> (B, N_b, N_w, d)
        B, N_b, N_w, C, H, W = words.shape        
        words_cnn = self.cnn(words.view(B*N_b*N_w, C, H, W))
        bl_flag = bl_flag.view(B, 1, 1, 3, 1, 1).repeat(1, N_b, N_w, 1, *words_cnn.shape[-2:])
        bl_flag = bl_flag.view(B*N_b*N_w, 3, *words_cnn.shape[-2:])
        attn_words_cnn = torch.cat([words_cnn, bl_flag], dim=1)
        attn_words_cnn = torch.sigmoid(self.attn_layer(attn_words_cnn))
        words_cnn = words_cnn * attn_words_cnn
        words_cnn = words_cnn.view(B, N_b, N_w, -1)
        words_cnn = self.project_words(words_cnn)

        # STEP2: Words to Bags (Attn words | CNN words)
        words_cnn = words_cnn.view(B*N_b, N_w, -1)

        words_attn, words_attnmap = self.attn_over_words(key=words_cnn, query=words_cnn, value=words_cnn)
        words_attn = words_attn.view(B, N_b, N_w, -1)
        words_attn_energy, words_attn_energy_unnorm = self.energy_function(words_attn, self.words_weight_fn)
        # (B, N_B, N_W, C) * (B, N_B, N_W, 1) --> (B, N_B, C)
        bags_from_attn_words = torch.sum(words_attn * words_attn_energy, dim=-2) 
        bags_from_attn_words = self.ffn_attn_w2b_lst(bags_from_attn_words)

        mask_words_pred = self.words_classifier(words_attn) # (B, N_B, N_W, 6)

        # STEP3: Bags to Image
        bags_attn, bags_attnmap = self.attn_over_bags(key=bags_from_attn_words, query=bags_from_attn_words, value=bags_from_attn_words)
        bags_energy, bags_energy_unnorm = self.energy_function(bags_attn, self.bags_weight_fn)
        image_from_bags = torch.sum(bags_attn * bags_energy, dim=-2)
        image_from_bags = self.ffn_b2i(image_from_bags)

        mask_bags_pred = self.bags_classifier(bags_attn) # (B, N_B, 6)
        
        return {
            "feat": image_from_bags,
            "mask_words_pred": mask_words_pred,
            "mask_bags_pred": mask_bags_pred,
            #"bl_pred": self.bl_classifier(image_from_bags)
        }