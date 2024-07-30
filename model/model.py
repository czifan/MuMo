from model.bl_model import *
from model.yx_model import *
from model.nn_layers.transformer import *
import os 

class BLYXModel(nn.Module):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model
        self.yx_model = yx_model 

        self.bl_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.yx_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

        self.bl_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
        self.yx_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

        self.blyx_fc = nn.Linear(opts.blyx_out_features*3, opts.blyx_out_features)

        self.classifier = nn.Linear(opts.blyx_out_features, opts.n_classes)

        self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
        self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        self.sex_fc = nn.Linear(2, opts.blyx_out_features)
        self.age_fc = nn.Linear(2, opts.blyx_out_features)
        self.buwei_fc = nn.Linear(2, opts.blyx_out_features)
        self.xianshu_fc = nn.Linear(2, opts.blyx_out_features)
        self.time_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenxing_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenhua_fc = nn.Linear(4, opts.blyx_out_features)
        self.ln_dis_fc = nn.Linear(13, opts.blyx_out_features)
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=opts.yx_attn_heads, dropout=opts.blyx_dropout, batch_first=True)

        self.bl_fc = nn.Linear(opts.bl_out_features, opts.blyx_out_features)
        self.yx_fc = nn.Linear(opts.yx_out_features, opts.blyx_out_features)

        # self.bl_from_yx_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, 
        #                             num_heads=1, dropout=opts.blyx_dropout, batch_first=True)
        # self.yx_from_bl_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, 
        #                             num_heads=1, dropout=opts.blyx_dropout, batch_first=True)

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C)
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C)

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        yx_spe_feat = self.yx_spe_fc(yx_feat) # (B, C)
        yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1)
        yx_spe_feat = yx_spe_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag) # (B, C)
        yx_com_feat = self.yx_com_fc(yx_feat) # (B, C)
        # yx_com_feat_2, _ = self.yx_from_bl_attn(key=bl_feat.unsqueeze(dim=1), 
        #                                         query=yx_feat.unsqueeze(dim=1), 
        #                                         value=bl_feat.unsqueeze(dim=1))
        # yx_com_feat = yx_com_feat.squeeze(dim=1) # (B, C)
        
        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_spe_feat = self.bl_spe_fc(bl_feat) # (B, C)
        bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
        bl_spe_feat = bl_spe_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag) # (B, C)
        bl_com_feat = self.bl_com_fc(bl_feat) # (B, C)
        # bl_com_feat, _ = self.bl_from_yx_attn(key=yx_feat.unsqueeze(dim=1), 
        #                                         query=bl_feat.unsqueeze(dim=1), 
        #                                         value=yx_feat.unsqueeze(dim=1))
        # bl_com_feat = bl_com_feat.squeeze(dim=1) # (B, C)
        
        blyx_flag = bl_flag * yx_flag # (B, 1)
        com_feat = (yx_com_feat + bl_com_feat) / 2.0 * blyx_flag
        com_feat = com_feat * blyx_flag + yx_com_feat * yx_flag * (1.0 - blyx_flag)
        com_feat = com_feat * blyx_flag + bl_com_feat * bl_flag * (1.0 - blyx_flag)

        self.info_dict = {
            "id": batch["id"][0],
        }

        blyx_feat = torch.cat([bl_spe_feat, com_feat, yx_spe_feat], dim=1) # (B, 3C)
        #blyx_feat = bl_spe_feat * 0.25 + com_feat * 0.5 + yx_spe_feat * 0.25
        blyx_feat = self.blyx_fc(blyx_feat)

        # blyx_feat = (bl_spe_feat + com_feat + yx_spe_feat) / 3.0
        clinical_feat = torch.stack([
            self.sex_fc(batch["clinical_sex"]),
            self.age_fc(batch["clinical_age"]),
            self.buwei_fc(batch["clinical_buwei"]),
            self.xianshu_fc(batch["clinical_xianshu"]),
            self.time_fc(batch["clinical_time"]),
            self.fenhua_fc(batch["clinical_fenhua"]),
            self.fenxing_fc(batch["clinical_fenxing"]),
            self.ln_dis_fc(batch["clinical_ln_dis"]),
        ], dim=1) # (B, 7, C)
        clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
                                query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
        self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
        blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
        blyx_pred = self.classifier(blyx_feat)

        if self.opts.feat_dir not in [None, "None"]:
            npz_dir = self.opts.feat_dir
            os.makedirs(npz_dir, exist_ok=True)
            np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                    bl_feat=bl_spe_feat.detach().cpu().numpy(),
                    bl_syn_feat=bl_syn_feat.detach().cpu().numpy(),
                    bl_flag=bl_flag.detach().cpu().numpy(),
                    com_feat=com_feat.detach().cpu().numpy(),
                    yx_feat=yx_spe_feat.detach().cpu().numpy(),
                    yx_syn_feat=yx_syn_feat.detach().cpu().numpy(),
                    yx_flag=yx_flag.detach().cpu().numpy())

        if self.opts.attnmap_weight_dir not in [None, "None"]:
            npz_dir = os.path.join(self.opts.attnmap_weight_dir, "BLYX")
            os.makedirs(npz_dir, exist_ok=True)
            np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
                    clinical_weight=self.info_dict["clinical_weight"])

        # feat_dir = "../Analysis/Feat"
        # os.makedirs(feat_dir, exist_ok=True)
        # np.savez(os.path.join(feat_dir, self.info_dict["id"]+".npz"),
        #         bl_feat=bl_results["feat"][0].detach().cpu().numpy(),
        #         yx_feat=yx_results["feat"][0].detach().cpu().numpy(),
        #         blyx_feat=blyx_feat[0].detach().cpu().numpy(),
        #         yx_spe_feat=yx_spe_feat[0].detach().cpu().numpy(),
        #         bl_spe_feat=bl_spe_feat[0].detach().cpu().numpy(),
        #         yx_com_feat=yx_com_feat[0].detach().cpu().numpy(),
        #         bl_com_feat=bl_com_feat[0].detach().cpu().numpy(),
        #         yx_flag=yx_flag[0].item(),
        #         bl_flag=bl_flag[0].item())
        
        # yx_pred = yx_results["yx_pred"] * yx_flag + blyx_pred * (1.0 - yx_flag)
        # bl_pred = bl_results["bl_pred"] * bl_flag + blyx_pred * (1.0 - bl_flag)
        # blyx_pred = blyx_pred + yx_pred + bl_pred

        return {
            "pred": blyx_pred,
            "feat": blyx_feat,
            "bl_com_feat": bl_com_feat,
            "yx_com_feat": yx_com_feat,
            "blyx_flag": blyx_flag,

            "lesions_pred": yx_results["lesions_pred"],

            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }

class BLYXModelNonePationInfo(BLYXModel):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__(opts, bl_model, yx_model)

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C)
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C)

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        yx_spe_feat = self.yx_spe_fc(yx_feat) # (B, C)
        yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1)
        yx_spe_feat = yx_spe_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag) # (B, C)
        yx_com_feat = self.yx_com_fc(yx_feat) # (B, C)
        
        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_spe_feat = self.bl_spe_fc(bl_feat) # (B, C)
        bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
        bl_spe_feat = bl_spe_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag) # (B, C)
        bl_com_feat = self.bl_com_fc(bl_feat) # (B, C)
        
        blyx_flag = bl_flag * yx_flag # (B, 1)
        com_feat = (yx_com_feat + bl_com_feat) / 2.0 * blyx_flag
        com_feat = com_feat * blyx_flag + yx_com_feat * yx_flag * (1.0 - blyx_flag)
        com_feat = com_feat * blyx_flag + bl_com_feat * bl_flag * (1.0 - blyx_flag)

        blyx_feat = torch.cat([bl_spe_feat, com_feat, yx_spe_feat], dim=1) # (B, 3C)
        blyx_feat = self.blyx_fc(blyx_feat)
        blyx_pred = self.classifier(blyx_feat)

        return {
            "pred": blyx_pred,
            "feat": blyx_feat,
            "bl_com_feat": bl_com_feat,
            "yx_com_feat": yx_com_feat,
            "blyx_flag": blyx_flag,

            "lesions_pred": yx_results["lesions_pred"],

            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }


class BLYXModelConcat(nn.Module):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model
        self.yx_model = yx_model 

        self.blyx_fc = nn.Linear(opts.blyx_out_features*2, opts.blyx_out_features)

        self.classifier = nn.Linear(opts.blyx_out_features, opts.n_classes)

        self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
        self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        self.sex_fc = nn.Linear(2, opts.blyx_out_features)
        self.age_fc = nn.Linear(2, opts.blyx_out_features)
        self.buwei_fc = nn.Linear(2, opts.blyx_out_features)
        self.xianshu_fc = nn.Linear(2, opts.blyx_out_features)
        self.time_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenxing_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenhua_fc = nn.Linear(4, opts.blyx_out_features)
        self.ln_dis_fc = nn.Linear(13, opts.blyx_out_features)
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=1, dropout=opts.blyx_dropout, batch_first=True)

        self.bl_fc = nn.Linear(opts.bl_out_features, opts.blyx_out_features)
        self.yx_fc = nn.Linear(opts.yx_out_features, opts.blyx_out_features)

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C)
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C)

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1)
        yx_flag = yx_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag)

        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
        bl_feat = bl_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag)

        blyx_feat = torch.cat([yx_feat, bl_feat], dim=1)
        blyx_feat = self.blyx_fc(blyx_feat)
        blyx_pred = self.classifier(blyx_feat)

        return {
            "pred": blyx_pred,
            "feat": blyx_feat,

            "lesions_pred": yx_results["lesions_pred"],

            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }

class BLYXModelPlus(nn.Module):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model
        self.yx_model = yx_model 

        self.blyx_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

        self.classifier = nn.Linear(opts.blyx_out_features, opts.n_classes)

        self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
        self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        self.sex_fc = nn.Linear(2, opts.blyx_out_features)
        self.age_fc = nn.Linear(2, opts.blyx_out_features)
        self.buwei_fc = nn.Linear(2, opts.blyx_out_features)
        self.xianshu_fc = nn.Linear(2, opts.blyx_out_features)
        self.time_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenxing_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenhua_fc = nn.Linear(4, opts.blyx_out_features)
        self.ln_dis_fc = nn.Linear(13, opts.blyx_out_features)
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=1, dropout=opts.blyx_dropout, batch_first=True)

        self.bl_fc = nn.Linear(opts.bl_out_features, opts.blyx_out_features)
        self.yx_fc = nn.Linear(opts.yx_out_features, opts.blyx_out_features)

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C)
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C)

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1)
        yx_flag = yx_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag)

        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
        bl_feat = bl_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag)

        blyx_feat = yx_feat + bl_feat
        blyx_feat = self.blyx_fc(blyx_feat)
        blyx_pred = self.classifier(blyx_feat)

        return {
            "pred": blyx_pred,
            "feat": blyx_feat,

            "lesions_pred": yx_results["lesions_pred"],

            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }

class BLYXModelTimes(nn.Module):
    def __init__(self, opts, bl_model, yx_model):
        super().__init__()
        self.opts = opts
        self.bl_model = bl_model
        self.yx_model = yx_model 

        self.blyx_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

        self.classifier = nn.Linear(opts.blyx_out_features, opts.n_classes)

        self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
        self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

        self.sex_fc = nn.Linear(2, opts.blyx_out_features)
        self.age_fc = nn.Linear(2, opts.blyx_out_features)
        self.buwei_fc = nn.Linear(2, opts.blyx_out_features)
        self.xianshu_fc = nn.Linear(2, opts.blyx_out_features)
        self.time_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenxing_fc = nn.Linear(3, opts.blyx_out_features)
        self.fenhua_fc = nn.Linear(4, opts.blyx_out_features)
        self.ln_dis_fc = nn.Linear(13, opts.blyx_out_features)
        self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=1, dropout=opts.blyx_dropout, batch_first=True)

        self.bl_fc = nn.Linear(opts.bl_out_features, opts.blyx_out_features)
        self.yx_fc = nn.Linear(opts.yx_out_features, opts.blyx_out_features)

    def forward(self, batch, *args, **kwargs):
        yx_results = self.yx_model(batch, *args, **kwargs)
        yx_feat = self.yx_fc(yx_results["feat"]) # (B, C)
        bl_results = self.bl_model(batch, *args, **kwargs)
        bl_feat = self.bl_fc(bl_results["feat"]) # (B, C)

        yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
        yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1)
        yx_flag = yx_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag)

        bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
        bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
        bl_feat = bl_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag)

        blyx_feat = yx_feat * bl_feat
        blyx_feat = self.blyx_fc(blyx_feat)
        blyx_pred = self.classifier(blyx_feat)

        return {
            "pred": blyx_pred,
            "feat": blyx_feat,

            "lesions_pred": yx_results["lesions_pred"],

            "mask_words_pred": bl_results["mask_words_pred"],
            "mask_bags_pred": bl_results["mask_bags_pred"],
        }


# class BLYXModelWeight(nn.Module):
#     def __init__(self, opts, bl_model, yx_model):
#         super().__init__()
#         self.opts = opts
#         self.bl_model = bl_model
#         self.yx_model = yx_model 

#         self.bl_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
#         self.yx_spe_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

#         self.bl_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)
#         self.yx_com_fc = nn.Linear(opts.blyx_out_features, opts.blyx_out_features)

#         self.blyx_fc = nn.Linear(opts.blyx_out_features*3, opts.blyx_out_features)

#         self.classifier = nn.Linear(opts.blyx_out_features, opts.n_classes)

#         self.yx_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))
#         self.bl_param = nn.Parameter(torch.zeros(1, opts.blyx_out_features))

#         self.sex_fc = nn.Linear(2, opts.blyx_out_features)
#         self.age_fc = nn.Linear(2, opts.blyx_out_features)
#         self.buwei_fc = nn.Linear(2, opts.blyx_out_features)
#         self.xianshu_fc = nn.Linear(2, opts.blyx_out_features)
#         self.time_fc = nn.Linear(3, opts.blyx_out_features)
#         self.fenxing_fc = nn.Linear(3, opts.blyx_out_features)
#         self.fenhua_fc = nn.Linear(4, opts.blyx_out_features)
#         self.ln_dis_fc = nn.Linear(13, opts.blyx_out_features)
#         self.clinical_image_attn = nn.MultiheadAttention(embed_dim=opts.blyx_out_features, num_heads=1, dropout=opts.blyx_dropout, batch_first=True)

#         self.bl_fc = nn.Linear(opts.bl_out_features, opts.blyx_out_features)
#         self.yx_fc = nn.Linear(opts.yx_out_features, opts.blyx_out_features)

#     def forward(self, batch, *args, **kwargs):
#         yx_results = self.yx_model(batch, *args, **kwargs)
#         yx_feat = self.yx_fc(yx_results["feat"]) # (B, C)
#         bl_results = self.bl_model(batch, *args, **kwargs)
#         bl_feat = self.bl_fc(bl_results["feat"]) # (B, C)

#         yx_flag = batch["yx_flag"].unsqueeze(dim=1).float() # (B, 1)
#         yx_spe_feat = self.yx_spe_fc(yx_feat) # (B, C)
#         yx_syn_feat = self.yx_param.repeat(yx_feat.shape[0], 1)
#         yx_spe_feat = yx_spe_feat * yx_flag + yx_syn_feat * (1.0 - yx_flag) # (B, C)
#         yx_com_feat = self.yx_com_fc(yx_feat) # (B, C)
        
#         bl_flag = batch["bl_flag"].unsqueeze(dim=1).float() # (B, 1)
#         bl_spe_feat = self.bl_spe_fc(bl_feat) # (B, C)
#         bl_syn_feat = self.bl_param.repeat(bl_feat.shape[0], 1)
#         bl_spe_feat = bl_spe_feat * bl_flag + bl_syn_feat * (1.0 - bl_flag) # (B, C)
#         bl_com_feat = self.bl_com_fc(bl_feat) # (B, C)
        
#         blyx_flag = bl_flag * yx_flag # (B, 1)
#         com_feat = (yx_com_feat + bl_com_feat) / 2.0 * blyx_flag
#         com_feat = com_feat * blyx_flag + yx_com_feat * yx_flag * (1.0 - blyx_flag)
#         com_feat = com_feat * blyx_flag + bl_com_feat * bl_flag * (1.0 - blyx_flag)

#         self.info_dict = {
#             "id": batch["id"][0],
#         }
#         blyx_feat = torch.cat([bl_spe_feat, com_feat, yx_spe_feat], dim=1) # (B, 3C)
#         blyx_feat = self.blyx_fc(blyx_feat)
#         clinical_feat = torch.stack([
#             self.sex_fc(batch["clinical_sex"]),
#             self.age_fc(batch["clinical_age"]),
#             self.buwei_fc(batch["clinical_buwei"]),
#             self.xianshu_fc(batch["clinical_xianshu"]),
#             self.time_fc(batch["clinical_time"]),
#             self.fenhua_fc(batch["clinical_fenhua"]),
#             self.fenxing_fc(batch["clinical_fenxing"]),
#             self.ln_dis_fc(batch["clinical_ln_dis"]),
#         ], dim=1) # (B, 7, C)
#         clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
#                                 query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
#         self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
#         blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
#         blyx_pred = self.classifier(blyx_feat)

#         feat_dir = os.path.join("../Analysis.0201/BLYXweight")
#         os.makedirs(feat_dir, exist_ok=True)
#         np.savez(os.path.join(feat_dir, self.info_dict["id"]+".npz"),
#                 bl_spe_feat=bl_spe_feat.detach().cpu().numpy(),
#                 com_feat=com_feat.detach().cpu().numpy(),
#                 yx_spe_feat=yx_spe_feat.detach().cpu().numpy())



#         yx_lst = [blyx_pred.detach().cpu().numpy(),]
#         for _ in range(30):
#             yx_spe_feat_tmp = yx_spe_feat+(0.1**0.5)*torch.randn(yx_spe_feat.shape).float().to(yx_feat.device)
#             blyx_feat = torch.cat([bl_spe_feat, com_feat, yx_spe_feat_tmp], dim=1) # (B, 3C)
#             blyx_feat = self.blyx_fc(blyx_feat)
#             clinical_feat = torch.stack([
#                 self.sex_fc(batch["clinical_sex"]),
#                 self.age_fc(batch["clinical_age"]),
#                 self.buwei_fc(batch["clinical_buwei"]),
#                 self.xianshu_fc(batch["clinical_xianshu"]),
#                 self.time_fc(batch["clinical_time"]),
#                 self.fenhua_fc(batch["clinical_fenhua"]),
#                 self.fenxing_fc(batch["clinical_fenxing"]),
#                 self.ln_dis_fc(batch["clinical_ln_dis"]),
#             ], dim=1) # (B, 7, C)
#             clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
#                                     query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
#             self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
#             blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
#             blyx_pred_tmp = self.classifier(blyx_feat)
#             yx_lst.append(blyx_pred_tmp.detach().cpu().numpy())
#         yx_lst = np.concatenate(yx_lst, axis=0)
#         print(yx_lst.shape)

#         bl_lst = [blyx_pred.detach().cpu().numpy(),]
#         for _ in range(30):
#             bl_spe_feat_tmp =  bl_spe_feat+(0.1**0.5)*torch.randn(bl_spe_feat.shape).float().to(bl_feat.device)
#             blyx_feat = torch.cat([bl_spe_feat_tmp, com_feat, yx_spe_feat_tmp], dim=1) # (B, 3C)
#             blyx_feat = self.blyx_fc(blyx_feat)
#             clinical_feat = torch.stack([
#                 self.sex_fc(batch["clinical_sex"]),
#                 self.age_fc(batch["clinical_age"]),
#                 self.buwei_fc(batch["clinical_buwei"]),
#                 self.xianshu_fc(batch["clinical_xianshu"]),
#                 self.time_fc(batch["clinical_time"]),
#                 self.fenhua_fc(batch["clinical_fenhua"]),
#                 self.fenxing_fc(batch["clinical_fenxing"]),
#                 self.ln_dis_fc(batch["clinical_ln_dis"]),
#             ], dim=1) # (B, 7, C)
#             clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
#                                     query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
#             self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
#             blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
#             blyx_pred_tmp = self.classifier(blyx_feat)
#             bl_lst.append(blyx_pred_tmp.detach().cpu().numpy())
#         bl_lst = np.concatenate(bl_lst, axis=0)

#         # yx_lst = [blyx_pred.detach().cpu().numpy(),]
#         # for npz_file in os.listdir(feat_dir):
#         #     if self.info_dict["id"] in npz_file:
#         #         continue
#         #     npz_data = np.load(os.path.join(feat_dir, npz_file))
#         #     yx_spe_feat_tmp = torch.Tensor(npz_data["yx_spe_feat"]).float().to(yx_feat.device)
#         #     blyx_feat = torch.cat([bl_spe_feat, com_feat, yx_spe_feat_tmp], dim=1) # (B, 3C)
#         #     blyx_feat = self.blyx_fc(blyx_feat)
#         #     clinical_feat = torch.stack([
#         #         self.sex_fc(batch["clinical_sex"]),
#         #         self.age_fc(batch["clinical_age"]),
#         #         self.buwei_fc(batch["clinical_buwei"]),
#         #         self.xianshu_fc(batch["clinical_xianshu"]),
#         #         self.time_fc(batch["clinical_time"]),
#         #         self.fenhua_fc(batch["clinical_fenhua"]),
#         #         self.fenxing_fc(batch["clinical_fenxing"]),
#         #         self.ln_dis_fc(batch["clinical_ln_dis"]),
#         #     ], dim=1) # (B, 7, C)
#         #     clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
#         #                             query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
#         #     self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
#         #     blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
#         #     blyx_pred_tmp = self.classifier(blyx_feat)
#         #     yx_lst.append(blyx_pred_tmp.detach().cpu().numpy())
#         # yx_lst = np.concatenate(yx_lst, axis=0)
#         # print(yx_lst.shape)

#         # bl_lst = [blyx_pred.detach().cpu().numpy(),]
#         # for npz_file in os.listdir(feat_dir):
#         #     if self.info_dict["id"] in npz_file:
#         #         continue
#         #     npz_data = np.load(os.path.join(feat_dir, npz_file))
#         #     bl_spe_feat_tmp = torch.Tensor(npz_data["bl_spe_feat"]).float().to(bl_feat.device)
#         #     blyx_feat = torch.cat([bl_spe_feat_tmp, com_feat, yx_spe_feat_tmp], dim=1) # (B, 3C)
#         #     blyx_feat = self.blyx_fc(blyx_feat)
#         #     clinical_feat = torch.stack([
#         #         self.sex_fc(batch["clinical_sex"]),
#         #         self.age_fc(batch["clinical_age"]),
#         #         self.buwei_fc(batch["clinical_buwei"]),
#         #         self.xianshu_fc(batch["clinical_xianshu"]),
#         #         self.time_fc(batch["clinical_time"]),
#         #         self.fenhua_fc(batch["clinical_fenhua"]),
#         #         self.fenxing_fc(batch["clinical_fenxing"]),
#         #         self.ln_dis_fc(batch["clinical_ln_dis"]),
#         #     ], dim=1) # (B, 7, C)
#         #     clinical_image_feat, clinical_attnmap = self.clinical_image_attn(key=clinical_feat, 
#         #                             query=blyx_feat.unsqueeze(dim=1), value=clinical_feat)
#         #     self.info_dict["clinical_weight"] = clinical_attnmap[0, 0].detach().cpu().numpy()
#         #     blyx_feat = blyx_feat + clinical_image_feat.squeeze(dim=1)
#         #     blyx_pred_tmp = self.classifier(blyx_feat)
#         #     bl_lst.append(blyx_pred_tmp.detach().cpu().numpy())
#         # bl_lst = np.concatenate(bl_lst, axis=0)
#         np.savez(os.path.join(feat_dir, self.info_dict["id"]+".npz"),
#                 bl_spe_feat=bl_spe_feat.detach().cpu().numpy(),
#                 com_feat=com_feat.detach().cpu().numpy(),
#                 yx_spe_feat=yx_spe_feat.detach().cpu().numpy(),
#                 yx_lst=yx_lst,
#                 bl_lst=bl_lst)

#         if self.opts.attnmap_weight_dir not in [None, "None"]:
#             npz_dir = os.path.join(self.opts.attnmap_weight_dir, "BLYX")
#             os.makedirs(npz_dir, exist_ok=True)
#             np.savez(os.path.join(npz_dir, self.info_dict["id"]+".npz"),
#                     clinical_weight=self.info_dict["clinical_weight"])

#         # feat_dir = "../Analysis/Feat"
#         # os.makedirs(feat_dir, exist_ok=True)
#         # np.savez(os.path.join(feat_dir, self.info_dict["id"]+".npz"),
#         #         bl_feat=bl_results["feat"][0].detach().cpu().numpy(),
#         #         yx_feat=yx_results["feat"][0].detach().cpu().numpy(),
#         #         blyx_feat=blyx_feat[0].detach().cpu().numpy(),
#         #         yx_spe_feat=yx_spe_feat[0].detach().cpu().numpy(),
#         #         bl_spe_feat=bl_spe_feat[0].detach().cpu().numpy(),
#         #         yx_com_feat=yx_com_feat[0].detach().cpu().numpy(),
#         #         bl_com_feat=bl_com_feat[0].detach().cpu().numpy(),
#         #         yx_flag=yx_flag[0].item(),
#         #         bl_flag=bl_flag[0].item())
        
#         # yx_pred = yx_results["yx_pred"] * yx_flag + blyx_pred * (1.0 - yx_flag)
#         # bl_pred = bl_results["bl_pred"] * bl_flag + blyx_pred * (1.0 - bl_flag)
#         # blyx_pred = blyx_pred + yx_pred + bl_pred

#         return {
#             "pred": blyx_pred,
#             "feat": blyx_feat,
#             "bl_com_feat": bl_com_feat,
#             "yx_com_feat": yx_com_feat,
#             "blyx_flag": blyx_flag,

#             "lesions_pred": yx_results["lesions_pred"],

#             "mask_words_pred": bl_results["mask_words_pred"],
#             "mask_bags_pred": bl_results["mask_bags_pred"],
#         }
