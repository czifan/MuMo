from torch.utils.data import Dataset
import torch 
from PIL import Image
import numpy as np
from data_loader.utils import *
from utils.print_utils import *
import pandas as pd
from copy import deepcopy
import pickle

BLACK_LST = []
YX_BLACK_LST = []
BL_BLACK_LST = []
yx_black_lst_v1 = []
bl_black_lst_v1 = []

class BLYXDataset(Dataset):
    def __init__(self, opts, split, split_file, printer, ignore=True, cohort=None):
        super().__init__()

        self.split = split
        self.opts = opts
        self.is_training = ("train" == split)

        self.bl_img_dir = opts.bl_img_dir
        self.bl_num_bags = opts.bl_num_bags
        self.bl_bag_size = opts.bl_bag_size
        self.bl_img_extn = opts.bl_img_extn
        self.bl_word_size = opts.bl_word_size

        self.yx_img_dir = opts.yx_img_dir
        self.yx_img_extn = opts.yx_img_extn
        self.yx_num_lesions = opts.yx_num_lesions
        self.yx_lesion_size = opts.yx_lesion_size

        pd_data = pd.read_excel(split_file)
        self.bl_num, self.yx_num, self.blyx_num = 0, 0, 0
        self.id_lst, self.name_lst, self.bl_pid_lst, self.yx_pid_lst, self.liaoxiao_lst = [], [], [], [], []
        self.OS_lst, self.OSCensor_lst, self.PFS_lst, self.PFSCensor_lst = [], [], [], []
        self.fangan_lst, self.label_lst, self.time_lst = [], [], []
        self.sex_lst, self.age_lst, self.buwei_lst, self.xianshu_lst = [], [], [], []
        self.ln_dis_lst = []
        self.zl_ln_lst, self.zl_ln_pos_lst, self.zl_multi_lst, self.zl_per_lst = [], [], [], []
        self.fenhua_lst, self.fenxing_lst, self.tils_lst, self.her2_lst, self.tumor_lst = [], [], [], [], []
        self.yx_flag_lst, self.bl_flag_lst = [], []
        for (id, name, bl_pid, yx_pid, start_time, liaoxiao, PFS, PFSCensor, OS, OSCensor, fangan,
            sex, age, buwei, xianshu,
            zl_ln, zl_ln_pos, zl_multi, zl_per,
            fenhua, fenxing, tils, her2, tumor, ln_dis,
            yx_flag, bl_flag
            ) in zip(
            pd_data["住院号"], pd_data["姓名"], pd_data["病理勾画编号"], pd_data["影像勾画编号"], pd_data["开始抗HER2治疗日期"], pd_data["最佳疗效"],
            pd_data["PFS"], pd_data["PFSCensor"], pd_data["OS"], pd_data["OSCensor"], pd_data["联合免疫（是=1，否=0）"],
            pd_data["性别"], pd_data["年龄"], pd_data["肿瘤部位"], pd_data["治疗线数"],
            pd_data["转移淋巴结（0 没有转移淋巴结  1存在转移淋巴结，但不融合  2存在融合的转移淋巴结）"],
            pd_data["转移淋巴结位置（0没有转移淋巴结  1 存在局域转移淋巴结 2 存在M分期的腹腔或腹膜后淋巴结转移 3存在纵隔或锁骨上淋巴结转移 4存在其他少见远隔部位淋巴结转移（如腋窝、颈旁、腹股沟等等区域）"],
            pd_data["肝或肺多发转移，多发为≥3个病灶（0没有多发肝转移和多发肺转移 1仅存在多发肝转移 2仅存在多发肺转移 3两者均有）"],
            pd_data["腹膜转移（0 无腹膜转移 1存在腹膜转移）"],
            pd_data["分化程度"], pd_data["LAUREN分型"], pd_data["肿瘤相关淋巴细胞TILs"], pd_data["HER2表达异质性"], pd_data["肿瘤占比"], pd_data["转移部位整理"],
            pd_data["影像采样时间"], pd_data["病理采样时间"]):

            if id in BLACK_LST:
                continue

            if isinstance(yx_pid, str):
                yx_pid = int(yx_pid.replace(",", ""))
            label = convert_label(self.opts.label_type, OS, OSCensor, liaoxiao, PFS=PFS, PFSCensor=PFSCensor)
            if label < -1 or (yx_pid in YX_BLACK_LST) or (bl_pid in BL_BLACK_LST):
                continue
            if str(bl_pid) != str(np.nan) and bl_flag == -1:
                continue
            if str(yx_pid) != str(np.nan) and yx_flag == -1:
                continue

            if cohort is not None:
                assert cohort in ["her2", "ci", "all"], (cohort)
                if cohort == "ci" and fangan == 0:
                    continue
                elif cohort == "her2" and fangan == 1:
                    continue
                elif cohort == "all":
                    pass

            if opts.model == "bingli" and (str(bl_pid) == str(np.nan) or bl_flag == -1):
                continue
            elif opts.model == "yingxiang" and (str(yx_pid) == str(np.nan) or yx_flag == -1):
                continue

            self.bl_num += int(str(bl_pid) != str(np.nan))
            self.yx_num += int(str(yx_pid) != str(np.nan))
            self.blyx_num += int(str(bl_pid) != str(np.nan) and (str(yx_pid) != str(np.nan)))

            self.id_lst.append(id)
            self.name_lst.append(name)
            self.bl_pid_lst.append(str(bl_pid).replace(".0", "") if bl_pid != str(np.nan) else [str(np.nan),])
            self.yx_pid_lst.append(str(int(yx_pid)) if not np.isnan(yx_pid) else str(np.nan))
            self.liaoxiao_lst.append("NA" if str(liaoxiao)==str(np.nan) else liaoxiao)

            self.OS_lst.append(OS)
            self.OSCensor_lst.append(OSCensor)
            self.PFS_lst.append(PFS)
            self.PFSCensor_lst.append(PFSCensor)

            self.fangan_lst.append(fangan)
            self.label_lst.append(label)

            self.sex_lst.append(sex)
            self.age_lst.append(age)
            self.buwei_lst.append(buwei)
            self.xianshu_lst.append(xianshu)
            self.time_lst.append(str(start_time).split("/")[0])

            self.zl_ln_lst.append(zl_ln)
            self.zl_ln_pos_lst.append(zl_ln_pos)
            self.zl_multi_lst.append(zl_multi)
            self.zl_per_lst.append(zl_per)

            self.fenhua_lst.append(fenhua)
            self.fenxing_lst.append(fenxing)
            self.tils_lst.append(tils)
            self.her2_lst.append(her2)
            self.tumor_lst.append(tumor)
            self.ln_dis_lst.append(ln_dis)

            self.bl_flag_lst.append(bl_flag)
            self.yx_flag_lst.append(yx_flag)
        
        self.diag_labels = deepcopy(self.label_lst)
        self.n_classes = len(np.unique(self.diag_labels))
        self.printer = printer

        print_info_message('Samples in {}: {}\t(bl={}\tyx={}\tblyx={} ({:.2f}%))'.format(
            split_file, self.__len__(), self.bl_num, self.yx_num, self.blyx_num, 100.0*self.blyx_num/self.__len__()), self.printer)
        print_info_message('-- {} ({:.2f}%) Non-response | {} ({:.2f}%) Response | {} ({:.2f}%) Others'.format(
            sum(np.asarray(self.label_lst)==0), 100.0*sum(np.asarray(self.label_lst)==0)/self.__len__(),
            sum(np.asarray(self.label_lst)==1), 100.0*sum(np.asarray(self.label_lst)==1)/self.__len__(),
            sum(np.asarray(self.label_lst)==-1), 100.0*sum(np.asarray(self.label_lst)==-1)/self.__len__(),
        ), self.printer)

    def __len__(self):
        return len(self.bl_pid_lst)

    def _generate_mask_bags_label(self, mask_bags):
        # mask_bags: (N_B, B_H, B_W)
        mask_bags_label = []
        mask_bags = mask_bags.reshape((mask_bags.shape[0], -1))
        for nb in range(mask_bags.shape[0]):
            mask_bags_label.append(np.argmax(np.bincount(mask_bags[nb])))
        mask_bags_label = torch.LongTensor(mask_bags_label) # (N_B,)
        return mask_bags_label

    def _generate_mask_words_label(self, mask_words):
        # mask_words: (N_B, N_W, W_H, W_W)
        mask_words_label = []
        mask_words = mask_words.reshape((mask_words.shape[0], mask_words.shape[1], -1))
        for nb in range(mask_words.shape[0]):
            mask_words_label_tmp = []
            for nw in range(mask_words.shape[1]):
                mask_words_label_tmp.append(np.argmax(np.bincount(mask_words[nb, nw])))
            mask_words_label.append(mask_words_label_tmp)
        mask_words_label = torch.LongTensor(mask_words_label) # (N_B, N_W)
        return mask_words_label

    def _load_bl_data(self, index):
        num_words_per_bag = (self.opts.bl_bag_size // self.opts.bl_word_size) ** 2
        if self.bl_pid_lst[index] != str(np.nan):
            bl_pid = sorted(self.bl_pid_lst[index].strip().split("+"))[-1] # 如果有两个切片，取最后一个
            # 大切块 bags: (N_B, B_H, B_W, C) | masks: (N_B, B_H, B_W)
            bags, masks, keys = load_all_bags_with_masks(os.path.join(self.bl_img_dir, bl_pid), self.bl_bag_size, 
                                                self.bl_bag_size, self.bl_img_extn, self.bl_num_bags, split=self.split)
            mask_bags_label = self._generate_mask_bags_label(masks)
            # 小切块 words: (N_B, N_W, C, W_H, W_W)
            feat_words = bags_to_words(bags, self.bl_word_size, self.bl_word_size, bl_pid).float()
            # 小切块对应的mask: (N_B, N_W, W_H, W_W)
            mask_words = masks_to_words(masks, self.bl_word_size, self.bl_word_size, bl_pid).long()
            mask_words_label = self._generate_mask_words_label(mask_words)

            radiomics_file = os.path.join(self.opts.bl_rad_dir, f"radiomics_{bl_pid}_norm.csv")
            radiomics_data = pd.read_csv(radiomics_file).values
            radiomics_dict = {}
            for line in radiomics_data:
                radiomics_dict[str(line[0])] = torch.FloatTensor(np.nan_to_num(np.asarray(line[1:], dtype=np.float32), 0.0)).float() # (736,)
            radiomics_feat = torch.stack([radiomics_dict[key] for key in keys], dim=0) # (N_B, 736)
            flag = 1
        else:
            feat_words = torch.zeros(max(1, self.opts.bl_num_bags), num_words_per_bag, 3, self.opts.bl_word_size, self.opts.bl_word_size).float()
            radiomics_feat = torch.zeros(max(1, self.opts.bl_num_bags), 736).float()
            mask_bags_label = torch.full((max(1, self.opts.bl_num_bags),), -1).long()
            mask_words_label = torch.full((max(1, self.opts.bl_num_bags), num_words_per_bag), -1).long()
            flag = 0
        return feat_words, radiomics_feat, mask_bags_label, mask_words_label, flag

    def _load_yx_data(self, index):
        if self.yx_pid_lst[index] != str(np.nan):
            yx_pid = self.yx_pid_lst[index]
            lesions, keys, lesions_label = load_lesions(self.yx_img_dir, yx_pid,
                            self.yx_lesion_size, self.yx_lesion_size, 
                            self.yx_img_extn, is_training=self.is_training, num_lesions=self.opts.yx_num_lesions, split=self.split)

            radiomics_file = os.path.join(self.opts.yx_rad_dir, f"radiomics_{yx_pid}_norm.csv")
            radiomics_data = pd.read_csv(radiomics_file).values
            radiomics_dict = {}
            for line in radiomics_data:
                radiomics_dict[str(line[0])] = torch.FloatTensor(np.nan_to_num(np.asarray(line[1:], dtype=np.float32), 0.0)).float() # (736,)
            radiomics_feat = torch.stack([radiomics_dict[key] for key in keys], dim=0) # (N_B, 736)
            flag = 1
        else:
            lesions = torch.zeros(max(1, self.opts.yx_num_lesions), 3, self.opts.yx_lesion_size, self.opts.yx_lesion_size).float()
            radiomics_feat = torch.zeros(max(1, self.opts.yx_num_lesions), 736).float()
            lesions_label = torch.full((max(1, self.opts.yx_num_lesions),), -1).long()
            flag = 0
        return lesions, radiomics_feat, lesions_label, flag

    def __getitem__(self, index):
        feat_words, bl_radiomics_feat, mask_bags_label, mask_words_label, bl_flag = self._load_bl_data(index)
        lesions, yx_radiomics_feat, lesions_label, yx_flag = self._load_yx_data(index)

        #print(self.yx_pid_lst[index], self.bl_pid_lst[index])
        assert bl_flag or yx_flag, (self.id_lst[index], self.yx_pid_lst[index], self.bl_pid_lst[index])

        return {
            "id": self.id_lst[index],
            "name": self.name_lst[index],
            
            "feat_words": feat_words,
            "mask_bags_label": mask_bags_label,
            "mask_words_label": mask_words_label,
            "bl_radiomics_feat": bl_radiomics_feat,
            "bl_pid": self.bl_pid_lst[index],
            "bl_flag": bl_flag,

            "lesions": lesions,
            "lesions_label": lesions_label,
            "yx_radiomics_feat": yx_radiomics_feat,
            "yx_pid": self.yx_pid_lst[index],
            "yx_flag": yx_flag,

            "liaoxiao": self.liaoxiao_lst[index],
            "os": self.OS_lst[index],
            "os_censor": self.OSCensor_lst[index],
            "pfs": self.PFS_lst[index],
            "pfs_censor": self.PFSCensor_lst[index],

            "fangan": self.fangan_lst[index],
            "label": self.label_lst[index],


            "clinical_sex": ccd_sex(self.sex_lst[index]), # (2,)
            "clinical_age": ccd_age(self.age_lst[index]), # (2,)
            "clinical_buwei": ccd_buwei(self.buwei_lst[index]), # (2,)
            "clinical_xianshu": ccd_xianshu(self.xianshu_lst[index]), # (2,)
            "clinical_time": ccd_time(self.time_lst[index]), # (3,)
            "clinical_fenxing": ccd_fenxing(self.fenxing_lst[index]), # (3,)
            "clinical_fenhua": ccd_fenhua(self.fenhua_lst[index]), # (4,)
            "clinical_ln_dis": ccd_ln_dis(self.ln_dis_lst[index]), # (13,) 
            "clinical_yx_flag": ccd_yx_flag(self.yx_flag_lst[index]), # (3,)
            "clinical_bl_flag": ccd_bl_flag(self.bl_flag_lst[index]), # (3,)


            "clinical_bl_tils": ccd_bl_tils(self.tils_lst[index]), # (10,)
            "clinical_bl_her2": ccd_bl_her2(self.her2_lst[index]), # (4,)
            "clinical_bl_tumor": ccd_bl_tumor(self.tumor_lst[index]), # (10,)

            "clinical_yx_stomach": ccd_yx_stomach(os.path.join(self.yx_img_dir, self.yx_pid_lst[index])), # (2,)
            #"clinical_yx_ln_dis": ccd_yx_ln_dis(os.path.join(self.yx_img_dir, self.yx_pid_lst[index])), # (9,)
            "clinical_yx_ln_num": ccd_yx_ln_num(os.path.join(self.yx_img_dir, self.yx_pid_lst[index])), # (7,)
            "clinical_yx_zl_ln": ccd_yx_zl_ln(self.zl_ln_lst[index]), # (3,)
            "clinical_yx_zl_ln_pos": ccd_yx_zl_ln_pos(self.zl_ln_pos_lst[index]), # (5,)
            "clinical_yx_zl_multi": ccd_yx_zl_multi(self.zl_multi_lst[index]), # (4,)
            "clinical_yx_zl_per": ccd_yx_zl_per(self.zl_per_lst[index]), # (2,)
        }