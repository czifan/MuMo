import numpy as np
import random
import cv2
import torch 
import gc
from glob import glob
import os 
from PIL import Image, ImageFile, ImageEnhance

ImageFile.LOAD_TRUNCATED_IMAGES = True
Image.MAX_IMAGE_PIXELS = 1000000000000

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]
yx_MEAN = [0.485, 0.456, 0.406]
yx_STD = [0.229, 0.224, 0.225]
WHITE_BALANCE_LST = ["NF-10", "NF-26", "NF-20", "NF-19", "NF-17", "NF-1", "NF-27", "NF-25", "NF-6", "NF-8", "NF-5", "NF-9", "NF-29", "NF-4", "NF-28"]

def map_s(s):
    if '一线' in s:
        return torch.Tensor([0.0, 1.0]).float()
    else:
        return torch.Tensor([1.0, 0.0]).float()

def map_c(c):
    if c <= 0:
        return torch.Tensor([1.0, 0.0]).float()
    else:
        return torch.Tensor([0.0, 1.0]).float()

def normalize_words_np(words_np):
    # (N_B, N_W*N_H, W_H, W_W, C)
    words_np = words_np.astype(float)
    words_np /= 255.0
    words_np -= MEAN
    words_np /= STD
    # (N_B, N_W*N_H, W_H, W_W, C) -> (N_B, N_W*N_H, C, W_H, W_W)
    words_np = words_np.transpose(0, 1, 4, 2, 3)
    return words_np

lesion_to_label = {
    'LN': 0,
    'stomach': 1,
    'Liver': 2,
    'Peritoneum': 3,
    'Other': 4,
    'Spleen': 4,
    'Bone': 4,
    'Soft': 4,
    'Aden': 4,
}

def load_lesions(data_dir, pid, lesion_height, lesion_width, img_extn, is_training, num_lesions=4, split="train"):
    files = glob(os.path.join(data_dir, "0", pid, f"*.{img_extn}"))
    files = [file for file in files if not file.endswith(f"_mask.{img_extn}")]
    lesions = []
    keys = []
    lesions_label = []
    if len(files) == 0:
        print(data_dir, pid)

    if num_lesions > 0:
        files = np.random.choice(files, num_lesions, replace=True)

    for file in files:
        key = os.path.basename(file).split('.')[0]
        lesion = np.stack([
            np.asarray(Image.open(file.replace("/0/", "/-1/")).convert("L")),
            np.asarray(Image.open(file).convert("L")),
            np.asarray(Image.open(file.replace("/0/", "/1/")).convert("L")),
        ], axis=-1)
        if lesion_height != lesion.shape[0] or lesion_width != lesion.shape[1]:
            lesion = cv2.resize(lesion, (lesion_width, lesion_height))
        if is_training:
            lesion = random_transform_np(lesion, max_rotation=30, pad_value=0)
        lesions.append(lesion)
        keys.append(os.path.basename(file).split(".")[0])
        lesions_label.append(lesion_to_label[os.path.basename(file).split('_')[-1].split('.')[0].replace('1', '').replace('2', '')])
    if len(lesions) == 0:
        print(data_dir, pid, "+++++++") # debug
        return torch.Tensor(1)
    lesions = np.stack(lesions, axis=0)
    lesions = lesions.astype(float)
    lesions /= 255.0
    lesions -= yx_MEAN
    lesions /= yx_STD
    lesions = torch.Tensor(lesions.transpose(0, 3, 1, 2)).float() # (N_l, 3, 224, 224)
    lesions_label = torch.LongTensor(lesions_label)
    return lesions, keys, lesions_label

def random_transform_np(img_np, max_rotation=10, pad_value=255):
    h, w = img_np.shape[:2]
    # flip the bag
    if random.random() < 0.5:
        flip_code = random.choice([0, 1])  # 0 for horizontal and 1 for vertical
        img_np = cv2.flip(img_np, flip_code)

    # rotate the image
    if random.random() < 0.5:
        angle = random.choice(np.arange(-max_rotation, max_rotation + 1).tolist())
        # note that these functions take argument as (w, h)
        rot_mat = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1)
        img_np = cv2.warpAffine(img_np, rot_mat, (w, h),
                                borderValue=(pad_value, pad_value, pad_value))  # 255 because that correpond to background in WSIs

    # random crop and scale
    if random.random() < 0.5:
        x = random.randint(0, w - w // 4)
        y = random.randint(0, h - h // 4)
        img_np = img_np[y:, x:]
        img_np = cv2.resize(img_np, (w, h))

    return img_np

def load_all_bags_with_masks(data_dir, bag_height, bag_width, img_extn, num_bags, split="train"):
    files = glob(os.path.join(data_dir, f"*-img.{img_extn}"))
    bags = []
    masks = []
    keys = []
    if num_bags > 0:
        files = np.random.choice(files, num_bags)

    for file in files:
        pid = file.split("/")[-2]
        if pid in WHITE_BALANCE_LST:
            #print(pid)
            file = file.replace("GuoZhongBingLiData", "GuoZhongBingLiDataWB")
        # (B_H, B_W, C)
        bag = np.asarray(Image.open(file).convert("RGB"))
        if bag_height != bag.shape[0] or bag_width != bag.shape[1]:
            bag = cv2.resize(bag, (bag_width, bag_height))
        bags.append(bag)

        mask_file = file.replace("GuoZhongBingLiDataWB", "GuoZhongBingLiData").replace('-img', '-mask')
        # mask_file = file.replace('-img', '-mask')
        mask = np.asarray(Image.open(mask_file).convert('L'))

        mask[mask==1] = 1
        mask[mask==2] = 1
        mask[mask==3] = 1
        mask[mask==4] = 2
        mask[mask==5] = 3

        if bag_height != mask.shape[0] or bag_width != mask.shape[1]:
            mask = cv2.resize(mask, (bag_width, bag_height))
        masks.append(mask)
        
        keys.append(os.path.basename(file).replace("-img.png", "").replace("-mask.png", ""))
    if len(bags) == 0:
        return torch.Tensor(1)
    bags = np.stack(bags, axis=0) # (N_B, B_H, B_W, C)
    masks = np.stack(masks, axis=0) # (N_B, B_H, B_W)
    keys = np.asarray(keys) # (N_B)
    return bags, masks, keys

def bags_to_words(bags, word_height, word_width, pid):
    # bags: (N_B, B_H, B_W, C)
    num_bags, bag_height, bag_width, channel = bags.shape
    # (N_B, B_H, B_W, C) -> (N_B, B_H, N_W, W_W, C)
    words = np.reshape(bags, (num_bags, bag_height, -1, word_width, channel))
    # (N_B, B_H, N_W, W_W, C) -> (N_B, N_W, B_H, W_W, C)
    words = words.transpose(0, 2, 1, 3, 4)
    # (N_B, N_W, B_H, W_W, C) -> (N_B, N_W*N_H, W_H, W_W, C)
    words = np.reshape(words, (num_bags, -1, word_height, word_width, channel))

    words = normalize_words_np(words)
    words_torch = torch.from_numpy(words).float()
    del words 
    gc.collect()

    return words_torch

def masks_to_words(masks, word_height, word_width, pid):
    # masks: (N_B, B_H, B_W)
    num_bags, bag_height, bag_width = masks.shape
    # (N_B, B_H, B_W) -> (N_B, B_H, N_W, W_W)
    words = np.reshape(masks, (num_bags, bag_height, -1, word_width))
    # (N_B, B_H, N_W, W_W) -> (N_B, N_W, B_H, W_W)
    words = words.transpose(0, 2, 1, 3)
    # (N_B, N_W, B_H, W_W) -> (N_B, N_W*N_H, W_H, W_W)
    words = np.reshape(words, (num_bags, -1, word_height, word_width))

    words_torch = torch.from_numpy(words).long()
    del words

    return words_torch


# 影像临床特征（七个特征）
ln_to_id = {
    "stomach": 0,
    "LN": 1,
    "Liver": 2,
    "Aden": 3,
    "Soft": 4,
    "Peritoneum": 5,
    "Other": 6,
    "Spleen": 7,
    "Bone": 8
}

def ccd_yx_stomach(pid_dir):
    # 影像：有无原发灶
    if os.path.isdir(pid_dir):
        stomach_files = [file for file in glob(os.path.join(pid_dir, "v_stomach*.jpg")) if not file.endswith("_mask.jpg")]
        if len(stomach_files):
            return torch.FloatTensor([0.0, 1.0]) # 有原发灶
        else:
            return torch.FloatTensor([1.0, 0.0]) # 无原发灶
    else:
        return torch.FloatTensor([0.0, 0.0])

def ccd_yx_ln_num(pid_dir):
    # 影像：转移灶数量
    ln_num = [0.0] * 7
    if os.path.isdir(pid_dir):
        stomach_files = [file for file in glob(os.path.join(pid_dir, "v_stomach*.jpg")) if not file.endswith("_mask.jpg")]
        files = [file for file in glob(os.path.join(pid_dir, "v_*.jpg")) if not file.endswith("_mask.jpg")]
        ln_num[len(files)-len(stomach_files)] = 1.0
        return torch.FloatTensor(ln_num)
    else:
        return torch.FloatTensor(ln_num)

def ccd_yx_zl_ln(x):
    # 影像：转移淋巴结（0 没有转移淋巴结  1存在转移淋巴结，但不融合  2存在融合的转移淋巴结）
    zl_ln = [0.0] * 3
    if str(x) != str(np.nan):
        zl_ln[int(x)] = 1.0
    return torch.FloatTensor(zl_ln)

def ccd_yx_zl_ln_pos(x):
    # 影像：转移淋巴结位置（0没有转移淋巴结  1 存在局域转移淋巴结 2 存在M分期的腹腔或腹膜后淋巴结转移 3存在纵隔或锁骨上淋巴结转移 4存在其他少见远隔部位淋巴结转移（如腋窝、颈旁、腹股沟等等区域）
    zl_ln_pos = [0.0] * 5
    if str(x) != str(np.nan): 
        for i in str(int(x)):
            zl_ln_pos[int(i)] = 1.0
    return torch.FloatTensor(zl_ln_pos)

def ccd_yx_zl_multi(x):
    # 影像：肝或肺多发转移，多发为≥3个病灶（0没有多发肝转移和多发肺转移 1仅存在多发肝转移 2仅存在多发肺转移 3两者均有）
    zl_multi = [0.0] * 4
    if str(x) != str(np.nan):
        zl_multi[int(x)] = 1.0
    return torch.FloatTensor(zl_multi)

def ccd_yx_zl_per(x):
    # 影像：腹膜转移（0 无腹膜转移 1存在腹膜转移）
    zl_per = [0.0] * 2
    if str(x) != str(np.nan):
        zl_per[int(x)] = 1.0
    return torch.FloatTensor(zl_per)

def ccd_bl_tils(x):
    # 病理：肿瘤相关淋巴细胞TILs
    tils = [0.0] * 10
    if str(x) != str(np.nan):
        tils[int(x*100)//10] = 1.0
    return torch.FloatTensor(tils)

def ccd_bl_her2(x):
    # 病理：HER2表达异质性
    her2 = [0.0] * 4
    if str(x) != str(np.nan):
        if x.startswith("无"):
            her2[int(x[3:4])] = 1.0
        elif x.startswith("异质性"):
            for t in x[4:-1].split("，"):
                try:
                    her2[int(t[:1])] = 1.0*int(t[2:-1])/100.0        
                except:
                    pass
        else: 
            raise NotImplementedError
    return torch.FloatTensor(her2)

def ccd_bl_tumor(x):
    tumor = [0.0] * 10
    if str(x) != str(np.nan):
        tumor[int(x*100)//10] = 1.0
    return torch.FloatTensor(tumor)

# 患者层面临床特征（四个特征）

def ccd_sex(x):
    # 患者：性别
    x = str(x).strip()
    if x in ["男"]:
        return torch.FloatTensor([0.0, 1.0])
    elif x in ["女"]:
        return torch.FloatTensor([1.0, 0.0])
    else:
        raise NotImplementedError

def ccd_age(x, split_age=60):
    # 患者：年龄
    if x <= split_age:
        return torch.FloatTensor([1.0, 0.0])
    elif x > split_age:
        return torch.FloatTensor([0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_buwei(x):
    # 患者：肿瘤部位
    x = str(x).strip()
    if x in ["GEJ"]:
        return torch.FloatTensor([1.0, 0.0])
    elif x in ["non-GJE"]:
        return torch.FloatTensor([0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_xianshu(x):
    # 患者：治疗线数
    x = str(x).strip()
    if "一线" in x:
        return torch.FloatTensor([1.0, 0.0])
    else:
        return torch.FloatTensor([0.0, 1.0])

def ccd_time(x):
    # 患者：开始治疗时间
    if int(x) >= 2007 and int(x) < 2012:
        return torch.FloatTensor([1.0, 0.0, 0.0])
    elif int(x) >= 2012 and int(x) < 2017:
        return torch.FloatTensor([0.0, 1.0, 0.0])
    elif int(x) >= 2017 and int(x) <= 2023:
        return torch.FloatTensor([0.0, 0.0, 1.0])
    else:
        raise NotImplementedError

def ccd_fenxing(x):
    # 病理：LAUREN分型
    if x == "肠型": 
        return torch.FloatTensor([1.0, 0.0, 0.0])
    elif x == "弥漫型": 
        return torch.FloatTensor([0.0, 1.0, 0.0])
    elif x == "混合型": 
        return torch.FloatTensor([0.0, 0.0, 1.0])
    elif str(x) == str(np.nan): 
        return torch.FloatTensor([0.0, 0.0, 0.0])
    else:
        raise NotImplementedError

def ccd_fenhua(x):
    # 病理：分化程度
    if x in ["低分化"]: 
        return torch.FloatTensor([1.0, 0.0, 0.0, 0.0])
    elif x in ["中分化"]: 
        return torch.FloatTensor([0.0, 1.0, 0.0, 0.0])
    elif x in ["高分化"]: 
        return torch.FloatTensor([0.0, 0.0, 1.0, 0.0])
    elif x in ["弥漫型"]:
        return torch.FloatTensor([0.0, 0.0, 0.0, 1.0])
    elif str(x) == str(np.nan):
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
    else:
        return torch.FloatTensor([0.0, 0.0, 0.0, 0.0])
        raise NotImplementedError

def ccd_ln_dis(x):
    # 影像：病灶分布
    x = str(x)
    ln_dis = [0.0] * 13
    for i in range(len(x)):
        ln_dis[i] = float(int(x[i]))
    return torch.FloatTensor(ln_dis)

def ccd_yx_flag(x):
    yx_flag = [0.0] * 3
    if x >= 0: yx_flag[x] = 1.0
    return torch.FloatTensor(yx_flag)

def ccd_bl_flag(x):
    bl_flag = [0.0] * 3
    if x >= 0: bl_flag[x] = 1.0
    return torch.FloatTensor(bl_flag)

def convert_label(label_type, OS, OSCensor, liaoxiao, PFS=None, PFSCensor=None):
    if label_type == "ORR":
        if liaoxiao in ["CR", "PR"]:
            return 1
        elif liaoxiao in ["SD", "PD"]:
            return 0
        else:
            return -1
    elif label_type == "ORR_OS180":
        if liaoxiao in ["CR", "PR"]:
            return 1
        elif liaoxiao in ["PD",]:
            return 0
        else:
            cutoff = 180 # about six months
            if OS > cutoff:
                return 1
            elif OSCensor == 1 and OS <= cutoff:
                return 0
            else:
                return -1            
    elif label_type == "ORR_PFS240":
        if liaoxiao in ["CR", "PR"]:
            return 1
        elif liaoxiao in ["PD",]:
            return 0
        else:
            cutoff = 240 # about eight months
            if PFS > cutoff:
                return 1
            elif PFSCensor == 1 and PFS <= cutoff:
                return 0
            else:
                return -1
    elif label_type == "ORR_PFS300":
        if liaoxiao in ["CR", "PR"]:
            return 1
        elif liaoxiao in ["PD",]:
            return 0
        else:
            cutoff = 300 # about ten months
            if PFS > cutoff:
                return 1
            elif PFSCensor == 1 and PFS <= cutoff:
                return 0
            else:
                return -1
    elif label_type == "PFS240":
        cutoff = 240 # about eight months
        if PFS > cutoff:
            return 1
        elif PFSCensor == 1 and PFS <= cutoff:
            return 0
        else:
            return -1
    elif label_type == "PFS300":
        cutoff = 300 # about ten months
        if PFS > cutoff:
            return 1
        elif PFSCensor == 1 and PFS <= cutoff:
            return 0
        else:
            return -1
    else:
        raise NotImplementedError