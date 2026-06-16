import torch
from torch.utils.data import Dataset
import pandas as pd
import os
import random
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms

from .config import Config, map_label_for_3cls_training


class MultimodalDataset(Dataset):
    def __init__(self, data_source, mode='train', expand_factor=1, use_txt=True, use_img=True, use_svs=True):
        """
        Args:
            data_source: CSV路径 (str) 或 pd.DataFrame 对象
            mode: 'train' 或 'val'/'test'
            expand_factor: (int) 仅在训练模式有效。将数据集复制多少倍。
            use_txt: 是否使用文本模态
            use_img: 是否使用普通图片模态
            use_svs: 是否使用 WSI SVS 模态
        """
        self.mode = mode
        self.use_txt = use_txt
        self.use_img = use_img
        self.use_svs = use_svs
        
        if isinstance(data_source, str):
            self.data = pd.read_csv(data_source)
        else:
            self.data = data_source.copy()
        
        # [离线增强思路]：通过复制 DataFrame 来扩充样本基数
        if self.mode == 'train' and expand_factor > 1:
            self.data = pd.concat([self.data] * expand_factor, ignore_index=True)
            # 打乱顺序，避免连续看到同一个样本的变体
            self.data = self.data.sample(frac=1).reset_index(drop=True)

        # 初始化 Qwen Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(Config.QWEN_MODEL_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # [在线增强思路]：训练集使用强数据增强
        if self.mode == 'train':
            self.transform = transforms.Compose([
                transforms.Resize((256, 256)), # 先放大一点
                transforms.RandomResizedCrop(224, scale=(0.8, 1.0)), # 随机裁剪
                transforms.RandomHorizontalFlip(p=0.5), # 随机水平翻转
                transforms.RandomVerticalFlip(p=0.5),   # 随机垂直翻转
                transforms.RandomRotation(15),          # 随机旋转
                transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1), # 颜色抖动
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
        else:
            # 验证/测试集仅做标准化
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

    def __len__(self):
        return len(self.data)

    def _get_random_subset(self, paths_str, min_count=0):
        """
        [离线/结构增强思路]：从路径字符串中随机选取子集
        paths_str: "path1;path2;path3"
        """
        if not isinstance(paths_str, str) or paths_str == 'nan':
            return []
        
        paths = paths_str.split(';')
        if len(paths) == 0:
            return []

        if self.mode == 'train':
            # 随机选取 k 个，k 在 [min_count, len] 之间
            # 允许选 0 个，模拟模态缺失
            k = random.randint(min_count, len(paths))
            if k == 0:
                return []
            return random.sample(paths, k)
        else:
            # 验证/测试模式，全选
            return paths

    def _augment_text(self, text):
        """
        [在线增强思路]：文本增强
        """
        if self.mode != 'train':
            return text
        
        # 1. 模态丢失模拟 (10% 概率文本完全丢失)
        if random.random() < 0.1:
            return ""
        
        # 2. 内容随机截取 (20% 概率随机截断文本)
        # 这里只是简单的按字符截断，也可以按句子截断
        if len(text) > 10 and random.random() < 0.2:
            cut_len = random.randint(int(len(text)*0.5), len(text))
            return text[:cut_len]
            
        return text

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        
        # ================= 1. 读取文本文件 =================
        txt_path = row['txt_path']
        text_content = ""
        try:
            if self.use_txt and isinstance(txt_path, str) and os.path.exists(txt_path):
                with open(txt_path, 'r', encoding='utf-8') as f:
                    text_content = f.read()
        except Exception:
            pass
            
        # 应用文本增强
        text_content = self._augment_text(text_content)
        
        text_enc = self.tokenizer(
            text_content,
            max_length=Config.MAX_TEXT_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_enc['input_ids'].squeeze(0)
        attention_mask = text_enc['attention_mask'].squeeze(0)

        # ================= 2. 读取普通图片 (多个) =================
        # 随机选取子集 (可能选出 0 张，模拟图片模态缺失)
        if self.use_img:
            img_paths = self._get_random_subset(str(row['img_paths']), min_count=0)
        else:
            img_paths = []
        
        img_tensors = []
        for path in img_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGB')
                    # 应用图片增强 (crop, rotate, etc.)
                    img_tensors.append(self.transform(img))
                except:
                    pass
        
        if len(img_tensors) > 0:
            normal_imgs = torch.stack(img_tensors)
        else:
            # 如果没有图片（或被随机mask掉了），返回全0张量
            normal_imgs = torch.zeros(1, 3, 224, 224) 

        # ================= 3. 读取 WSI 特征 (可能多个) =================
        # 随机选取子集
        if self.use_svs:
            wsi_paths = self._get_random_subset(str(row['wsi_paths']), min_count=0)
        else:
            wsi_paths = []
        
        wsi_feat_list = []
        for path in wsi_paths:
            if os.path.exists(path):
                try:
                    feat = torch.load(path, map_location='cpu')
                    if isinstance(feat, dict) and 'features' in feat:
                        feat = feat['features']
                    if feat.ndim == 3:
                        M, T, D = feat.shape
                        feat = feat.view(M * T, D)
                    wsi_feat_list.append(feat.float())
                except Exception as e:
                    pass
        
        if len(wsi_feat_list) > 0:
            wsi_feat = torch.cat(wsi_feat_list, dim=0) 
        else:
            # 如果没有 WSI (或被随机mask掉了)
            wsi_feat = torch.zeros(1, Config.WSI_INPUT_DIM)

        # ================= Label 处理 =================
        old_label = int(row['label'])

        if Config.MULTI_LABEL:
            # 多标签模式 (7类): OSF+OLK 拆为两个独立标签
            target = torch.zeros(Config.NUM_CLASSES, dtype=torch.float32)
            label_mapping = {
                0: [0],       # OLK
                1: [1],       # OLP
                2: [2],       # OSCC
                3: [3],       # OSF
                4: [0, 3],    # OSF+OLK -> OLK + OSF
                5: [4],       # SCP
                6: [5],       # Mucocele
                7: [6]        # Fibroma
            }
            if old_label in label_mapping:
                for new_idx in label_mapping[old_label]:
                    target[new_idx] = 1.0
            label = target
        else:
            # 单标签：8 类为 0–7；3 类且开启 MAP_RAW_LABEL_TO_MAJOR_3CLS 时映射为大类 0–2
            label = torch.tensor(map_label_for_3cls_training(old_label), dtype=torch.long)

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'normal_imgs': normal_imgs,
            'wsi_feat': wsi_feat,
            'label': label
        }

def collate_fn(batch):
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])

    if Config.MULTI_LABEL:
        labels = torch.stack([item['label'] for item in batch])          # (B, C)
    else:
        labels = torch.stack([item['label'] for item in batch])          # (B,)
    
    # Normal Imgs padding
    normal_imgs = [item['normal_imgs'] for item in batch]
    max_n = max([img.shape[0] for img in normal_imgs])
    padded_imgs = []
    for img in normal_imgs:
        pad_size = max_n - img.shape[0]
        if pad_size > 0:
            pad = torch.zeros(pad_size, 3, 224, 224)
            img = torch.cat([img, pad], dim=0)
        padded_imgs.append(img)
    normal_imgs_batch = torch.stack(padded_imgs)
    
    # WSI padding
    wsi_feats = [item['wsi_feat'] for item in batch]
    wsi_batch = torch.nn.utils.rnn.pad_sequence(wsi_feats, batch_first=True, padding_value=0)
    
    wsi_mask = torch.zeros(wsi_batch.shape[0], wsi_batch.shape[1])
    for i, feat in enumerate(wsi_feats):
        wsi_mask[i, :feat.shape[0]] = 1
        
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'normal_imgs': normal_imgs_batch,
        'wsi_feat': wsi_batch,
        'wsi_mask': wsi_mask,
        'labels': labels
    }
