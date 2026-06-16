import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms

from .config import Config
from .moe_model import MoEMultimodalModel
from .wsi_processor_fast import get_virchow2_backbone, extract_wsi_features

class MoEInferencePipeline:
    def __init__(self, model_checkpoint_path):
        self.device = torch.device(Config.DEVICE)
        
        # 1. 加载 MoE 模型
        print("Loading MoE Multimodal Model...")
        self.model = MoEMultimodalModel().to(self.device)
        
        # 加载权重
        print(f"Loading checkpoint from {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        # 移除可能的 module. 前缀 (如果用了 DataParallel)
        new_state_dict = {}
        for k, v in state_dict.items():
            if k.startswith('module.'):
                new_state_dict[k[7:]] = v
            else:
                new_state_dict[k] = v
        
        try:
            self.model.load_state_dict(new_state_dict)
        except RuntimeError as e:
            print(f"Error loading state dict: {e}")
            print("Attempting strict=False load...")
            self.model.load_state_dict(new_state_dict, strict=False)
            
        self.model.eval()
        
        # 2. 文本预处理
        print("Initializing Text Processor...")
        self.tokenizer = AutoTokenizer.from_pretrained(Config.QWEN_MODEL_PATH, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        # 3. 普通图片预处理
        print("Initializing Normal Image Processor...")
        self.normal_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # 4. WSI 预处理 (Virchow2)
        print("Initializing Virchow2 for WSI...")
        self.virchow_model, self.virchow_transform = get_virchow2_backbone(Config.VIRCHOW2_MODEL_ID, self.device)
        
        print("MoE Inference Pipeline Ready!")

    def predict(self, text_input: str, image_paths: list, svs_paths: list):
        return self.predict_proba(text_input, image_paths, svs_paths)

    def predict_proba(self, text_input: str, image_paths: list, svs_paths: list):
        """
        进行单样本推理
        """
        
        # --- 1. 处理文本 ---
        text_enc = self.tokenizer(
            text_input if text_input else "",
            max_length=Config.MAX_TEXT_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = text_enc['input_ids'].to(self.device)        # (1, L)
        attention_mask = text_enc['attention_mask'].to(self.device) # (1, L)
        
        # --- 2. 处理普通图片 ---
        img_tensors = []
        for path in image_paths:
            if os.path.exists(path):
                try:
                    img = Image.open(path).convert('RGB')
                    img_tensors.append(self.normal_transform(img))
                except Exception as e:
                    print(f"Warning: Failed to load image {path}: {e}")
        
        if img_tensors:
            normal_imgs = torch.stack(img_tensors).unsqueeze(0).to(self.device)
        else:
            normal_imgs = torch.zeros(1, 1, 3, 224, 224).to(self.device)

        # --- 3. 处理 WSI ---
        wsi_feat_list = []
        for svs_path in svs_paths:
            if not os.path.exists(svs_path):
                continue
            
            try:
                if svs_path.endswith('.svs'):
                    feat = extract_wsi_features(svs_path, None, self.virchow_model, self.virchow_transform)   
                elif svs_path.endswith('.pt'):
                    feat = torch.load(svs_path, map_location='cpu')
                if feat.ndim == 3:
                    M, T, D = feat.shape
                    feat = feat.view(M * T, D)
                
                wsi_feat_list.append(feat)
            except Exception as e:
                print(f"Error processing WSI {svs_path}: {e}")

        if wsi_feat_list:
            wsi_feat = torch.cat(wsi_feat_list, dim=0).unsqueeze(0).to(self.device)
            wsi_mask = torch.ones(1, wsi_feat.size(1)).to(self.device)
        else:
            wsi_feat = torch.zeros(1, 1, Config.WSI_INPUT_DIM).to(self.device)
            wsi_mask = torch.zeros(1, 1).to(self.device)

        # --- 4. 模型推理 ---
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=Config.USE_AMP, dtype=torch.float16):
                # MoE 模型 forward 在 eval 模式下直接返回 logits
                logits = self.model(input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask)

                if Config.MULTI_LABEL:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)

        return probs[0].cpu().float().numpy()
