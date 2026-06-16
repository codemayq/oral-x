import os
import torch
import torch.nn as nn
from PIL import Image
from transformers import AutoTokenizer
import torchvision.transforms as transforms


from .config import Config
from .model import UnifiedMultimodalModel
from .wsi_processor_fast import get_virchow2_backbone, extract_wsi_features

class InferencePipeline:
    def __init__(self, model_checkpoint_path):
        self.device = torch.device(Config.DEVICE)
        
        # 1. 加载主模型
        print("Loading Main Multimodal Model...")
        self.model = UnifiedMultimodalModel().to(self.device)
        
        # 加载权重
        print(f"Loading checkpoint from {model_checkpoint_path}")
        checkpoint = torch.load(model_checkpoint_path, map_location=self.device)
        # 兼容只保存了 state_dict 的情况，也兼容保存了完整 dict 的情况
        if 'state_dict' in checkpoint:
            state_dict = checkpoint['state_dict']
        elif 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
            
        self.model.load_state_dict(state_dict)
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
        
        print("Inference Pipeline Ready!")

    def predict(self, text_input: str, image_paths: list, svs_paths: list):
        # 兼容旧接口，调用新方法
        return self.predict_proba(text_input, image_paths, svs_paths)

    def predict_proba(self, text_input: str, image_paths: list, svs_paths: list):
        """
        进行单样本推理
        多标签模式: 使用 sigmoid，各类别独立概率
        单标签模式: 使用 softmax，概率分布
        Returns:
            probabilities: numpy array of shape (NumClasses,)
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
            # Stack and add batch dim: (N, C, H, W) -> (1, N, C, H, W)
            normal_imgs = torch.stack(img_tensors).unsqueeze(0).to(self.device)
        else:
            # Placeholder: (1, 1, 3, 224, 224)
            normal_imgs = torch.zeros(1, 1, 3, 224, 224).to(self.device)

        # --- 3. 处理 WSI (实时提取特征) ---
        wsi_feat_list = []
        for svs_path in svs_paths:
            if not os.path.exists(svs_path):
                print(f"Warning: SVS file not found: {svs_path}")
                continue
            
            try:
                # 提取特征
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
            # (Total_Tokens, D) -> (1, Total_Tokens, D)
            wsi_feat = torch.cat(wsi_feat_list, dim=0).unsqueeze(0).to(self.device)
            wsi_mask = torch.ones(1, wsi_feat.size(1)).to(self.device)
        else:
            # Placeholder
            wsi_feat = torch.zeros(1, 1, Config.WSI_INPUT_DIM).to(self.device)
            wsi_mask = torch.zeros(1, 1).to(self.device)

        # --- 4. 模型推理 ---
        self.model.eval()
        with torch.no_grad():
            with torch.amp.autocast(self.device.type, enabled=Config.USE_AMP, dtype=torch.float16):
                logits = self.model(input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask)

                if Config.MULTI_LABEL:
                    probs = torch.sigmoid(logits)
                else:
                    probs = torch.softmax(logits, dim=1)

        return probs[0].cpu().float().numpy()

def main():
    # 示例用法
    # 1. 设置权重路径
    checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "best_val.pth")
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint not found at {checkpoint_path}, trying last.pth")
        checkpoint_path = os.path.join(Config.CHECKPOINT_DIR, "last.pth")
        
    if not os.path.exists(checkpoint_path):
        print("No checkpoints found! Please train the model first.")
        return

    # 2. 初始化推理管道
    pipeline = InferencePipeline(checkpoint_path)

    # 3. 准备测试数据 (请替换为实际路径)
    test_text = "患者左颊粘膜白色斑纹"
    test_images = []
    test_svs = []

    print("\nStarting Inference...")
    probs = pipeline.predict_proba(test_text, test_images, test_svs)
    
    # 4. 输出结果
    print("\n================ Results ================")
    print("Probabilities:")
    for idx, prob in enumerate(probs):
        class_name = Config.TARGET_CLASS_NAMES[idx] if idx < len(Config.TARGET_CLASS_NAMES) else f"Class {idx}"
        print(f"  {class_name}: {prob:.4f}")
    print("=========================================")

if __name__ == "__main__":
    main()
