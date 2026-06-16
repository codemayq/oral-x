import torch
import torch.nn as nn
import torchvision.models as models
from transformers import AutoModel

from .config import Config

class GatedAttentionMIL(nn.Module):
    """
    带 Mask 的门控注意力 MIL
    """
    def __init__(self, dim):
        super().__init__()
        self.attention_V = nn.Sequential(nn.Linear(dim, dim), nn.Tanh())
        self.attention_U = nn.Sequential(nn.Linear(dim, dim), nn.Sigmoid())
        self.attention_weights = nn.Linear(dim, 1)

    def forward(self, x, mask=None):
        # x: (Batch, N_instances, Dim)
        # mask: (Batch, N_instances) - 1 for valid, 0 for pad
        
        # 1. 计算 Attention Scores
        A_V = self.attention_V(x)
        A_U = self.attention_U(x)
        A = self.attention_weights(A_V * A_U) # (B, N, 1)
        
        # 2. 处理 Mask (将填充部分的 Attention 设为负无穷)
        if mask is not None:
            mask = mask.unsqueeze(-1) # (B, N, 1)
            # mask 为 0 的地方填一个极小值，softmax 后变为 0
            # 注意：混合精度下 -1e9 会溢出 float16
            min_value = torch.finfo(A.dtype).min
            A = A.masked_fill(mask == 0, min_value)
        
        A = torch.softmax(A, dim=1) # (B, N, 1)
        
        # 3. 聚合
        M = torch.sum(x * A, dim=1) # (B, Dim)
        return M

class UnifiedMultimodalModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        # ============ 1. 文本分支 (Qwen) ============
        print(f"Loading Text Model: {Config.QWEN_MODEL_PATH}...")
        self.text_backbone = AutoModel.from_pretrained(
            Config.QWEN_MODEL_PATH, 
            trust_remote_code=True,
            torch_dtype=torch.float16 if Config.USE_AMP else torch.float32
        )
        
        if Config.FREEZE_TEXT_MODEL:
            for param in self.text_backbone.parameters():
                param.requires_grad = False
                
        # Qwen hidden size 自动获取
        self.text_hidden_size = self.text_backbone.config.hidden_size
        self.text_proj = nn.Linear(self.text_hidden_size, Config.FUSION_DIM)

        # ============ 2. 普通图片分支 (ResNet101) ============
        # 升级: ResNet50 -> ResNet101
        print("Loading Image Model: ResNet101...")
        resnet = models.resnet101(weights='DEFAULT')
        # 去掉最后两层 (AvgPool, FC)，只保留特征层
        # 输出: (B, 2048, 7, 7) -> 需要先 Pool -> (B, 2048)
        self.img_backbone = nn.Sequential(*list(resnet.children())[:-2]) 
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.img_mil = GatedAttentionMIL(dim=Config.FUSION_DIM)
        # ResNet 特征 2048 -> 投影到 512
        self.img_proj = nn.Linear(2048, Config.FUSION_DIM)

        # ============ 3. WSI 分支 (Virchow2) ============
        # 输入维度 1280
        # 策略：先投影降维到 512，再做 MIL，大幅节省显存
        self.wsi_proj_pre = nn.Linear(Config.WSI_INPUT_DIM, Config.FUSION_DIM)
        self.wsi_mil = GatedAttentionMIL(dim=Config.FUSION_DIM)

        # ============ 4. 融合与分类 ============
        # Cross-Modal Fusion: 使用 Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=Config.FUSION_DIM, 
            nhead=Config.FUSION_HEADS, 
            dim_feedforward=Config.FUSION_DIM * 4,
            dropout=Config.FUSION_DROPOUT,
            batch_first=True
        )
        self.fusion_transformer = nn.TransformerEncoder(encoder_layer, num_layers=Config.FUSION_LAYERS)
        
        # [CLS] Token 用于最终分类
        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, Config.FUSION_DIM))
        
        self.classifier = nn.Sequential(
            nn.LayerNorm(Config.FUSION_DIM),
            nn.Dropout(0.3),
            nn.Linear(Config.FUSION_DIM, Config.NUM_CLASSES)
        )

    def forward(self, input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask):
        batch_size = input_ids.size(0)

        # --- 1. Text Forward ---
        # 如果冻结了，使用 no_grad
        if Config.USE_TXT:
            with torch.set_grad_enabled(not Config.FREEZE_TEXT_MODEL):
                text_out = self.text_backbone(input_ids=input_ids, attention_mask=attention_mask)
                # (B, L, H) -> (B, H)
                txt_emb = text_out.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(txt_emb.size()).float()
                txt_emb = torch.sum(txt_emb * mask_expanded, 1) / torch.clamp(mask_expanded.sum(1), min=1e-9)
            txt_feat = self.text_proj(txt_emb) # (B, 512)
        else:
            # 返回全0特征 (B, 512)
            txt_feat = torch.zeros(batch_size, Config.FUSION_DIM, device=input_ids.device, dtype=self.text_proj.weight.dtype)


        # --- 2. Normal Image Forward ---
        # input: (B, N, 3, 224, 224)
        if Config.USE_IMG:
            B, N, C, H, W = normal_imgs.shape
            # Flatten batch and N: (B*N, 3, 224, 224)
            imgs_flat = normal_imgs.view(B * N, C, H, W)
            
            cnn_feat = self.img_backbone(imgs_flat) # (B*N, 2048, 7, 7)
            cnn_feat = self.img_pool(cnn_feat).flatten(1) # (B*N, 2048)
            
            # Unflatten: (B, N, 2048)
            cnn_feat = cnn_feat.view(B, N, -1)
            
            # 投影到 512
            cnn_feat = self.img_proj(cnn_feat) # (B, N, 512)
            
            # MIL 聚合: (B, N, 512) -> (B, 512)
            img_feat = self.img_mil(cnn_feat) 
        else:
             img_feat = torch.zeros(batch_size, Config.FUSION_DIM, device=input_ids.device, dtype=self.img_proj.weight.dtype)


        # --- 3. WSI Forward ---
        # input: (B, Total_Tokens, 1280)
        if Config.USE_SVS:
            # 先降维，省显存！
            wsi_feat = self.wsi_proj_pre(wsi_feat) # (B, Total_Tokens, 512)
            
            # 激活函数
            wsi_feat = torch.relu(wsi_feat)
            
            # MIL 聚合 (带 Mask): (B, Total_Tokens, 512) -> (B, 512)
            wsi_feat_agg = self.wsi_mil(wsi_feat, mask=wsi_mask)
        else:
            wsi_feat_agg = torch.zeros(batch_size, Config.FUSION_DIM, device=input_ids.device, dtype=self.wsi_proj_pre.weight.dtype)

        # --- 4. Fusion ---

        # --- 4. Fusion ---
        # Stack: [CLS, Text, Normal, WSI]
        cls_tokens = self.fusion_cls_token.expand(batch_size, -1, -1) # (B, 1, 512)
        
        # (B, 1, 512), (B, 1, 512), (B, 1, 512), (B, 1, 512)
        # Unsqueeze features to match sequence dim
        fusion_input = torch.cat([
            cls_tokens, 
            txt_feat.unsqueeze(1), 
            img_feat.unsqueeze(1), 
            wsi_feat_agg.unsqueeze(1)
        ], dim=1) # (B, 4, 512)
        
        fusion_out = self.fusion_transformer(fusion_input)
        
        # 取 CLS token
        final_emb = fusion_out[:, 0, :]
        
        logits = self.classifier(final_emb)
        return logits