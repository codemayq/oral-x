"""
MoE 增强多模态模型 v2 —— DeepSeek 风格 Sparse MoE Transformer

核心改进（相比 v1）：
1. SparseMoE 嵌入 Transformer FFN: shared expert + N routed experts (DeepSeek 风格)
2. 稳定训练: 提高原型温度, router z-loss, EMA 支持
3. 层次化分类 + 类别原型保留, 但去掉不稳定的 feature-level mixup
4. 分类头改为简洁的 MoE 分类器
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from transformers import AutoModel

from .config import Config
from .model import GatedAttentionMIL


# ============================================================
#  1. DeepSeek-style SparseMoE FFN Block
# ============================================================

class MoEExpertFFN(nn.Module):
    """单个 FFN 专家: Linear -> GELU -> Linear, 输出层缩放初始化"""

    def __init__(self, dim: int, ffn_dim: int, dropout: float = 0.1,
                 init_scale: float = 1.0):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim)
        self.w2 = nn.Linear(ffn_dim, dim)
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # 残差路径输出层缩放: GPT-2 / DeepSeek 风格
        # 让深层网络初始阶段残差贡献极小
        if init_scale != 1.0:
            with torch.no_grad():
                self.w2.weight.mul_(init_scale)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))


class SparseMoEFFN(nn.Module):
    """
    Sparse MoE FFN。

    架构:
    - 1 个 shared expert (所有 token 都经过)
    - N 个 routed experts (Top-K 稀疏路由)
    - 输出 = shared_out + routed_scale * sum(gate_i * expert_i_out)
    - routed_scale 是可学习的标量，初始值很小，避免初始阶段 routed 输出太大

    辅助损失:
    - 负载均衡损失 (Switch Transformer)
    - Router z-loss (稳定训练, 防止 logits 过大)
    """

    def __init__(self, dim: int, ffn_dim: int, num_routed_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.1,
                 residual_scale: float = 1.0):
        super().__init__()
        self.num_routed_experts = num_routed_experts
        self.top_k = top_k

        self.shared_expert = MoEExpertFFN(dim, ffn_dim, dropout, init_scale=residual_scale)

        self.routed_experts = nn.ModuleList([
            MoEExpertFFN(dim, ffn_dim, dropout, init_scale=residual_scale)
            for _ in range(num_routed_experts)
        ])

        self.router = nn.Linear(dim, num_routed_experts, bias=False)
        nn.init.zeros_(self.router.weight)

        self.routed_scale = nn.Parameter(torch.tensor(0.01))

    def forward(self, x):
        """
        Args:
            x: (B, S, dim)
        Returns:
            out: (B, S, dim)
            aux_loss: scalar
        """
        B, S, D = x.shape
        x_flat = x.reshape(B * S, D)  # (T, D) where T = B*S

        shared_out = self.shared_expert(x_flat)  # (T, D)

        router_logits = self.router(x_flat)  # (T, E)
        router_probs = F.softmax(router_logits, dim=-1)  # (T, E)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)  # (T, K)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        all_expert_out = torch.stack(
            [expert(x_flat) for expert in self.routed_experts], dim=1
        )  # (T, E, D)

        idx_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, D)  # (T, K, D)
        selected_out = torch.gather(all_expert_out, 1, idx_expanded)  # (T, K, D)
        routed_out = (selected_out * top_k_probs.unsqueeze(-1)).sum(dim=1)  # (T, D)

        out = shared_out + self.routed_scale * routed_out  # (T, D)
        out = out.reshape(B, S, D)

        # --- 辅助损失 ---
        mask = F.one_hot(top_k_indices, self.num_routed_experts).float().sum(dim=1)  # (T, E)
        f = mask.mean(dim=0)
        P = router_probs.mean(dim=0)
        balance_loss = self.num_routed_experts * (f * P).sum()

        z_loss = torch.logsumexp(router_logits, dim=-1).square().mean()

        aux_loss = balance_loss + 0.01 * z_loss

        return out, aux_loss


# ============================================================
#  2. MoE Transformer Encoder Layer
# ============================================================

class MoETransformerEncoderLayer(nn.Module):
    """
    标准 Transformer Encoder Layer, 但 FFN 替换为 SparseMoE FFN。
    结构: LayerNorm -> Self-Attention -> Residual -> LayerNorm -> SparseMoE FFN -> Residual

    residual_scale: 按 1/sqrt(2*N_layers) 缩放残差路径的输出权重初始化,
    防止深层网络初始阶段梯度/激活值爆炸 (GPT-2 / DeepSeek 风格)。
    """

    def __init__(self, dim: int, nhead: int, ffn_dim: int,
                 num_routed_experts: int = 4, top_k: int = 2,
                 dropout: float = 0.1, residual_scale: float = 1.0):
        super().__init__()

        self.self_attn = nn.MultiheadAttention(
            embed_dim=dim, num_heads=nhead, dropout=dropout, batch_first=True,
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout1 = nn.Dropout(dropout)

        self.moe_ffn = SparseMoEFFN(
            dim=dim, ffn_dim=ffn_dim,
            num_routed_experts=num_routed_experts,
            top_k=top_k, dropout=dropout,
            residual_scale=residual_scale,
        )

        # 缩放 attention 输出投影的初始化
        if residual_scale != 1.0:
            with torch.no_grad():
                self.self_attn.out_proj.weight.mul_(residual_scale)

    def forward(self, x):
        x2 = self.norm1(x)
        attn_out, _ = self.self_attn(x2, x2, x2)
        x = x + self.dropout1(attn_out)

        x2 = self.norm2(x)
        ffn_out, aux_loss = self.moe_ffn(x2)
        x = x + ffn_out

        return x, aux_loss


class MoETransformerEncoder(nn.Module):
    """
    堆叠多层 MoETransformerEncoderLayer, 累积所有层的辅助损失。

    支持混合配置: 可以让前 N 层用普通 FFN, 后 M 层用 MoE FFN,
    通过 moe_layer_interval 控制 (默认每层都用 MoE)。
    """

    def __init__(self, dim: int, nhead: int, ffn_dim: int, num_layers: int,
                 num_routed_experts: int = 8, top_k: int = 2,
                 dropout: float = 0.1, moe_layer_interval: int = 1):
        super().__init__()
        self.layers = nn.ModuleList()
        self.is_moe = []

        # GPT-2 style: 每个残差分支的输出层初始化缩放 1/sqrt(2*N)
        # 每层有 2 个残差分支 (attn + ffn), 共 2*N 个残差分支
        import math
        residual_scale = 1.0 / math.sqrt(2 * num_layers)

        for i in range(num_layers):
            use_moe = (i % moe_layer_interval == 0)
            if use_moe:
                self.layers.append(MoETransformerEncoderLayer(
                    dim=dim, nhead=nhead, ffn_dim=ffn_dim,
                    num_routed_experts=num_routed_experts,
                    top_k=top_k, dropout=dropout,
                    residual_scale=residual_scale,
                ))
            else:
                layer = nn.TransformerEncoderLayer(
                    d_model=dim, nhead=nhead, dim_feedforward=ffn_dim,
                    dropout=dropout, batch_first=True,
                )
                # 对普通层也做相同的残差缩放
                with torch.no_grad():
                    layer.self_attn.out_proj.weight.mul_(residual_scale)
                    layer.linear2.weight.mul_(residual_scale)
                self.layers.append(layer)
            self.is_moe.append(use_moe)

        self.final_norm = nn.LayerNorm(dim)

    def forward(self, x):
        """
        Returns:
            out: (B, S, dim)
            total_aux_loss: scalar — 所有 MoE 层的辅助损失总和
        """
        total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        for layer, is_moe in zip(self.layers, self.is_moe):
            if is_moe:
                x, aux = layer(x)
                total_aux = total_aux + aux
            else:
                x = layer(x)

        x = self.final_norm(x)
        return x, total_aux


# ============================================================
#  3. 层次化分类头
# ============================================================

FINE_TO_COARSE = [1, 1, 0, 1, 1, 2, 2, 2]


class HierarchicalClassifier(nn.Module):
    """
    粗分类 (3类) + 粗->细 可学习映射。
    """

    def __init__(self, dim: int, num_fine: int = 8, num_coarse: int = 3):
        super().__init__()
        self.coarse_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_coarse),
        )
        self.register_buffer(
            "fine_to_coarse",
            torch.tensor(FINE_TO_COARSE[:num_fine], dtype=torch.long),
        )
        self.coarse_to_fine_proj = nn.Linear(num_coarse, num_fine, bias=False)

    def forward(self, x):
        coarse_logits = self.coarse_head(x)
        coarse_boost = self.coarse_to_fine_proj(coarse_logits)
        return coarse_logits, coarse_boost


# ============================================================
#  4. 可学习类别原型 (温度提高以稳定训练)
# ============================================================

class ClassPrototypes(nn.Module):

    def __init__(self, dim: int, num_classes: int, temperature: float = 0.2):
        super().__init__()
        self.prototypes = nn.Parameter(torch.randn(num_classes, dim) * 0.02)
        self.temperature = temperature

    def forward(self, x):
        x_norm = F.normalize(x, dim=-1)
        p_norm = F.normalize(self.prototypes, dim=-1)
        return torch.mm(x_norm, p_norm.t()) / self.temperature


# ============================================================
#  5. MoE 分类头 (简化版)
# ============================================================

class MoEClassifier(nn.Module):
    """轻量 MoE 分类头, 4 专家 Top-2 路由。"""

    def __init__(self, dim: int, num_classes: int, num_experts: int = 8,
                 top_k: int = 2, dropout: float = 0.2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k

        self.router = nn.Linear(dim, num_experts, bias=False)
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.LayerNorm(dim),
                nn.Linear(dim, dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(dim, num_classes),
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        router_logits = self.router(x)  # (B, E)
        router_probs = F.softmax(router_logits, dim=-1)

        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k, dim=-1)
        top_k_probs = top_k_probs / (top_k_probs.sum(dim=-1, keepdim=True) + 1e-9)

        expert_outputs = torch.stack([e(x) for e in self.experts], dim=1)  # (B, E, C)
        idx_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, expert_outputs.size(-1))
        selected = torch.gather(expert_outputs, 1, idx_expanded)
        logits = (selected * top_k_probs.unsqueeze(-1)).sum(dim=1)

        # 负载均衡
        mask = F.one_hot(top_k_indices, self.num_experts).float().sum(dim=1)
        f = mask.mean(dim=0)
        P = router_probs.mean(dim=0)
        aux_loss = self.num_experts * (f * P).sum()

        return logits, aux_loss


# ============================================================
#  6. 完整 MoE 多模态模型 v2
# ============================================================

class MoEMultimodalModel(nn.Module):
    """
    DeepSeek 风格 SparseMoE 多模态分类模型。

    与 v1 的核心区别: MoE 嵌入 Transformer FFN 内部, 而非仅在分类头。
    每一层 Transformer 都有 shared expert + routed experts,
    让不同专家学习不同的模态交互和类别判别模式。

    forward() 输入签名与原 UnifiedMultimodalModel 相同。
    训练时返回 dict, 推理时返回 logits 张量。
    """

    def __init__(self):
        super().__init__()

        num_classes = Config.NUM_CLASSES
        dim = Config.FUSION_DIM

        # ============ 1. 文本分支 (Qwen) ============
        print(f"[MoE-v2] Loading Text Model: {Config.QWEN_MODEL_PATH}...")
        self.text_backbone = AutoModel.from_pretrained(
            Config.QWEN_MODEL_PATH,
            trust_remote_code=True,
            torch_dtype=torch.float16 if Config.USE_AMP else torch.float32,
        )
        if Config.FREEZE_TEXT_MODEL:
            for param in self.text_backbone.parameters():
                param.requires_grad = False

        self.text_hidden_size = self.text_backbone.config.hidden_size
        self.text_proj = nn.Linear(self.text_hidden_size, dim)

        # ============ 2. 普通图片分支 (ResNet101) ============
        print("[MoE-v2] Loading Image Model: ResNet101...")
        resnet = models.resnet101(weights="DEFAULT")
        self.img_backbone = nn.Sequential(*list(resnet.children())[:-2])
        self.img_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.img_mil = GatedAttentionMIL(dim=dim)
        self.img_proj = nn.Linear(2048, dim)

        # ============ 3. WSI 分支 (Virchow2) ============
        self.wsi_proj_pre = nn.Linear(Config.WSI_INPUT_DIM, dim)
        self.wsi_mil = GatedAttentionMIL(dim=dim)

        # ============ 4. SparseMoE Transformer Fusion ============
        num_routed = getattr(Config, "MOE_NUM_ROUTED_EXPERTS", 4)
        moe_top_k = getattr(Config, "MOE_TOP_K", 2)
        # 每隔 moe_interval 层插入一个 MoE 层, 其余用普通 FFN
        # 例如 12层, interval=2: 层0,2,4,6,8,10 用 MoE, 层1,3,5,7,9,11 用普通FFN
        moe_interval = getattr(Config, "MOE_LAYER_INTERVAL", 2)

        self.fusion_transformer = MoETransformerEncoder(
            dim=dim,
            nhead=Config.FUSION_HEADS,
            ffn_dim=dim * 4,
            num_layers=Config.FUSION_LAYERS,
            num_routed_experts=num_routed,
            top_k=moe_top_k,
            dropout=Config.FUSION_DROPOUT,
            moe_layer_interval=moe_interval,
        )

        self.fusion_cls_token = nn.Parameter(torch.randn(1, 1, dim) * 0.02)

        # ============ 5. MoE 分类头 ============
        self.moe_classifier = MoEClassifier(
            dim=dim, num_classes=num_classes, num_experts=4, top_k=2,
        )

        # ============ 6. 层次化分类头 ============
        self.hierarchical = HierarchicalClassifier(
            dim=dim, num_fine=num_classes, num_coarse=3,
        )

        # ============ 7. 类别原型 ============
        self.prototypes = ClassPrototypes(
            dim=dim, num_classes=num_classes, temperature=0.2,
        )

        # 可学习的融合门控 (初始化使 sigmoid 接近 0.3, 让模型先依赖 MoE logits)
        self.hier_gate = nn.Parameter(torch.tensor(-0.85))
        self.proto_gate = nn.Parameter(torch.tensor(-0.85))

    # ------------------------------------------------------------------
    #  模态特征提取
    # ------------------------------------------------------------------

    def _extract_text(self, input_ids, attention_mask, batch_size):
        if Config.USE_TXT:
            with torch.set_grad_enabled(not Config.FREEZE_TEXT_MODEL):
                text_out = self.text_backbone(
                    input_ids=input_ids, attention_mask=attention_mask,
                )
                txt_emb = text_out.last_hidden_state
                mask_expanded = attention_mask.unsqueeze(-1).expand(txt_emb.size()).float()
                txt_emb = (
                    torch.sum(txt_emb * mask_expanded, 1)
                    / torch.clamp(mask_expanded.sum(1), min=1e-9)
                )
            # Ensure dtype matches the projection layer to avoid fp16/bf16 mismatch
            txt_emb = txt_emb.to(dtype=self.text_proj.weight.dtype)
            return self.text_proj(txt_emb)
        return torch.zeros(
            batch_size, Config.FUSION_DIM,
            device=input_ids.device, dtype=self.text_proj.weight.dtype,
        )

    def _extract_image(self, normal_imgs, batch_size, device):
        if Config.USE_IMG:
            # Ensure input matches model dtype (e.g. bfloat16)
            if normal_imgs.dtype != self.img_proj.weight.dtype:
                normal_imgs = normal_imgs.to(dtype=self.img_proj.weight.dtype)

            B, N, C, H, W = normal_imgs.shape
            imgs_flat = normal_imgs.view(B * N, C, H, W)
            cnn_feat = self.img_backbone(imgs_flat)
            cnn_feat = self.img_pool(cnn_feat).flatten(1)
            cnn_feat = cnn_feat.view(B, N, -1)
            cnn_feat = self.img_proj(cnn_feat)
            return self.img_mil(cnn_feat)
        return torch.zeros(
            batch_size, Config.FUSION_DIM,
            device=device, dtype=self.img_proj.weight.dtype,
        )

    def _extract_wsi(self, wsi_feat, wsi_mask, batch_size, device):
        if Config.USE_SVS:
            # Ensure input matches model dtype
            if wsi_feat.dtype != self.wsi_proj_pre.weight.dtype:
                wsi_feat = wsi_feat.to(dtype=self.wsi_proj_pre.weight.dtype)

            wsi_feat = self.wsi_proj_pre(wsi_feat)
            wsi_feat = torch.relu(wsi_feat)
            return self.wsi_mil(wsi_feat, mask=wsi_mask)
        return torch.zeros(
            batch_size, Config.FUSION_DIM,
            device=device, dtype=self.wsi_proj_pre.weight.dtype,
        )

    # ------------------------------------------------------------------
    #  Forward
    # ------------------------------------------------------------------

    def forward(self, input_ids, attention_mask, normal_imgs, wsi_feat, wsi_mask,
                labels=None):
        batch_size = input_ids.size(0)
        device = input_ids.device

        txt_feat = self._extract_text(input_ids, attention_mask, batch_size)
        img_feat = self._extract_image(normal_imgs, batch_size, device)
        wsi_feat_agg = self._extract_wsi(wsi_feat, wsi_mask, batch_size, device)

        # --- SparseMoE Cross-Modal Fusion ---
        cls_tokens = self.fusion_cls_token.expand(batch_size, -1, -1)
        fusion_input = torch.cat([
            cls_tokens,
            txt_feat.unsqueeze(1),
            img_feat.unsqueeze(1),
            wsi_feat_agg.unsqueeze(1),
        ], dim=1)  # (B, 4, dim)

        fusion_out, transformer_aux_loss = self.fusion_transformer(fusion_input)
        final_emb = fusion_out[:, 0, :]  # (B, dim)

        # --- 分类 ---
        coarse_logits, coarse_boost = self.hierarchical(final_emb)
        moe_logits, classifier_aux_loss = self.moe_classifier(final_emb)
        proto_logits = self.prototypes(final_emb)

        hier_w = torch.sigmoid(self.hier_gate)
        proto_w = torch.sigmoid(self.proto_gate)
        logits = moe_logits + hier_w * coarse_boost + proto_w * proto_logits

        total_aux = transformer_aux_loss + classifier_aux_loss

        if self.training:
            return {
                "logits": logits,
                "coarse_logits": coarse_logits,
                "aux_loss": total_aux,
                "proto_logits": proto_logits,
            }

        return logits


# ============================================================
#  7. 联合损失函数
# ============================================================

class MoEFocalLoss(nn.Module):
    """
    联合损失 = Focal Loss (细分类)
             + lambda_coarse * CE (粗分类)
             + lambda_proto  * CE (原型)
             + lambda_aux    * MoE 辅助损失 (负载均衡 + z-loss)
    """

    def __init__(self, alpha=None, gamma: float = 2.0,
                 label_smoothing: float = 0.1,
                 lambda_coarse: float = 0.3,
                 lambda_proto: float = 0.2,
                 lambda_aux: float = 0.01):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        self.lambda_coarse = lambda_coarse
        self.lambda_proto = lambda_proto
        self.lambda_aux = lambda_aux

        self.register_buffer(
            "fine_to_coarse",
            torch.tensor(FINE_TO_COARSE, dtype=torch.long),
        )

    def forward(self, model_output: dict, targets):
        logits = model_output["logits"]
        coarse_logits = model_output["coarse_logits"]
        proto_logits = model_output["proto_logits"]
        aux_loss = model_output["aux_loss"]

        if self.alpha is not None:
            alpha = self.alpha.to(device=logits.device, dtype=logits.dtype)
        else:
            alpha = None

        ce = F.cross_entropy(
            logits, targets,
            weight=alpha,
            label_smoothing=self.label_smoothing,
            reduction="none",
        )
        pt = torch.exp(-ce)
        focal_loss = ((1 - pt) ** self.gamma * ce).mean()

        coarse_labels = self.fine_to_coarse.to(targets.device)[targets]
        coarse_loss = F.cross_entropy(coarse_logits, coarse_labels)

        proto_loss = F.cross_entropy(proto_logits, targets)

        total = (
            focal_loss
            + self.lambda_coarse * coarse_loss
            + self.lambda_proto * proto_loss
            + self.lambda_aux * aux_loss
        )

        return total


# ============================================================
#  8. EMA (Exponential Moving Average)
# ============================================================

class ModelEMA:
    """
    模型参数指数移动平均。

    验证/测试时使用 EMA 参数, 可以显著稳定少样本类的预测。
    """

    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()

    @torch.no_grad()
    def update(self, model: nn.Module):
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.shadow[name].mul_(self.decay).add_(param.data, alpha=1 - self.decay)

    def apply_shadow(self, model: nn.Module):
        """将 EMA 参数应用到模型 (验证时)"""
        for name, param in model.named_parameters():
            if param.requires_grad and name in self.shadow:
                self.backup[name] = param.data.clone()
                param.data.copy_(self.shadow[name])

    def restore(self, model: nn.Module):
        """恢复原始参数 (验证后)"""
        for name, param in model.named_parameters():
            if name in self.backup:
                param.data.copy_(self.backup[name])
        self.backup = {}
