

class Config:
    # ================= 路径配置 =================
    # 数据集根目录 
    RAW_DATA_ROOT = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets-8csl-70_30-new/train"
    #RAW_DATA_ROOT = "/media/codingma/LLM/data-1005"

    DATA_INDEX_PATH = "/media/codingma/LLM/lcx/Medical_Info_Classification/datasets-8csl-70_30-new/index.csv"
    OVERWRITE_SWI_FEATURES = False

    # 模型路径
    VIRCHOW2_MODEL_ID = "local-dir:/media/codingma/LLM/lcx/Virchow2" # HuggingFace ID
    QWEN_MODEL_PATH = "/media/codingma/LLM/lcx/Qwen3-Embedding-0.6B" 

    # ================= 分类模式 =================
    # True  = 多类别多标签 (7类, OSF+OLK 拆为 OLK+OSF 两个标签)
    # False = 多类别单标签 (8类, OSF+OLK 作为独立类别)
    MULTI_LABEL = False
    
    NUM_CLASSES = 3 # 7为多标签，8为单标签，3为大类

    # False = 多类别单标签 (8类, OSF+OLK 作为独立类别)
    # MULTI_LABEL = True
    
    # NUM_CLASSES = 7 # 7为多标签，8为单标签，3为大类
    
    # 原始数据集的类别映射
    RAW_CLASS_MAP = {
        "OLK": 0,
        "OLP": 1,
        "OSCC": 2,
        "OSF": 3,
        "OSF+OLK": 4,
        "SCP": 5,
        "Mucocele": 6,
        "Fibroma": 7,
    }
    
    # 多标签模式：模型输出 7 类
    MULTI_LABEL_CLASS_NAMES = [
        "OLK",      # 0
        "OLP",      # 1
        "OSCC",     # 2
        "OSF",      # 3
        "SCP",      # 4
        "Mucocele", # 5
        "Fibroma"   # 6
    ]

    # 单标签模式：模型输出 8 类 (与 RAW_CLASS_MAP 顺序一致)
    SINGLE_LABEL_CLASS_NAMES = [
        "OLK",      # 0
        "OLP",      # 1
        "OSCC",     # 2
        "OSF",      # 3
        "OSF+OLK",  # 4
        "SCP",      # 5
        "Mucocele", # 6
        "Fibroma"   # 7
    ]

    # 大类模式，模型输出 3 类
    MAJOR_CLASS_NAMES = [
        "OSCC",      # 0
        "OPMDs",      # 1
        "Benign",      # 2
    ]
    # 根据模式自动选择类别名列表
    if NUM_CLASSES == 3:
        TARGET_CLASS_NAMES = MAJOR_CLASS_NAMES
    elif NUM_CLASSES == 7:
        TARGET_CLASS_NAMES = MULTI_LABEL_CLASS_NAMES
    elif NUM_CLASSES == 8:
        TARGET_CLASS_NAMES = SINGLE_LABEL_CLASS_NAMES
    else:
        raise ValueError(f"Invalid NUM_CLASSES: {NUM_CLASSES}")

    MAJOR_CLASS_MAP = {
        "OSCC": 0,
        "OPMDs": 1,
        "Benign": 2,
    }

    # ---------- 3 分类 + 8 类目录结构 ----------
    # 默认仍使用 8 类文件夹名与 RAW_CLASS_MAP（index.csv 中 label 为 0–7），
    # 训练/验证时在 dataset 中映射到 3 大类；无需另建 OSCC/OPMDs/Benign 目录。
    # 若仍使用已整理的三类文件夹（OPMDs/OSCC/Benign），将此项改为 True，并令 MAP_RAW_LABEL_TO_MAJOR_3CLS = False。
    THREE_CLS_COARSE_FOLDERS = False

    # CSV / Dataset 的 label 为 8 类细 id (0–7) 时，训练目标映射为 3 大类 id：
    # 0 OSCC, 1 OPMDs (OLK/OLP/OSF/OSF+OLK), 2 Benign (SCP/Mucocele/Fibroma)
    # 键与 RAW_CLASS_MAP 中细类 id 一致
    FINE_8CLS_TO_MAJOR_3CLS = {
        0: 1,  # OLK
        1: 1,  # OLP
        2: 0,  # OSCC
        3: 1,  # OSF
        4: 1,  # OSF+OLK（口语 OLK&OSF）
        5: 2,  # SCP
        6: 2,  # Mucocele
        7: 2,  # Fibroma
    }

    # True：index 等为细类 0–7，训练时按 FINE_8CLS_TO_MAJOR_3CLS 映射
    # False：已为三大类 id 0–2（如粗粒度文件夹 + 对应 CSV）
    MAP_RAW_LABEL_TO_MAJOR_3CLS = True

    # 与 "OSF+OLK" 同义的一类文件夹命名（若数据中存在）
    FINE_8CLS_FOLDER_ALIASES = {
        "OLK&OSF": 4,  # 等同于 OSF+OLK 的细类 id
    }

    if NUM_CLASSES == 3:
        if THREE_CLS_COARSE_FOLDERS:
            CLASS_MAP = MAJOR_CLASS_MAP
        else:
            CLASS_MAP = {**RAW_CLASS_MAP, **FINE_8CLS_FOLDER_ALIASES}
    else:
        CLASS_MAP = RAW_CLASS_MAP
    
    # WSI 处理参数
    PATCH_SIZE = 224       # Virchow2 固定尺寸
    TILE_LEVEL = 0         # 40x / 20x 最高倍率
    BATCH_SIZE_WSI = 256    # 特征提取时的 Batch Size
    NUM_WORKERS = 8        # DataLoader workers
    BG_THRESHOLD = 220     # 去除背景的阈值
    SAVE_COORDS = False
    USE_FAST_VERSION = True

    # ================= 训练参数 =================
    # 权重保存目录
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints-txt_only"
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_80_20_multi_label_20260221"
    #CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label"
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_single_label_3cls_20262027"
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_95_5_multi_label_8cls_202620304"
    
    #CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/checkpoints_70_30_multi_label_8cls_202620302"
    
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/get_log_checkpoints_70_30_multi_label_7cls_202620408"
    # CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/get_log_checkpoints_70_30_single_label_3cls_202620406"
    CHECKPOINT_DIR = "/media/codingma/LLM/lcx/Medical_Info_Classification/revisied_ckpt_20260602_3cls_v3"


    
    # [模型容量配置] 7类多标签模型配置
    FUSION_DIM = 1280       # 512 -> 768 (增大维度)
    FUSION_LAYERS = 12      # 2 -> 6 (增加深度)
    FUSION_HEADS = 16      # 768 / 12 = 64 (必须能整除 FUSION_DIM)
    FUSION_DROPOUT = 0.1

    # [模型容量配置] 3类大类模型配置
    # FUSION_DIM = 1024       # 512 -> 768 (增大维度)
    # FUSION_LAYERS = 6      # 2 -> 6 (增加深度)
    # FUSION_HEADS = 16      # 768 / 12 = 64 (必须能整除 FUSION_DIM)
    # FUSION_DROPOUT = 0.1

    WSI_INPUT_DIM = 2560
    MAX_TEXT_LEN = 1024
    FREEZE_TEXT_MODEL = True
    
    BATCH_SIZE = 4
    EPOCHS = 10
    MAX_STEPS = 700 # 800 for 3-class, 1400 for 7-class
    LEARNING_RATE = 1e-5
    WEIGHT_DECAY = 0.01
    
    # [新增策略]
    DATA_EXPAND_FACTOR = 10  # 虚拟扩充数据集大小 
    GRAD_ACCUM_STEPS = 4   # 梯度累积步数 
    
    # [验证集划分]
    TRAIN_VAL_SPLIT = 0.8   # 训练集比例
    SEED = 42               # 随机种子

    DEVICE = "cuda"
    USE_AMP = True
    EVAL_EVERY_STEPS = 500


    # [模态开关]
    USE_TXT = True
    USE_IMG = True
    USE_SVS = True

    # ================= 学习曲线参数 =================
    # 训练集规模可用“比例(<=1)”或“绝对数量(>1)”
    LEARNING_CURVE_ENABLE = False
    LEARNING_CURVE_TRAIN_SIZES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 1.0]
    LEARNING_CURVE_REPEATS = 1
    LEARNING_CURVE_BASE_SEED = 42


def map_label_for_3cls_training(raw_label: int) -> int:
    """
    单标签训练时：若 NUM_CLASSES==3 且 MAP_RAW_LABEL_TO_MAJOR_3CLS，
    将 8 类细 id 转为 3 大类 id；否则原样返回。
    """
    x = int(raw_label)
    if Config.NUM_CLASSES != 3:
        return x
    if not getattr(Config, "MAP_RAW_LABEL_TO_MAJOR_3CLS", True):
        return x
    return int(Config.FINE_8CLS_TO_MAJOR_3CLS.get(x, x))
