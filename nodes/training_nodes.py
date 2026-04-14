from .training_backend import list_voxcpm_models
from .training_backend import prepare_dataset
from .training_backend import run_lora_training


class VoxCPM_Dataset_Preparer:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "来源模式": (
                ["长音频", "数据集目录", "批量音频"],
                {"default": "长音频"},
            ),
            "来源路径": ("STRING", {"default": "", "multiline": False}),
            "输出目录": ("STRING", {"default": "", "multiline": False}),
            "训练清单文件名": (
                "STRING",
                {"default": "train.jsonl", "multiline": False},
            ),
            "递归扫描": ("BOOLEAN", {"default": True}),
            "启用参考音频": ("BOOLEAN", {"default": False}),
            "参考音频比例": (
                "FLOAT",
                {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.05},
            ),
            "最短片段秒数": (
                "FLOAT",
                {"default": 2.0, "min": 0.5, "max": 30.0, "step": 0.1},
            ),
            "最长片段秒数": (
                "FLOAT",
                {"default": 10.0, "min": 1.0, "max": 60.0, "step": 0.1},
            ),
            "目标片段秒数": (
                "FLOAT",
                {"default": 6.0, "min": 1.0, "max": 30.0, "step": 0.1},
            ),
            "前后保留静音毫秒": (
                "INT",
                {"default": 150, "min": 0, "max": 2000, "step": 10},
            ),
            "静音阈值分贝": (
                "FLOAT",
                {"default": -40.0, "min": -80.0, "max": -5.0, "step": 1.0},
            ),
            "ui_language": (["中文", "English"], {"default": "中文"}),
        }

        return {
            "required": required,
            "optional": {
                "长音频": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("train_jsonl", "dataset_dir", "info")
    FUNCTION = "prepare_dataset_node"
    CATEGORY = "VoxCPM/Training"

    def prepare_dataset_node(self, **kwargs):
        manifest_path, dataset_dir, info = prepare_dataset(
            source_mode=kwargs["来源模式"],
            source_path=kwargs.get("来源路径"),
            output_dir=kwargs.get("输出目录"),
            manifest_name=kwargs.get("训练清单文件名"),
            recursive=kwargs.get("递归扫描", True),
            min_seconds=kwargs.get("最短片段秒数", 2.0),
            max_seconds=kwargs.get("最长片段秒数", 10.0),
            target_seconds=kwargs.get("目标片段秒数", 6.0),
            keep_silence_ms=kwargs.get("前后保留静音毫秒", 150),
            silence_threshold_db=kwargs.get("静音阈值分贝", -40.0),
            enable_ref_audio=kwargs.get("启用参考音频", False),
            ref_audio_ratio=kwargs.get("参考音频比例", 0.4),
            audio_input=kwargs.get("长音频"),
        )
        print(f"[VoxCPM][数据集] {info}")
        return (manifest_path, dataset_dir, info)


class VoxCPM_Lora_Trainer:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "模型": (list_voxcpm_models(),),
                "训练清单": ("STRING", {"default": "", "multiline": False}),
                "输出名称": ("STRING", {"default": "my_voxcpm_lora", "multiline": False}),
                "总步数": ("INT", {"default": 1000, "min": 10, "max": 200000, "step": 10}),
                "保存间隔": ("INT", {"default": 200, "min": 10, "max": 50000, "step": 10}),
                "学习率": ("FLOAT", {"default": 1e-4, "min": 1e-6, "max": 1e-2, "step": 1e-5}),
                "继续训练": ("BOOLEAN", {"default": False}),
                "show_advanced": ("BOOLEAN", {"default": False}),
                "ui_language": (["中文", "English"], {"default": "中文"}),
                "LoRA秩": ("INT", {"default": 32, "min": 4, "max": 128, "step": 4}),
                "LoRA缩放": ("INT", {"default": 32, "min": 1, "max": 128, "step": 1}),
                "LoRA丢弃": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.05}),
                "预热步数": ("INT", {"default": 100, "min": 0, "max": 5000, "step": 10}),
                "梯度累积": ("INT", {"default": 1, "min": 1, "max": 64, "step": 1}),
                "批量大小": ("INT", {"default": 1, "min": 1, "max": 16, "step": 1}),
                "最大批量Token": ("INT", {"default": 8192, "min": 0, "max": 65536, "step": 256}),
                "权重衰减": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 0.1, "step": 0.001}),
                "数据加载线程": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "启用LM LoRA": ("BOOLEAN", {"default": True}),
                "启用DiT LoRA": ("BOOLEAN", {"default": True}),
                "启用投影层LoRA": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING", "STRING")
    RETURN_NAMES = ("lora_dir", "latest_weights", "info")
    FUNCTION = "train_lora"
    CATEGORY = "VoxCPM/Training"

    def train_lora(self, **kwargs):
        output_dir, latest_weights, info = run_lora_training(
            model_name=kwargs["模型"],
            dataset_path=kwargs["训练清单"],
            output_name=kwargs["输出名称"],
            max_steps=kwargs["总步数"],
            save_every_steps=kwargs["保存间隔"],
            learning_rate=kwargs["学习率"],
            lora_rank=kwargs["LoRA秩"],
            lora_alpha=kwargs["LoRA缩放"],
            lora_dropout=kwargs["LoRA丢弃"],
            warmup_steps=kwargs["预热步数"],
            grad_accum_steps=kwargs["梯度累积"],
            batch_size=kwargs["批量大小"],
            max_batch_tokens=kwargs["最大批量Token"],
            weight_decay=kwargs["权重衰减"],
            num_workers=kwargs["数据加载线程"],
            enable_lm_lora=kwargs["启用LM LoRA"],
            enable_dit_lora=kwargs["启用DiT LoRA"],
            enable_proj_lora=kwargs["启用投影层LoRA"],
            resume_training=kwargs["继续训练"],
        )
        print(f"[VoxCPM][训练] {info}")
        return (output_dir, latest_weights, info)
