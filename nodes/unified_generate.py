import gc
import json
import os
import re

import comfy.model_management as model_management
import comfy.utils
import folder_paths
import numpy as np
import torch
import torchaudio

from .audio_utils import _denoise_audio, _recognize_audio, _save_audio_to_temp

VOXCPM_MODEL_TYPE = "voxcpm"
LORA_MODEL_TYPE = "voxcpm_lora"

ARCH_V2 = "voxcpm2"
ARCH_LEGACY = "voxcpm"

V2_MODES_ZH = ("声音设计", "极致克隆", "可控克隆", "多人配音")
LEGACY_MODES_ZH = ("常规克隆", "多人配音")

MODE_ALIASES = {
    "Voice Design": "声音设计",
    "Ultimate Cloning": "极致克隆",
    "Controllable Cloning": "可控克隆",
    "Regular Cloning": "常规克隆",
    "Multi-Speaker Dubbing": "多人配音",
    "Multi-Speaker Dialog": "多人配音",
    "多角色对话": "多人配音",
}

folder_paths.add_model_folder_path(
    VOXCPM_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE),
)
folder_paths.add_model_folder_path(
    LORA_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE, "loras"),
)

_cached_pipe = None
_cached_config_hash = None


def _normalize_architecture(architecture):
    return ARCH_V2 if architecture == ARCH_V2 else ARCH_LEGACY


def _normalize_work_mode(work_mode):
    return MODE_ALIASES.get(work_mode, work_mode)


def _resolve_effective_work_mode(work_mode, architecture):
    mode = _normalize_work_mode(work_mode)
    is_v2 = _normalize_architecture(architecture) == ARCH_V2

    if is_v2:
        if mode == "常规克隆":
            return "极致克隆"
        return mode if mode in V2_MODES_ZH else "声音设计"

    if mode in ("声音设计", "极致克隆", "可控克隆"):
        return "常规克隆"
    return mode if mode in LEGACY_MODES_ZH else "常规克隆"


def _mode_requires_reference(work_mode):
    return work_mode in {"极致克隆", "可控克隆", "常规克隆"}


def _mode_requires_prompt_text(work_mode):
    return work_mode in {"极致克隆", "常规克隆"}


def _mode_uses_control(work_mode, architecture):
    return _normalize_architecture(architecture) == ARCH_V2 and work_mode in {"声音设计", "可控克隆"}


def _list_model_dirs():
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    seen, results = set(), []
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name in seen:
                continue
            full = os.path.join(base, name)
            if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
                seen.add(name)
                results.append(name)
    if not results:
        return ["None"]
    if "VoxCPM2" in results:
        results.remove("VoxCPM2")
        results.insert(0, "VoxCPM2")
    return results


def _list_lora_files():
    base_dirs = folder_paths.get_folder_paths(LORA_MODEL_TYPE)
    seen, results = set(), ["None"]
    for base in base_dirs:
        if not os.path.isdir(base):
            continue
        for name in sorted(os.listdir(base)):
            if name in seen:
                continue
            full = os.path.join(base, name)
            is_weight_file = os.path.isfile(full) and name.endswith((".pth", ".ckpt", ".safetensors"))
            if is_weight_file:
                seen.add(name)
                results.append(name)
    return results


def _resolve_model_path(model_name):
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    for base in base_dirs:
        full = os.path.join(base, model_name)
        if os.path.isdir(full) and os.path.isfile(os.path.join(full, "config.json")):
            return full
    raise FileNotFoundError(f"找不到模型: {model_name}。请确保已放入 models/voxcpm/ 目录。")


def _resolve_lora_path(lora_name):
    if not lora_name or lora_name == "None":
        return None
    base_dirs = folder_paths.get_folder_paths(LORA_MODEL_TYPE)
    for base in base_dirs:
        full = os.path.join(base, lora_name)
        if os.path.isfile(full) or os.path.isdir(full):
            return full
    raise FileNotFoundError(f"找不到 LoRA: {lora_name}。请确保已放入 models/voxcpm/loras/ 目录。")


def _parse_lora_metadata_value(value):
    if value is None:
        return None
    if isinstance(value, (bool, int, float)):
        return value

    text = str(value).strip()
    if not text:
        return None

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False

    try:
        return json.loads(text)
    except Exception:
        return text


def _get_metadata_by_suffix(metadata, suffix):
    if not isinstance(metadata, dict):
        return None
    for key, value in metadata.items():
        if str(key).endswith(suffix):
            return value
    return None


def _merge_lora_info(base_info, override_info):
    merged = {}
    if isinstance(base_info, dict):
        merged.update(base_info)
    if not isinstance(override_info, dict):
        return merged

    for key, value in override_info.items():
        if key == "lora_config" and isinstance(value, dict):
            existing = merged.get("lora_config")
            merged["lora_config"] = dict(existing) if isinstance(existing, dict) else {}
            merged["lora_config"].update(value)
        else:
            merged[key] = value
    return merged


def _read_lora_info_from_safetensors(lora_path):
    if not lora_path or not os.path.isfile(lora_path) or not str(lora_path).lower().endswith(".safetensors"):
        return {}

    try:
        from safetensors import safe_open
    except ImportError:
        return {}

    try:
        with safe_open(lora_path, framework="pt", device="cpu") as f:
            metadata = f.metadata() or {}
    except Exception:
        return {}

    raw_info = (
        metadata.get("voxcpm.lora_info")
        or metadata.get("voxcpm_lora_info")
        or _get_metadata_by_suffix(metadata, ".lora_info")
        or metadata.get("lora_info")
    )
    if raw_info:
        try:
            parsed = json.loads(raw_info)
            if isinstance(parsed, dict):
                return parsed
        except Exception:
            pass

    lora_info = {}
    base_model = metadata.get("voxcpm.base_model") or _get_metadata_by_suffix(metadata, ".base_model") or metadata.get("base_model")
    if base_model:
        lora_info["base_model"] = str(base_model)

    base_model_name = metadata.get("voxcpm.base_model_name") or _get_metadata_by_suffix(metadata, ".base_model_name") or metadata.get("base_model_name")
    if base_model_name:
        lora_info["base_model_name"] = str(base_model_name)

    architecture = metadata.get("voxcpm.architecture") or _get_metadata_by_suffix(metadata, ".architecture") or metadata.get("architecture")
    if architecture:
        lora_info["architecture"] = str(architecture)

    config_dict = {}
    for key in ("r", "alpha", "dropout", "enable_lm", "enable_dit", "enable_proj"):
        raw_value = metadata.get(f"voxcpm.{key}") or _get_metadata_by_suffix(metadata, f".{key}") or metadata.get(key)
        parsed_value = _parse_lora_metadata_value(raw_value)
        if parsed_value is not None:
            config_dict[key] = parsed_value

    if config_dict:
        lora_info["lora_config"] = config_dict
    return lora_info


def _read_lora_info_from_json(config_path):
    if not config_path or not os.path.isfile(config_path):
        return {}
    try:
        with open(config_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"LoRA 配置文件读取失败: {config_path}，{e}")


def _read_lora_info(lora_path):
    if not lora_path:
        return {}

    lora_info = {}
    if os.path.isfile(lora_path):
        metadata_info = _read_lora_info_from_safetensors(lora_path)
        if metadata_info:
            lora_info = _merge_lora_info(lora_info, metadata_info)

    candidates = []
    if os.path.isdir(lora_path):
        candidates.append(os.path.join(lora_path, "lora_config.json"))
    else:
        parent_dir = os.path.dirname(lora_path)
        stem_name = os.path.splitext(os.path.basename(lora_path))[0]
        if parent_dir and stem_name:
            candidates.append(os.path.join(parent_dir, f"{stem_name}.lora_config.json"))
            candidates.append(os.path.join(parent_dir, f"{stem_name}.json"))
            candidates.append(os.path.join(parent_dir, stem_name, "lora_config.json"))
        if parent_dir:
            candidates.append(os.path.join(parent_dir, "lora_config.json"))

    for config_path in candidates:
        if not os.path.isfile(config_path):
            continue
        config_info = _read_lora_info_from_json(config_path)
        lora_info = _merge_lora_info(config_info, lora_info)
        break
    return lora_info


def _extract_lora_config_dict(lora_info):
    if not isinstance(lora_info, dict):
        return {}

    raw_config = lora_info.get("lora_config")
    if not isinstance(raw_config, dict):
        raw_config = lora_info

    allowed_keys = {
        "enable_lm",
        "enable_dit",
        "enable_proj",
        "r",
        "alpha",
        "dropout",
        "target_modules_lm",
        "target_modules_dit",
        "target_proj_modules",
    }
    return {key: value for key, value in raw_config.items() if key in allowed_keys}


def _extract_lora_base_model_name(lora_info):
    if not isinstance(lora_info, dict):
        return None
    base_model_name = str(lora_info.get("base_model_name", "") or "").strip()
    if base_model_name:
        return base_model_name
    base_model = str(lora_info.get("base_model", "") or "").strip()
    if not base_model:
        return None
    normalized = base_model.rstrip("/\\")
    return os.path.basename(normalized) if normalized else None


def _build_lora_config_for_model(lora_path, architecture):
    lora_info = _read_lora_info(lora_path)
    config_dict = _extract_lora_config_dict(lora_info)
    if not config_dict:
        return None, lora_info

    if architecture == ARCH_V2:
        from voxcpm.model.voxcpm2 import LoRAConfig as LoraConfigClass
    else:
        from voxcpm.model.voxcpm import LoRAConfig as LoraConfigClass

    try:
        return LoraConfigClass(**config_dict), lora_info
    except Exception as e:
        raise RuntimeError(f"LoRA 配置与当前模型不兼容，无法创建加载配置：{e}")


def _validate_lora_compatibility(model_name, architecture, lora_info):
    if not lora_info:
        return

    base_model_name = _extract_lora_base_model_name(lora_info)
    if base_model_name and base_model_name != model_name:
        raise ValueError(
            f"当前 LoRA 是基于 {base_model_name} 训练的，不能加载到 {model_name}。请切换为匹配的基座模型。"
        )

    lora_architecture = str(lora_info.get("architecture", "") or "").strip().lower()
    if lora_architecture:
        lora_architecture = _normalize_architecture(lora_architecture)
    else:
        base_model_path = str(lora_info.get("base_model", "") or "").strip()
        if not base_model_path:
            return

        config_path = os.path.join(base_model_path, "config.json")
        if not os.path.isfile(config_path):
            return

        try:
            with open(config_path, "r", encoding="utf-8") as f:
                config = json.load(f)
            lora_architecture = _normalize_architecture(str(config.get("architecture", ARCH_LEGACY)).lower())
        except Exception:
            return

    if lora_architecture != architecture:
        raise ValueError(
            f"当前 LoRA 的架构是 {lora_architecture}，而所选模型 {model_name} 的架构是 {architecture}，两者不兼容。"
        )


def _read_model_architecture(model_name):
    model_path = _resolve_model_path(model_name)
    with open(os.path.join(model_path, "config.json"), "r", encoding="utf-8") as f:
        config = json.load(f)
    return _normalize_architecture(config.get("architecture", ARCH_LEGACY).lower())


def _build_model_profiles_json():
    profiles = {}
    for model_name in _list_model_dirs():
        if model_name == "None":
            continue
        try:
            profiles[model_name] = {
                "architecture": _read_model_architecture(model_name),
            }
        except Exception:
            profiles[model_name] = {
                "architecture": ARCH_LEGACY,
            }
    return json.dumps(profiles, ensure_ascii=False)


def load_voxcpm_model(model_name, lora_name="None", optimize=False):
    global _cached_pipe, _cached_config_hash
    config_hash = hash((model_name, optimize, lora_name))

    if _cached_pipe is not None and config_hash == _cached_config_hash:
        return _cached_pipe

    if _cached_pipe is not None:
        del _cached_pipe
        _cached_pipe, _cached_config_hash = None, None
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    from voxcpm import VoxCPM

    model_path = _resolve_model_path(model_name)
    lora_path = _resolve_lora_path(lora_name)
    architecture = _read_model_architecture(model_name)
    lora_config = None

    if lora_path:
        lora_config, lora_info = _build_lora_config_for_model(lora_path, architecture)
        _validate_lora_compatibility(model_name, architecture, lora_info)
        if lora_config is not None:
            config_dict = _extract_lora_config_dict(lora_info)
            print(
                "[VoxCPM] 读取 LoRA 配置: "
                f"r={config_dict.get('r')}, alpha={config_dict.get('alpha')}, "
                f"dropout={config_dict.get('dropout')}, "
                f"enable_lm={config_dict.get('enable_lm')}, "
                f"enable_dit={config_dict.get('enable_dit')}, "
                f"enable_proj={config_dict.get('enable_proj')}"
            )

    print(f"[VoxCPM] 正在加载模型: {model_name}" + (f" (LoRA: {lora_name})" if lora_name != "None" else ""))
    model = VoxCPM(
        voxcpm_model_path=model_path,
        zipenhancer_model_path=None,
        enable_denoiser=False,
        optimize=optimize,
        lora_config=lora_config,
        lora_weights_path=lora_path,
    )

    _cached_pipe = {
        "model": model,
        "sample_rate": model.tts_model.sample_rate,
        "architecture": architecture,
        "model_path": model_path,
    }
    _cached_config_hash = config_hash
    return _cached_pipe


def force_unload_model():
    global _cached_pipe, _cached_config_hash
    if _cached_pipe is not None:
        del _cached_pipe
        _cached_pipe, _cached_config_hash = None, None
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    model_management.soft_empty_cache()
    print("[VoxCPM] 模型已强制从显存中卸载以释放资源。")


def _normalize_loudness_tensor(waveform, sample_rate, target_lufs=-20.0):
    try:
        if waveform.dim() == 3:
            waveform = waveform.squeeze(0)
        loudness = torchaudio.functional.loudness(waveform, sample_rate)
        loudness_val = loudness.mean().item()
        normalized_audio = torchaudio.functional.gain(waveform, target_lufs - loudness_val)
        return normalized_audio.unsqueeze(0)
    except Exception as e:
        print(f"[VoxCPM] 响度标准化失败: {e}，将输出原始音频。")
        return waveform if waveform.dim() == 3 else waveform.unsqueeze(0)


class VoxCPM_Unified_Generator:
    @classmethod
    def INPUT_TYPES(cls):
        req = {
            "model_name": (_list_model_dirs(),),
            "work_mode": (
                [
                    "声音设计",
                    "极致克隆",
                    "可控克隆",
                    "常规克隆",
                    "多人配音",
                    "Voice Design",
                    "Ultimate Cloning",
                    "Controllable Cloning",
                    "Regular Cloning",
                    "Multi-Speaker Dubbing",
                    "Multi-Speaker Dialog",
                    "多角色对话",
                ],
                {"default": "声音设计"},
            ),
            "control_instruction": ("STRING", {"default": "", "multiline": True}),
        }

        req.update(
            {
                "target_text": ("STRING", {"default": "欢迎使用 VoxCPM 全能语音节点。", "multiline": True}),
                # 【修改点】：将 speaker_count 移入 required 区域，物理层面上紧邻目标文本
                "speaker_count": ("INT", {"default": 2, "min": 1, "max": 10, "step": 1}),
                "show_advanced": ("BOOLEAN", {"default": False}),
                "cfg_value": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 5.0, "step": 0.1}),
                "inference_steps": ("INT", {"default": 10, "min": 1, "max": 50, "step": 1}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            }
        )

        return {
            "required": req,
            "optional": {
                "ui_language": (["中文", "English"], {"default": "中文"}),
                "lora_name": (_list_lora_files(),),
                "force_offload": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "生成完毕后强制卸载模型以释放显存（适用于显存紧张的用户）"},
                ),
                "auto_asr": ("BOOLEAN", {"default": True}),
                "reference_text": ("STRING", {"default": "", "multiline": True}),
                "denoise_reference": ("BOOLEAN", {"default": False}),
                "normalize_text": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "自动规范化数字、日期及缩写（基于 wetext）"},
                ),
                "normalize_loudness": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "将最终输出音量统一标准化至 -20 LUFS"},
                ),
                "model_profiles_json": ("STRING", {"default": _build_model_profiles_json(), "multiline": False}),
            },
        }

    RETURN_TYPES = ("AUDIO",)
    RETURN_NAMES = ("audio",)
    FUNCTION = "generate"
    CATEGORY = "VoxCPM"

    def _build_generate_kwargs(self, text, control, ref_audio_path, ref_text, work_mode, architecture, normalize_text=True):
        effective_mode = _resolve_effective_work_mode(work_mode, architecture)
        final_text = f"({control}){text}" if _mode_uses_control(effective_mode, architecture) and control else text

        generate_kwargs = {
            "text": final_text,
            "cfg_value": float(self.cfg_value),
            "inference_timesteps": int(self.inference_steps),
            "normalize": normalize_text,
            "denoise": False,
            "max_len": 4096,
            "retry_badcase": True,
        }

        if effective_mode == "声音设计":
            return generate_kwargs

        if effective_mode == "可控克隆":
            if ref_audio_path:
                generate_kwargs["reference_wav_path"] = ref_audio_path
            return generate_kwargs

        if effective_mode == "极致克隆":
            if not ref_audio_path:
                raise ValueError("极致克隆需要连接参考音频。")
            if not ref_text:
                raise ValueError("极致克隆需要参考文本，请开启自动识别或填写参考文本。")
            generate_kwargs["prompt_wav_path"] = ref_audio_path
            generate_kwargs["prompt_text"] = ref_text
            generate_kwargs["reference_wav_path"] = ref_audio_path
            return generate_kwargs

        if effective_mode == "常规克隆":
            if not ref_audio_path:
                return generate_kwargs
            if not ref_text:
                raise ValueError("常规克隆需要参考文本，请开启自动识别或填写参考文本。")
            generate_kwargs["prompt_wav_path"] = ref_audio_path
            generate_kwargs["prompt_text"] = ref_text
            return generate_kwargs

        raise ValueError(f"不支持的模式: {effective_mode}")


    def _generate_single(self, text, control, ref_audio_path, ref_text, work_mode, architecture, pbar, step_increment, normalize_text=True):
        voxcpm_model = self.current_model["model"]
        generate_kwargs = self._build_generate_kwargs(
            text=text,
            control=control,
            ref_audio_path=ref_audio_path,
            ref_text=ref_text,
            work_mode=work_mode,
            architecture=architecture,
            normalize_text=normalize_text,
        )
        wav_np = voxcpm_model.generate(**generate_kwargs)
        if pbar:
            pbar.update(step_increment)
        return wav_np

    def _resolve_reference_text(self, work_mode, ref_path, auto_asr, reference_text):
        if not ref_path or not _mode_requires_prompt_text(work_mode):
            return ""
        if auto_asr:
            return _recognize_audio(ref_path).strip()
        return (reference_text or "").strip()

    def generate(self, model_name, work_mode, target_text, cfg_value, inference_steps, seed, **kwargs):
        lora_name = kwargs.get("lora_name", "None")
        force_offload = kwargs.get("force_offload", False)
        do_normalize_text = kwargs.get("normalize_text", True)

        self.current_model = load_voxcpm_model(model_name, lora_name=lora_name, optimize=False)
        self.cfg_value = cfg_value
        self.inference_steps = inference_steps

        sample_rate = self.current_model["sample_rate"]
        architecture = self.current_model["architecture"]
        effective_mode = _resolve_effective_work_mode(work_mode, architecture)
        target_text = (target_text or "").strip()

        if not target_text:
            raise ValueError("文本不能为空！")

        temp_files = []

        def get_clean_ref(audio_key):
            audio_obj = kwargs.get(audio_key)
            if not audio_obj:
                return None

            waveform = audio_obj["waveform"]
            duration = waveform.shape[-1] / audio_obj["sample_rate"]
            if duration > 50.0:
                raise ValueError(f"参考音频时长为 {duration:.1f} 秒。为了防止显存溢出，请裁剪至 50 秒以内！")

            path = _save_audio_to_temp(audio_obj)
            temp_files.append(path)
            if kwargs.get("denoise_reference", False):
                denoised = _denoise_audio(path)
                temp_files.append(denoised)
                path = denoised
            return path

        try:
            if effective_mode != "多人配音":
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                ref_path = get_clean_ref("reference_audio") if _mode_requires_reference(effective_mode) else None
                control = kwargs.get("control_instruction", "").strip() if _mode_uses_control(effective_mode, architecture) else ""
                ref_text = self._resolve_reference_text(
                    effective_mode,
                    ref_path,
                    kwargs.get("auto_asr", True),
                    kwargs.get("reference_text", ""),
                )

                pbar = comfy.utils.ProgressBar(int(inference_steps) + 1)
                wav_np = self._generate_single(
                    target_text,
                    control,
                    ref_path,
                    ref_text,
                    effective_mode,
                    architecture,
                    pbar,
                    int(inference_steps),
                    do_normalize_text,
                )
                wav_tensor = torch.from_numpy(wav_np).float().unsqueeze(0)

            else:
                pattern = re.compile(r"\[(spk\d*|ctrl)\]", re.IGNORECASE)
                parts = pattern.split(target_text)
                segments = []
                current_ctrl = ""

                if parts[0].strip():
                    segments.append((None, parts[0].strip(), ""))

                for i in range(1, len(parts), 2):
                    tag = parts[i].lower()
                    content = parts[i + 1].strip()

                    if tag == "ctrl":
                        current_ctrl = content
                    elif tag.startswith("spk"):
                        spk_idx = None if tag == "spk" else int(tag[3:])
                        if content:
                            segments.append((spk_idx, content, current_ctrl))
                        current_ctrl = ""

                if not segments:
                    raise ValueError("未找到有效剧本，请参考格式：\n[ctrl]激动\n[spk1]文本\n[spk]龙套文本")

                total_steps = len(segments) * (int(inference_steps) + 1)
                pbar = comfy.utils.ProgressBar(total_steps)

                spk_cache = {}
                generated_wavs = []
                supports_control = _normalize_architecture(architecture) == ARCH_V2

                for i, (spk_idx, text, seg_ctrl) in enumerate(segments):
                    sentence_seed = seed + i
                    torch.manual_seed(sentence_seed)
                    if torch.cuda.is_available():
                        torch.cuda.manual_seed_all(sentence_seed)

                    seg_control = seg_ctrl.strip() if supports_control else ""
                    ref_text = ""

                    if spk_idx is None:
                        ref_path = None
                        seg_mode = "声音设计"
                    else:
                        if spk_idx not in spk_cache:
                            ref_path = get_clean_ref(f"audio_{spk_idx}")
                            spk_cache[spk_idx] = {
                                "ref_path": ref_path,
                                "ref_text": None,
                            }
                            if not ref_path:
                                print(f"[VoxCPM] 角色 [spk{spk_idx}] 未连接参考音频，将自动退回“声音设计”模式。")

                        cache = spk_cache[spk_idx]
                        ref_path = cache["ref_path"]

                        if not ref_path:
                            seg_mode = "声音设计"
                        elif _normalize_architecture(architecture) == ARCH_V2:
                            if seg_control:
                                seg_mode = "可控克隆"
                            else:
                                seg_mode = "极致克隆"
                                if cache["ref_text"] is None:
                                    cache["ref_text"] = _recognize_audio(ref_path).strip()
                                ref_text = cache["ref_text"] or ""
                        else:
                            seg_mode = "常规克隆"
                            seg_control = ""
                            if cache["ref_text"] is None:
                                cache["ref_text"] = _recognize_audio(ref_path).strip()
                            ref_text = cache["ref_text"] or ""

                    wav_np = self._generate_single(
                        text,
                        seg_control,
                        ref_path,
                        ref_text,
                        seg_mode,
                        architecture,
                        pbar,
                        int(inference_steps),
                        do_normalize_text,
                    )
                    generated_wavs.append(wav_np)

                combined = np.concatenate(generated_wavs, axis=-1)
                peak = np.max(np.abs(combined))
                if peak > 0.99:
                    combined = combined * (0.99 / peak)
                wav_tensor = torch.from_numpy(combined).float().unsqueeze(0)

            if kwargs.get("normalize_loudness", True):
                wav_tensor = _normalize_loudness_tensor(wav_tensor, sample_rate)
            else:
                wav_tensor = wav_tensor.unsqueeze(0)

            return ({"waveform": wav_tensor, "sample_rate": sample_rate},)

        finally:
            for tmp in temp_files:
                if tmp and os.path.exists(tmp):
                    try:
                        os.unlink(tmp)
                    except OSError:
                        pass

            if force_offload:
                self.current_model = None
                gc.collect()
                force_unload_model()