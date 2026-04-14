import contextlib
import gc
import importlib
import json
import math
import os
import re
import shutil
from datetime import datetime
from pathlib import Path

import comfy.model_management as model_management
import comfy.utils
import folder_paths
import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from .audio_utils import _recognize_audio


VOXCPM_MODEL_TYPE = "voxcpm"
LORA_MODEL_TYPE = "voxcpm_lora"
DATASET_SUBDIR_NAME = "datasets"
DEFAULT_MANIFEST_NAME = "train.jsonl"
SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".flac", ".mp3", ".ogg", ".m4a")


folder_paths.add_model_folder_path(
    VOXCPM_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE),
)
folder_paths.add_model_folder_path(
    LORA_MODEL_TYPE,
    os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE, "loras"),
)


def list_voxcpm_models():
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    seen = set()
    results = []
    for base_dir in base_dirs:
        if not os.path.isdir(base_dir):
            continue
        for name in sorted(os.listdir(base_dir)):
            if name in seen:
                continue
            full_path = os.path.join(base_dir, name)
            if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "config.json")):
                seen.add(name)
                results.append(name)

    if not results:
        return ["None"]

    if "VoxCPM2" in results:
        results.remove("VoxCPM2")
        results.insert(0, "VoxCPM2")
    return results


def resolve_model_path(model_name):
    if not model_name or model_name == "None":
        raise ValueError("请选择有效的 VoxCPM 模型。")

    for base_dir in folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE):
        full_path = os.path.join(base_dir, model_name)
        if os.path.isdir(full_path) and os.path.isfile(os.path.join(full_path, "config.json")):
            return full_path
    raise FileNotFoundError(f"找不到模型：{model_name}。请确认已放入 models/voxcpm/ 目录。")


def read_model_architecture(model_name):
    model_path = resolve_model_path(model_name)
    config_path = os.path.join(model_path, "config.json")
    with open(config_path, "r", encoding="utf-8") as file:
        config = json.load(file)
    return str(config.get("architecture", "voxcpm")).lower()


def ensure_dataset_root():
    dataset_root = os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE, DATASET_SUBDIR_NAME)
    os.makedirs(dataset_root, exist_ok=True)
    return dataset_root


def ensure_lora_root():
    lora_root = os.path.join(folder_paths.models_dir, VOXCPM_MODEL_TYPE, "loras")
    os.makedirs(lora_root, exist_ok=True)
    return lora_root


def sanitize_filename(name, fallback):
    value = (name or "").strip()
    if not value:
        value = fallback
    value = re.sub(r'[<>:"/\\|?*\x00-\x1F]+', "_", value)
    value = value.strip(" .")
    return value or fallback


def normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def load_audio_file(audio_path):
    try:
        waveform, sample_rate = torchaudio.load(audio_path)
    except Exception:
        audio_data, sample_rate = sf.read(audio_path, dtype="float32", always_2d=True)
        waveform = torch.from_numpy(audio_data.T)

    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.float(), int(sample_rate)


def save_waveform(audio_path, waveform, sample_rate):
    waveform = waveform.detach().cpu()
    try:
        torchaudio.save(audio_path, waveform, sample_rate)
    except Exception:
        sf.write(audio_path, waveform.numpy().T, sample_rate)


def save_audio_input(audio_input, audio_path):
    waveform = audio_input["waveform"]
    sample_rate = int(audio_input["sample_rate"])
    if waveform.dim() == 3:
        waveform = waveform.squeeze(0)
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    save_waveform(audio_path, waveform, sample_rate)
    return audio_path


def get_audio_duration_seconds(audio_path):
    try:
        info = torchaudio.info(audio_path)
        if info.num_frames and info.sample_rate:
            return round(float(info.num_frames) / float(info.sample_rate), 3)
    except Exception:
        pass

    try:
        info = sf.info(audio_path)
        if info.frames and info.samplerate:
            return round(float(info.frames) / float(info.samplerate), 3)
    except Exception:
        pass
    return None


def write_manifest(entries, manifest_path):
    manifest_path = Path(manifest_path)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.exists():
        stem = manifest_path.stem
        suffix = manifest_path.suffix or ".jsonl"
        match = re.match(r"^(.*)_(\d{3})$", stem)
        base_stem = match.group(1) if match else stem
        next_index = int(match.group(2)) + 1 if match else 1

        while True:
            candidate = manifest_path.with_name(f"{base_stem}_{next_index:03d}{suffix}")
            if not candidate.exists():
                manifest_path = candidate
                break
            next_index += 1

    with manifest_path.open("w", encoding="utf-8") as file:
        for entry in entries:
            file.write(json.dumps(entry, ensure_ascii=False) + "\n")
    return str(manifest_path.resolve())


def choose_unique_output_subdir(parent_dir, base_name):
    parent_dir = Path(parent_dir)
    candidate = parent_dir / base_name
    if not candidate.exists():
        return candidate

    try:
        has_existing_content = any(candidate.iterdir())
    except OSError:
        has_existing_content = True

    if not has_existing_content:
        return candidate

    match = re.match(r"^(.*)_(\d{3})$", base_name)
    base_stem = match.group(1) if match else base_name
    next_index = int(match.group(2)) + 1 if match else 1

    while True:
        candidate = parent_dir / f"{base_stem}_{next_index:03d}"
        if not candidate.exists():
            return candidate
        try:
            has_existing_content = any(candidate.iterdir())
        except OSError:
            has_existing_content = True
        if not has_existing_content:
            return candidate
        next_index += 1


def _resolve_manifest_media_path(base_dir, path_value):
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return ""

    media_path = Path(raw_path).expanduser()
    if not media_path.is_absolute():
        media_path = (base_dir / media_path).resolve()
    else:
        media_path = media_path.resolve()
    return str(media_path)


def _read_jsonl_manifest_entries(manifest_path):
    manifest_path = Path(manifest_path).expanduser().resolve()
    if not manifest_path.is_file():
        raise FileNotFoundError(f"找不到训练清单：{manifest_path}")

    entries = []
    skipped_count = 0
    with manifest_path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                entry = json.loads(line)
            except Exception as e:
                raise RuntimeError(f"训练清单第 {line_number} 行不是合法 JSON：{e}")

            if not isinstance(entry, dict):
                skipped_count += 1
                continue

            audio_path = _resolve_manifest_media_path(manifest_path.parent, entry.get("audio"))
            text = normalize_text(entry.get("text"))
            if not audio_path or not text:
                skipped_count += 1
                continue

            normalized_entry = dict(entry)
            normalized_entry["audio"] = audio_path
            normalized_entry["text"] = text

            ref_audio_path = _resolve_manifest_media_path(manifest_path.parent, entry.get("ref_audio"))
            if ref_audio_path:
                normalized_entry["ref_audio"] = ref_audio_path
            elif "ref_audio" in normalized_entry:
                normalized_entry.pop("ref_audio", None)

            entries.append(normalized_entry)

    return entries, skipped_count


def _pick_ref_indices(group_size, ref_audio_ratio):
    if group_size < 2 or ref_audio_ratio <= 0:
        return []

    desired = int(round(group_size * float(ref_audio_ratio)))
    desired = max(1, min(desired, group_size))
    if desired >= group_size:
        return list(range(group_size))

    step = float(group_size) / float(desired)
    cursor = step / 2.0
    indices = []
    seen = set()
    while len(indices) < desired:
        index = min(int(cursor), group_size - 1)
        while index in seen and index + 1 < group_size:
            index += 1
        while index in seen and index - 1 >= 0:
            index -= 1
        if index in seen:
            break
        seen.add(index)
        indices.append(index)
        cursor += step
    return sorted(indices)


def _assign_ref_audio(entries, ref_audio_ratio, group_key_func):
    ratio = float(ref_audio_ratio)
    if ratio <= 0:
        return 0

    groups = {}
    for entry in entries:
        groups.setdefault(group_key_func(entry), []).append(entry)

    assigned_count = 0
    for group_entries in groups.values():
        if len(group_entries) < 2:
            continue

        existing_count = sum(1 for entry in group_entries if str(entry.get("ref_audio", "")).strip())
        desired_count = int(round(len(group_entries) * ratio))
        desired_count = max(1, min(desired_count, len(group_entries)))
        if existing_count >= desired_count:
            continue

        for index in _pick_ref_indices(len(group_entries), ratio):
            if existing_count >= desired_count:
                break

            current_entry = group_entries[index]
            if str(current_entry.get("ref_audio", "")).strip():
                continue

            ref_index = index - 1 if index > 0 else 1
            if ref_index < 0 or ref_index >= len(group_entries):
                continue

            ref_audio = group_entries[ref_index].get("audio")
            if not ref_audio or ref_audio == current_entry.get("audio"):
                continue

            current_entry["ref_audio"] = ref_audio
            existing_count += 1
            assigned_count += 1

    return assigned_count


def choose_dataset_output_dir(source_mode, source_path, output_dir):
    if output_dir:
        target_dir = Path(output_dir).expanduser()
    elif source_mode == "长音频" and source_path:
        source_file = Path(source_path).expanduser()
        target_dir = source_file.parent / f"{source_file.stem}_dataset"
    elif source_mode == "数据集目录" and source_path:
        target_dir = Path(source_path).expanduser()
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        target_dir = Path(ensure_dataset_root()) / f"dataset_{timestamp}"

    target_dir = target_dir.resolve()
    target_dir.mkdir(parents=True, exist_ok=True)
    return target_dir


def _build_speech_regions(waveform, sample_rate, silence_threshold_db, min_silence_ms):
    analysis_sample_rate = 16000 if sample_rate > 16000 else sample_rate
    analysis_waveform = waveform
    if sample_rate != analysis_sample_rate:
        analysis_waveform = torchaudio.functional.resample(waveform, sample_rate, analysis_sample_rate)

    if analysis_waveform.dim() == 2:
        analysis_waveform = analysis_waveform[:1]

    if analysis_waveform.numel() > 0:
        window_size = max(1, int(analysis_sample_rate * 0.05))
        power_norm = analysis_waveform.pow(2).unsqueeze(0)
        rms_pool = F.avg_pool1d(power_norm, kernel_size=window_size, stride=window_size).sqrt()
        robust_peak = float(rms_pool.max().item())
        
        if robust_peak > 1e-6:
            analysis_waveform = analysis_waveform / robust_peak * 0.25
            analysis_waveform = torch.clamp(analysis_waveform, -1.0, 1.0)

    frame_length = max(1, int(analysis_sample_rate * 0.03))
    hop_length = max(1, int(analysis_sample_rate * 0.01))
    total_samples = int(analysis_waveform.shape[-1])

    if total_samples < frame_length:
        analysis_waveform = F.pad(analysis_waveform, (0, frame_length - total_samples))

    power = analysis_waveform.pow(2).unsqueeze(0)
    pooled = F.avg_pool1d(power, kernel_size=frame_length, stride=hop_length)
    rms = pooled.squeeze(0).squeeze(0).sqrt()
    decibels = 20.0 * torch.log10(rms + 1e-6)
    speech_mask = decibels > float(silence_threshold_db)

    regions = []
    start_index = None
    for index, is_speech in enumerate(speech_mask.tolist()):
        if is_speech and start_index is None:
            start_index = index
        elif not is_speech and start_index is not None:
            regions.append((start_index, index))
            start_index = None
    if start_index is not None:
        regions.append((start_index, len(speech_mask)))

    if not regions:
        full_frames = max(1, math.ceil(float(analysis_waveform.shape[-1]) / float(hop_length)))
        return [(0, full_frames)], speech_mask, analysis_sample_rate, hop_length

    min_gap_frames = max(1, int((float(min_silence_ms) / 1000.0) * analysis_sample_rate / hop_length))
    merged_regions = []
    for region_start, region_end in regions:
        if not merged_regions:
            merged_regions.append([region_start, region_end])
            continue
        last_start, last_end = merged_regions[-1]
        if region_start - last_end <= min_gap_frames:
            merged_regions[-1][1] = region_end
        else:
            merged_regions.append([region_start, region_end])

    merged_regions = [tuple(region) for region in merged_regions]

    total_frames = max(1, len(speech_mask))
    speech_frames = sum(max(0, end - start) for start, end in merged_regions)
    total_seconds = float(total_frames * hop_length) / float(analysis_sample_rate)
    speech_seconds = float(speech_frames * hop_length) / float(analysis_sample_rate)

    if total_seconds >= 30.0 and speech_seconds < max(8.0, total_seconds * 0.15):
        full_frames = max(1, math.ceil(float(analysis_waveform.shape[-1]) / float(hop_length)))
        return [(0, full_frames)], speech_mask, analysis_sample_rate, hop_length

    return merged_regions, speech_mask, analysis_sample_rate, hop_length


def _split_region(region_start, region_end, speech_mask, min_seconds, max_seconds, target_seconds, analysis_sample_rate, hop_length):
    frame_duration = float(hop_length) / float(analysis_sample_rate)
    min_frames = max(1, int(math.ceil(float(min_seconds) / frame_duration)))
    max_frames = max(min_frames, int(math.ceil(float(max_seconds) / frame_duration)))
    target_frames = max(min_frames, int(math.ceil(float(target_seconds) / frame_duration)))
    search_radius_frames = max(1, int(math.ceil(2.0 / frame_duration)))

    segments = []
    cursor = int(region_start)
    region_end = int(region_end)
    speech_mask_length = len(speech_mask)

    while region_end - cursor > max_frames:
        min_split = cursor + min_frames
        max_split = min(cursor + max_frames, region_end - min_frames)
        
        if min_split >= max_split:
            max_split = cursor + max_frames

        desired_split = min(cursor + target_frames, max_split)
        search_start = max(min_split, desired_split - search_radius_frames)
        search_end = min(max_split, desired_split + search_radius_frames)

        split_frame = None
        best_distance = None
        for frame_index in range(search_start, search_end + 1):
            if frame_index >= speech_mask_length:
                break
            if not bool(speech_mask[frame_index]):
                distance = abs(frame_index - desired_split)
                if best_distance is None or distance < best_distance:
                    best_distance = distance
                    split_frame = frame_index

        if split_frame is None:
            split_frame = min(cursor + max_frames, region_end)

        if split_frame <= cursor:
            split_frame = min(cursor + max_frames, region_end)

        segments.append((cursor, split_frame))
        cursor = split_frame

    if region_end > cursor:
        segments.append((cursor, region_end))

    return segments


def _normalize_segments(segments):
    normalized = []
    for start_seconds, end_seconds in sorted(segments, key=lambda item: (item[0], item[1])):
        if end_seconds <= start_seconds:
            continue
        if not normalized:
            normalized.append([start_seconds, end_seconds])
            continue

        previous_end = normalized[-1][1]
        if start_seconds < previous_end:
            start_seconds = previous_end
        if end_seconds <= start_seconds:
            continue
        normalized.append([start_seconds, end_seconds])
    return normalized


def _merge_short_segments(segments, min_seconds, max_gap_seconds):
    if not segments:
        return []

    merged = []
    pending = list(segments[0])

    for next_start, next_end in segments[1:]:
        pending_duration = pending[1] - pending[0]
        gap_seconds = max(0.0, next_start - pending[1])

        if pending_duration < float(min_seconds) and gap_seconds <= float(max_gap_seconds):
            pending[1] = next_end
            continue

        merged.append(pending)
        pending = [next_start, next_end]

    if merged:
        pending_duration = pending[1] - pending[0]
        previous_gap = max(0.0, pending[0] - merged[-1][1])
        if pending_duration < float(min_seconds) and previous_gap <= float(max_gap_seconds):
            merged[-1][1] = pending[1]
        else:
            merged.append(pending)
    else:
        merged.append(pending)

    return merged


def segment_long_audio(
    source_audio_path,
    output_dir,
    min_seconds,
    max_seconds,
    target_seconds,
    keep_silence_ms,
    silence_threshold_db,
):
    waveform, sample_rate = load_audio_file(source_audio_path)
    regions, speech_mask, analysis_sample_rate, hop_length = _build_speech_regions(
        waveform,
        sample_rate,
        silence_threshold_db=silence_threshold_db,
        min_silence_ms=250,
    )

    padded_segments = []
    keep_silence_seconds = float(keep_silence_ms) / 1000.0
    total_duration = float(waveform.shape[-1]) / float(sample_rate)

    for region_start, region_end in regions:
        split_regions = _split_region(
            region_start,
            region_end,
            speech_mask=speech_mask,
            min_seconds=min_seconds,
            max_seconds=max_seconds,
            target_seconds=target_seconds,
            analysis_sample_rate=analysis_sample_rate,
            hop_length=hop_length,
        )
        for clip_start, clip_end in split_regions:
            start_seconds = max(0.0, (clip_start * hop_length) / float(analysis_sample_rate) - keep_silence_seconds)
            end_seconds = min(total_duration, (clip_end * hop_length) / float(analysis_sample_rate) + keep_silence_seconds)
            if end_seconds > start_seconds:
                padded_segments.append((start_seconds, end_seconds))

    if not padded_segments:
        padded_segments.append((0.0, min(total_duration, float(max_seconds))))

    non_overlapping_segments = _normalize_segments(padded_segments)
    
    non_overlapping_segments = _merge_short_segments(
        non_overlapping_segments,
        min_seconds=float(min_seconds),
        max_gap_seconds=max(1.0, keep_silence_seconds * 6.0),
    )

    final_segments = []
    min_valid_duration = max(1.0, float(min_seconds) * 0.5)
    for start_sec, end_sec in non_overlapping_segments:
        if end_sec - start_sec >= min_valid_duration:
            final_segments.append((start_sec, end_sec))
    
    if not final_segments:
        final_segments.append((0.0, min(total_duration, float(max_seconds))))

    clips_dir = choose_unique_output_subdir(output_dir, "clips")
    clips_dir.mkdir(parents=True, exist_ok=True)

    clip_paths = []
    for index, (start_seconds, end_seconds) in enumerate(final_segments, start=1):
        start_sample = max(0, int(start_seconds * sample_rate))
        end_sample = min(int(end_seconds * sample_rate), waveform.shape[-1])
        if end_sample <= start_sample:
            continue
        clip_waveform = waveform[:, start_sample:end_sample]
        if clip_waveform.shape[-1] < int(0.5 * sample_rate):
            continue

        clip_path = clips_dir / f"segment_{index:04d}.wav"
        save_waveform(str(clip_path), clip_waveform, sample_rate)
        clip_paths.append(str(clip_path.resolve()))

    return clip_paths


def _format_ref_audio_suffix(ref_audio_count):
    if ref_audio_count > 0:
        return f" 其中 {ref_audio_count} 条样本已分配 ref_audio。"
    return ""


def build_manifest_from_long_audio(
    source_audio_path,
    output_dir,
    manifest_name=DEFAULT_MANIFEST_NAME,
    min_seconds=2.0,
    max_seconds=10.0,
    target_seconds=6.0,
    keep_silence_ms=150,
    silence_threshold_db=-40.0,
    enable_ref_audio=False,
    ref_audio_ratio=0.4,
):
    clip_paths = segment_long_audio(
        source_audio_path=source_audio_path,
        output_dir=output_dir,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        target_seconds=target_seconds,
        keep_silence_ms=keep_silence_ms,
        silence_threshold_db=silence_threshold_db,
    )
    if not clip_paths:
        raise RuntimeError("没有成功切分出可用于训练的语音片段。")

    progress_bar = comfy.utils.ProgressBar(len(clip_paths))
    entries = []
    skipped_count = 0

    for clip_path in clip_paths:
        text = normalize_text(_recognize_audio(clip_path))
        if not text:
            skipped_count += 1
            progress_bar.update(1)
            continue

        duration = get_audio_duration_seconds(clip_path)
        entry = {
            "audio": clip_path,
            "text": text,
        }
        if duration is not None:
            entry["duration"] = duration
        entries.append(entry)
        progress_bar.update(1)

    if not entries:
        raise RuntimeError("自动识别后没有得到有效文本，无法生成训练清单。")

    ref_audio_count = 0
    if enable_ref_audio:
        ref_audio_count = _assign_ref_audio(entries, ref_audio_ratio, lambda _entry: "single_speaker")

    manifest_path = write_manifest(entries, Path(output_dir) / manifest_name)
    info = (
        f"已从长音频生成数据集，共切分 {len(clip_paths)} 段，"
        f"保留 {len(entries)} 段，跳过 {skipped_count} 段。"
    )
    info += _format_ref_audio_suffix(ref_audio_count)
    return manifest_path, str(Path(output_dir).resolve()), info


def _iter_audio_files(dataset_dir, recursive):
    dataset_dir = Path(dataset_dir)
    audio_files = []
    for extension in SUPPORTED_AUDIO_EXTENSIONS:
        pattern = f"**/*{extension}" if recursive else f"*{extension}"
        for audio_path in dataset_dir.glob(pattern):
            if audio_path.is_file():
                audio_files.append(audio_path)
    return sorted(audio_files)


def build_manifest_from_directory(
    dataset_dir,
    manifest_name=DEFAULT_MANIFEST_NAME,
    recursive=True,
    output_dir=None,
    enable_ref_audio=False,
    ref_audio_ratio=0.4,
):
    dataset_dir = Path(dataset_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到数据集目录：{dataset_dir}")

    audio_files = _iter_audio_files(dataset_dir, recursive=recursive)
    if not audio_files:
        raise RuntimeError(f"在目录中没有找到可用音频文件：{dataset_dir}")

    entries = []
    skipped_count = 0
    for audio_path in audio_files:
        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file():
            skipped_count += 1
            continue

        text = normalize_text(text_path.read_text(encoding="utf-8"))
        if not text:
            skipped_count += 1
            continue

        entry = {
            "audio": str(audio_path.resolve()),
            "text": text,
        }
        duration = get_audio_duration_seconds(str(audio_path))
        if duration is not None:
            entry["duration"] = duration
        entries.append(entry)

    if not entries:
        raise RuntimeError("没有找到有效的 wav/txt 配对数据，无法生成训练清单。")

    ref_audio_count = 0
    if enable_ref_audio:
        def group_key(entry):
            audio_parent = Path(entry["audio"]).parent
            try:
                return str(audio_parent.relative_to(dataset_dir))
            except ValueError:
                return str(audio_parent)

        ref_audio_count = _assign_ref_audio(entries, ref_audio_ratio, group_key)

    manifest_parent = Path(output_dir).expanduser().resolve() if output_dir else dataset_dir
    manifest_parent.mkdir(parents=True, exist_ok=True)
    manifest_path = write_manifest(entries, manifest_parent / manifest_name)
    info = (
        f"已从目录生成训练清单，共写入 {len(entries)} 条样本，"
        f"跳过 {skipped_count} 条缺少文本或文本为空的样本。"
    )
    info += _format_ref_audio_suffix(ref_audio_count)
    return manifest_path, str(dataset_dir), info


def build_manifest_from_jsonl(
    manifest_path,
    manifest_name=DEFAULT_MANIFEST_NAME,
    output_dir=None,
    enable_ref_audio=False,
    ref_audio_ratio=0.4,
):
    source_manifest = Path(manifest_path).expanduser().resolve()
    entries, skipped_count = _read_jsonl_manifest_entries(source_manifest)
    if not entries:
        raise RuntimeError("现有 train.jsonl 中没有有效的 audio/text 样本，无法继续。")

    ref_audio_count = 0
    if enable_ref_audio:
        def group_key(entry):
            return str(Path(entry["audio"]).parent)

        ref_audio_count = _assign_ref_audio(entries, ref_audio_ratio, group_key)

    manifest_parent = Path(output_dir).expanduser().resolve() if output_dir else source_manifest.parent
    manifest_parent.mkdir(parents=True, exist_ok=True)
    saved_manifest_path = write_manifest(entries, manifest_parent / manifest_name)
    info = (
        f"已整理现有 train.jsonl，共写入 {len(entries)} 条样本，"
        f"跳过 {skipped_count} 条无效样本。"
    )
    info += _format_ref_audio_suffix(ref_audio_count)
    return saved_manifest_path, str(manifest_parent), info


def build_manifest_from_batch_audio(
    source_dir,
    output_dir,
    manifest_name=DEFAULT_MANIFEST_NAME,
    recursive=True,
    min_seconds=2.0,
    max_seconds=10.0,
    target_seconds=6.0,
    keep_silence_ms=150,
    silence_threshold_db=-40.0,
    enable_ref_audio=False,
    ref_audio_ratio=0.4,
):
    dataset_dir = Path(source_dir).expanduser().resolve()
    if not dataset_dir.is_dir():
        raise FileNotFoundError(f"找不到批量音频目录：{dataset_dir}")

    audio_files = _iter_audio_files(dataset_dir, recursive=recursive)
    if not audio_files:
        raise RuntimeError(f"在目录中没有找到可用音频文件：{dataset_dir}")

    if not output_dir:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = Path(ensure_dataset_root()) / f"batch_dataset_{timestamp}"
    else:
        output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    entries = []
    skipped_count = 0
    import comfy.utils
    progress_bar = comfy.utils.ProgressBar(len(audio_files))

    for audio_path in audio_files:
        duration = get_audio_duration_seconds(str(audio_path))
        
        # 宽限 0.5 秒容错，超过了就当作长音频进行降噪+切片
        if duration is not None and duration > float(max_seconds) + 0.5:
            clip_paths = segment_long_audio(
                source_audio_path=str(audio_path),
                output_dir=str(output_dir),
                min_seconds=min_seconds,
                max_seconds=max_seconds,
                target_seconds=target_seconds,
                keep_silence_ms=keep_silence_ms,
                silence_threshold_db=silence_threshold_db,
            )
            for clip_path in clip_paths:
                text = normalize_text(_recognize_audio(clip_path))
                if not text:
                    skipped_count += 1
                    continue
                clip_duration = get_audio_duration_seconds(clip_path)
                entry = {"audio": clip_path, "text": text}
                if clip_duration is not None:
                    entry["duration"] = clip_duration
                entries.append(entry)
        else:
            # 本身就是短音频，直接原文件打标写入 JSONL，不产生副本文本
            text = normalize_text(_recognize_audio(str(audio_path)))
            if not text:
                skipped_count += 1
            else:
                entry = {"audio": str(audio_path.resolve()), "text": text}
                if duration is not None:
                    entry["duration"] = duration
                entries.append(entry)
        
        progress_bar.update(1)

    if not entries:
        raise RuntimeError("批量音频处理后没有得到有效文本，无法生成训练清单。")

    ref_audio_count = 0
    if enable_ref_audio:
        def group_key(entry):
            return str(Path(entry["audio"]).parent)
        ref_audio_count = _assign_ref_audio(entries, ref_audio_ratio, group_key)

    manifest_path = write_manifest(entries, output_dir / manifest_name)
    info = (
        f"已从批量音频生成数据集，共处理生成 {len(entries)} 条样本，"
        f"跳过 {skipped_count} 条无效或无法识别片段。"
    )
    info += _format_ref_audio_suffix(ref_audio_count)
    
    return manifest_path, str(output_dir), info


def prepare_dataset(
    source_mode,
    source_path,
    output_dir,
    manifest_name,
    recursive,
    min_seconds,
    max_seconds,
    target_seconds,
    keep_silence_ms,
    silence_threshold_db,
    enable_ref_audio,
    ref_audio_ratio,
    audio_input=None,
):
    manifest_name = sanitize_filename(manifest_name or DEFAULT_MANIFEST_NAME, DEFAULT_MANIFEST_NAME)
    if not manifest_name.lower().endswith(".jsonl"):
        manifest_name = f"{manifest_name}.jsonl"

    source_mode = str(source_mode).strip()
    source_path = str(source_path or "").strip()

    if source_mode == "长音频":
        dataset_dir = choose_dataset_output_dir(source_mode, source_path, output_dir)
        if audio_input is not None:
            source_audio_path = str((dataset_dir / "source_audio.wav").resolve())
            save_audio_input(audio_input, source_audio_path)
        else:
            if not source_path:
                raise ValueError("长音频模式下请提供长音频路径，或者连接音频输入。")
            source_audio_path = str(Path(source_path).expanduser().resolve())
            if not os.path.isfile(source_audio_path):
                raise FileNotFoundError(f"找不到长音频文件：{source_audio_path}")
        return build_manifest_from_long_audio(
            source_audio_path=source_audio_path,
            output_dir=str(dataset_dir),
            manifest_name=manifest_name,
            min_seconds=float(min_seconds),
            max_seconds=float(max_seconds),
            target_seconds=float(target_seconds),
            keep_silence_ms=int(keep_silence_ms),
            silence_threshold_db=float(silence_threshold_db),
            enable_ref_audio=bool(enable_ref_audio),
            ref_audio_ratio=float(ref_audio_ratio),
        )

    if source_mode == "数据集目录":
        if not source_path:
            raise ValueError("数据集目录模式下请填写数据集目录或 train.jsonl 路径。")
        source = Path(source_path).expanduser().resolve()
        if source.is_file() and source.suffix.lower() == ".jsonl" and bool(enable_ref_audio):
            return build_manifest_from_jsonl(
                manifest_path=str(source),
                manifest_name=manifest_name,
                output_dir=output_dir,
                enable_ref_audio=True,
                ref_audio_ratio=float(ref_audio_ratio),
            )
        if source.is_file() and source.suffix.lower() == ".jsonl":
            info = "已直接使用现有 train.jsonl 训练清单。"
            return str(source), str(source.parent), info
        return build_manifest_from_directory(
            dataset_dir=str(source),
            manifest_name=manifest_name,
            recursive=bool(recursive),
            output_dir=output_dir,
            enable_ref_audio=bool(enable_ref_audio),
            ref_audio_ratio=float(ref_audio_ratio),
        )

    if source_mode == "批量音频":
        if not source_path:
            raise ValueError("批量音频模式下请填写来源路径（包含多个音频的目录）。")
        return build_manifest_from_batch_audio(
            source_dir=source_path,
            output_dir=output_dir,
            manifest_name=manifest_name,
            recursive=bool(recursive),
            min_seconds=float(min_seconds),
            max_seconds=float(max_seconds),
            target_seconds=float(target_seconds),
            keep_silence_ms=int(keep_silence_ms),
            silence_threshold_db=float(silence_threshold_db),
            enable_ref_audio=bool(enable_ref_audio),
            ref_audio_ratio=float(ref_audio_ratio),
        )

    raise ValueError(f"不支持的数据来源模式：{source_mode}")


def ensure_training_dependencies():
    missing = []
    for package_name in ("argbind", "datasets", "transformers"):
        try:
            importlib.import_module(package_name)
        except ImportError:
            missing.append(package_name)

    if missing:
        raise RuntimeError(
            "LoRA 训练依赖缺失："
            + ", ".join(missing)
            + "。请先安装这些包后再使用训练节点。"
        )


def import_training_modules():
    ensure_training_dependencies()

    from torch.optim import AdamW
    from transformers import get_cosine_schedule_with_warmup
    from voxcpm.model.voxcpm import LoRAConfig as LoRAConfigV1
    from voxcpm.model.voxcpm import VoxCPMModel
    from voxcpm.model.voxcpm2 import LoRAConfig as LoRAConfigV2
    from voxcpm.model.voxcpm2 import VoxCPM2Model
    from voxcpm.training import Accelerator
    from voxcpm.training import BatchProcessor
    from voxcpm.training import build_dataloader
    from voxcpm.training import load_audio_text_datasets

    try:
        from safetensors.torch import load_file as load_safetensors_file
        from safetensors.torch import save_file as save_safetensors_file
        safetensors_available = True
    except ImportError:
        load_safetensors_file = None
        save_safetensors_file = None
        safetensors_available = False

    return {
        "AdamW": AdamW,
        "get_cosine_schedule_with_warmup": get_cosine_schedule_with_warmup,
        "VoxCPMModel": VoxCPMModel,
        "VoxCPM2Model": VoxCPM2Model,
        "LoRAConfigV1": LoRAConfigV1,
        "LoRAConfigV2": LoRAConfigV2,
        "Accelerator": Accelerator,
        "BatchProcessor": BatchProcessor,
        "build_dataloader": build_dataloader,
        "load_audio_text_datasets": load_audio_text_datasets,
        "load_safetensors_file": load_safetensors_file,
        "save_safetensors_file": save_safetensors_file,
        "safetensors_available": safetensors_available,
    }


def build_output_dir(output_name, resume_training):
    base_name = sanitize_filename(output_name, f"voxcpm_lora_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    lora_root = Path(ensure_lora_root())
    output_dir = lora_root / base_name

    if resume_training:
        if not output_dir.is_dir():
            raise FileNotFoundError(f"继续训练时找不到输出目录：{output_dir}")
        return output_dir.resolve()

    if not output_dir.exists():
        output_dir.mkdir(parents=True, exist_ok=False)
        return output_dir.resolve()

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = lora_root / f"{base_name}_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=False)
    return output_dir.resolve()


def _extract_lora_state_dict(model):
    state_dict = model.state_dict()
    return {key: value.detach().cpu() for key, value in state_dict.items() if "lora_" in key}


def _build_lora_info_payload(model_path, architecture, lora_config):
    config_dict = lora_config.model_dump() if hasattr(lora_config, "model_dump") else vars(lora_config)
    base_model_path = str(model_path)
    return {
        "base_model": base_model_path,
        "base_model_name": os.path.basename(base_model_path.rstrip("/\\")),
        "architecture": str(architecture or ""),
        "lora_config": dict(config_dict),
    }


def _build_lora_safetensors_metadata(lora_info):
    metadata = {
        "voxcpm.format": "voxcpm_lora",
        "voxcpm.lora_info": json.dumps(lora_info, ensure_ascii=False),
    }

    base_model = str(lora_info.get("base_model", "") or "").strip()
    if base_model:
        metadata["voxcpm.base_model"] = base_model

    base_model_name = str(lora_info.get("base_model_name", "") or "").strip()
    if base_model_name:
        metadata["voxcpm.base_model_name"] = base_model_name

    architecture = str(lora_info.get("architecture", "") or "").strip()
    if architecture:
        metadata["voxcpm.architecture"] = architecture

    config_dict = lora_info.get("lora_config", {})
    if isinstance(config_dict, dict):
        for key in ("r", "alpha", "dropout", "enable_lm", "enable_dit", "enable_proj"):
            if key in config_dict:
                metadata[f"voxcpm.{key}"] = json.dumps(config_dict[key], ensure_ascii=False)

    return metadata


def _write_lora_info_json(file_path, lora_info):
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    with file_path.open("w", encoding="utf-8") as file:
        json.dump(lora_info, file, indent=2, ensure_ascii=False)


def _save_lora_state_file(lora_state, file_path, save_safetensors_file, safetensors_available, metadata=None):
    file_path = Path(file_path)
    if safetensors_available:
        save_safetensors_file(lora_state, str(file_path), metadata=metadata)
    else:
        torch.save({"state_dict": lora_state}, file_path)


def _save_checkpoint_snapshot(
    model,
    optimizer,
    scheduler,
    output_dir,
    model_path,
    architecture,
    lora_config,
    step,
    save_safetensors_file,
    safetensors_available,
):
    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    lora_state = _extract_lora_state_dict(model)
    if not lora_state:
        raise RuntimeError("没有提取到任何 LoRA 权重，训练无法保存。")

    lora_info = _build_lora_info_payload(
        model_path=model_path,
        architecture=architecture,
        lora_config=lora_config,
    )
    metadata = _build_lora_safetensors_metadata(lora_info) if safetensors_available else None
    snapshot_suffix = ".safetensors" if safetensors_available else ".ckpt"
    snapshot_path = checkpoints_dir / f"step_{step:07d}{snapshot_suffix}"
    latest_path = output_dir / f"lora_weights{snapshot_suffix}"
    export_path = output_dir.parent / f"{output_dir.name}{snapshot_suffix}"
    export_config_path = output_dir.parent / f"{output_dir.name}.lora_config.json"

    _save_lora_state_file(
        lora_state=lora_state,
        file_path=snapshot_path,
        save_safetensors_file=save_safetensors_file,
        safetensors_available=safetensors_available,
        metadata=metadata,
    )
    shutil.copy2(snapshot_path, latest_path)
    if safetensors_available:
        shutil.copy2(snapshot_path, export_path)
        _write_lora_info_json(export_config_path, lora_info)
    _write_lora_info_json(output_dir / "lora_config.json", lora_info)

    with (output_dir / "training_state.json").open("w", encoding="utf-8") as file:
        json.dump({"step": int(step)}, file, indent=2, ensure_ascii=False)

    torch.save(optimizer.state_dict(), output_dir / "optimizer.pth")
    torch.save(scheduler.state_dict(), output_dir / "scheduler.pth")
    return snapshot_path


def _load_resume_checkpoint(
    model,
    optimizer,
    scheduler,
    output_dir,
    load_safetensors_file,
):
    output_dir = Path(output_dir)
    latest_safe = output_dir / "lora_weights.safetensors"
    latest_ckpt = output_dir / "lora_weights.ckpt"
    if latest_safe.is_file():
        if load_safetensors_file is None:
            raise RuntimeError("当前环境缺少 safetensors，无法继续加载已有的 .safetensors LoRA 权重。")
        state_dict = load_safetensors_file(str(latest_safe))
    elif latest_ckpt.is_file():
        checkpoint = torch.load(latest_ckpt, map_location="cpu")
        state_dict = checkpoint.get("state_dict", checkpoint)
    else:
        return 0

    model.load_state_dict(state_dict, strict=False)

    optimizer_path = output_dir / "optimizer.pth"
    scheduler_path = output_dir / "scheduler.pth"
    state_path = output_dir / "training_state.json"

    if optimizer_path.is_file():
        optimizer.load_state_dict(torch.load(optimizer_path, map_location="cpu"))
    if scheduler_path.is_file():
        scheduler.load_state_dict(torch.load(scheduler_path, map_location="cpu"))

    if state_path.is_file():
        with state_path.open("r", encoding="utf-8") as file:
            state = json.load(file)
        return int(state.get("step", 0))
    return 0


def _collect_trainable_param_names(model):
    trainable_names = []
    lora_names = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        trainable_names.append(name)
        if "lora" in name.lower():
            lora_names.append(name)
    return trainable_names, lora_names


def _format_name_preview(names, limit=8):
    if not names:
        return "无"
    preview = ", ".join(names[:limit])
    if len(names) > limit:
        preview += f" ... 共 {len(names)} 项"
    return preview


def _read_batch_supervision_stats(processed):
    loss_mask = processed.get("loss_mask")
    audio_mask = processed.get("audio_mask")
    text_tokens = processed.get("text_tokens")
    audio_feats = processed.get("audio_feats")

    def tensor_sum(tensor):
        if torch.is_tensor(tensor) and tensor.numel() > 0:
            return int(tensor.sum().item())
        return 0

    return {
        "loss_tokens": tensor_sum(loss_mask),
        "audio_tokens": tensor_sum(audio_mask),
        "text_shape": tuple(text_tokens.shape) if torch.is_tensor(text_tokens) else (),
        "audio_shape": tuple(audio_feats.shape) if torch.is_tensor(audio_feats) else (),
    }


def _format_loss_state(outputs):
    if not isinstance(outputs, dict):
        return f"outputs 类型为 {type(outputs).__name__}"

    states = []
    for key, value in outputs.items():
        if not key.startswith("loss/"):
            continue
        requires_grad = bool(getattr(value, "requires_grad", False))
        shape = tuple(value.shape) if torch.is_tensor(value) else ()
        states.append(f"{key}(requires_grad={requires_grad}, shape={shape})")
    return ", ".join(states) if states else "没有找到 loss/* 输出"


@torch.inference_mode(False)
def run_lora_training(
    model_name,
    dataset_path,
    output_name,
    max_steps,
    save_every_steps,
    learning_rate,
    lora_rank,
    lora_alpha,
    lora_dropout,
    warmup_steps,
    grad_accum_steps,
    batch_size,
    max_batch_tokens,
    weight_decay,
    num_workers,
    enable_lm_lora,
    enable_dit_lora,
    enable_proj_lora,
    resume_training,
):
    dataset_path = Path(dataset_path).expanduser().resolve()
    if not dataset_path.is_file():
        raise FileNotFoundError(f"找不到训练清单：{dataset_path}")

    modules = import_training_modules()
    model_path = Path(resolve_model_path(model_name)).resolve()
    architecture = read_model_architecture(model_name)
    output_dir = build_output_dir(output_name, resume_training=resume_training)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_config = {
        "model_name": model_name,
        "model_path": str(model_path),
        "architecture": architecture,
        "dataset_path": str(dataset_path),
        "max_steps": int(max_steps),
        "save_every_steps": int(save_every_steps),
        "learning_rate": float(learning_rate),
        "lora_rank": int(lora_rank),
        "lora_alpha": int(lora_alpha),
        "lora_dropout": float(lora_dropout),
        "warmup_steps": int(warmup_steps),
        "grad_accum_steps": int(grad_accum_steps),
        "batch_size": int(batch_size),
        "max_batch_tokens": int(max_batch_tokens),
        "weight_decay": float(weight_decay),
        "num_workers": int(num_workers),
        "enable_lm_lora": bool(enable_lm_lora),
        "enable_dit_lora": bool(enable_dit_lora),
        "enable_proj_lora": bool(enable_proj_lora),
        "resume_training": bool(resume_training),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }
    with (output_dir / "train_config.json").open("w", encoding="utf-8") as file:
        json.dump(train_config, file, indent=2, ensure_ascii=False)

    if hasattr(model_management, "unload_all_models"):
        model_management.unload_all_models()
    model_management.soft_empty_cache()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    Accelerator = modules["Accelerator"]
    BatchProcessor = modules["BatchProcessor"]
    build_dataloader = modules["build_dataloader"]
    load_audio_text_datasets = modules["load_audio_text_datasets"]
    AdamW = modules["AdamW"]
    get_cosine_schedule_with_warmup = modules["get_cosine_schedule_with_warmup"]
    model_cls = modules["VoxCPM2Model"] if architecture == "voxcpm2" else modules["VoxCPMModel"]
    lora_config_cls = modules["LoRAConfigV2"] if architecture == "voxcpm2" else modules["LoRAConfigV1"]
    save_safetensors_file = modules["save_safetensors_file"]
    load_safetensors_file = modules["load_safetensors_file"]
    safetensors_available = modules["safetensors_available"]

    accelerator = Accelerator(amp=True)

    lora_config = lora_config_cls(
        enable_lm=bool(enable_lm_lora),
        enable_dit=bool(enable_dit_lora),
        enable_proj=bool(enable_proj_lora),
        r=int(lora_rank),
        alpha=int(lora_alpha),
        dropout=float(lora_dropout),
    )

    print(f"[VoxCPM][训练] 正在加载模型：{model_name}")
    base_model = model_cls.from_local(
        str(model_path),
        optimize=False,
        training=True,
        lora_config=lora_config,
    )
    base_model = base_model.to(accelerator.device)

    trainable_params, lora_trainable_params = _collect_trainable_param_names(base_model)
    if not lora_trainable_params:
        for name, param in base_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        trainable_params, lora_trainable_params = _collect_trainable_param_names(base_model)
    if not lora_trainable_params:
        raise RuntimeError(
            "没有检测到可训练的 LoRA 参数。当前训练环境里的 voxcpm 可能没有正确注入 LoRA 层，请检查训练依赖或模型实现。"
        )

    trainable_param_count = sum(
        int(param.numel()) for _, param in base_model.named_parameters() if param.requires_grad
    )
    print(
        f"[VoxCPM][训练] 检测到 {len(trainable_params)} 个可训练参数项，"
        f"其中 LoRA 参数 {len(lora_trainable_params)} 项，合计 {trainable_param_count} 个参数"
    )
    print(f"[VoxCPM][训练] LoRA 参数预览：{_format_name_preview(lora_trainable_params)}")

    trainable_params = [name for name, param in base_model.named_parameters() if param.requires_grad]
    if not trainable_params:
        for name, param in base_model.named_parameters():
            if "lora" in name.lower():
                param.requires_grad = True
        trainable_params = [name for name, param in base_model.named_parameters() if param.requires_grad]
    if not trainable_params:
        raise RuntimeError("没有检测到可训练的 LoRA 参数，训练已中止。")

    tokenizer = base_model.text_tokenizer
    detected_sample_rate = int(base_model.audio_vae.sample_rate)
    print(f"[VoxCPM][训练] 模型架构：{architecture}，训练采样率自动匹配为 {detected_sample_rate} Hz")

    train_dataset, _ = load_audio_text_datasets(
        train_manifest=str(dataset_path),
        sample_rate=detected_sample_rate,
    )

    def tokenize(batch):
        text_list = batch["text"]
        text_ids = [tokenizer(text) for text in text_list]
        return {"text_ids": text_ids}

    train_dataset = train_dataset.map(tokenize, batched=True, remove_columns=["text"])

    max_batch_tokens = int(max_batch_tokens)
    if max_batch_tokens > 0:
        from voxcpm.training.data import compute_sample_lengths

        audio_vae_fps = base_model.audio_vae.sample_rate / base_model.audio_vae.hop_length
        estimated_lengths = compute_sample_lengths(
            train_dataset,
            audio_vae_fps=audio_vae_fps,
            patch_size=base_model.config.patch_size,
        )
        keep_indices = [index for index, length in enumerate(estimated_lengths) if length <= max_batch_tokens]
        train_dataset = train_dataset.select(keep_indices)

    if len(train_dataset) == 0:
        raise RuntimeError("训练清单在过滤后为空，无法开始训练。请检查数据或调大最大批量 Token。")

    batch_size = max(1, int(batch_size))
    safe_num_workers = 0 if os.name == "nt" else max(0, int(num_workers))
    train_loader = build_dataloader(
        train_dataset,
        accelerator=accelerator,
        batch_size=batch_size,
        num_workers=safe_num_workers,
        drop_last=len(train_dataset) >= batch_size,
    )

    dataset_count = int(max(train_dataset["dataset_id"])) + 1 if "dataset_id" in train_dataset.column_names else 1
    batch_processor = BatchProcessor(
        config=base_model.config,
        audio_vae=base_model.audio_vae,
        dataset_cnt=dataset_count,
        device=accelerator.device,
    )
    base_model.audio_vae = None

    model = accelerator.prepare_model(base_model)
    unwrapped_model = accelerator.unwrap(model)
    unwrapped_model.train()
    _, post_prepare_lora_params = _collect_trainable_param_names(unwrapped_model)
    if not post_prepare_lora_params:
        raise RuntimeError("模型加载到训练设备后没有保留可训练的 LoRA 参数，请检查训练环境。")

    optimizer = AdamW(
        (parameter for parameter in model.parameters() if parameter.requires_grad),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    scheduler = get_cosine_schedule_with_warmup(
        optimizer,
        num_warmup_steps=int(warmup_steps),
        num_training_steps=int(max_steps),
    )

    start_step = 0
    if resume_training:
        start_step = _load_resume_checkpoint(
            model=unwrapped_model,
            optimizer=optimizer,
            scheduler=scheduler,
            output_dir=output_dir,
            load_safetensors_file=load_safetensors_file,
        )
        if start_step >= int(max_steps):
            raise RuntimeError(
                f"当前输出目录已经训练到第 {start_step} 步，已不小于目标总步数 {int(max_steps)}。"
            )

    total_steps = int(max_steps)
    save_every_steps = max(1, int(save_every_steps))
    grad_accum_steps = max(1, int(grad_accum_steps))
    max_batch_attempts = max(grad_accum_steps * 4, min(max(len(train_dataset) * 2, 8), 64))
    progress_bar = comfy.utils.ProgressBar(total_steps)
    if start_step > 0:
        progress_bar.update(start_step)

    train_iterator = iter(train_loader)
    data_epoch = 0
    skipped_batches = 0

    def get_next_batch():
        nonlocal train_iterator, data_epoch
        try:
            return next(train_iterator)
        except StopIteration:
            data_epoch += 1
            sampler = getattr(train_loader, "sampler", None)
            if hasattr(sampler, "set_epoch"):
                sampler.set_epoch(data_epoch)
            train_iterator = iter(train_loader)
            return next(train_iterator)

    try:
        with torch.enable_grad():
            for step in range(start_step, total_steps):
                if hasattr(model_management, "throw_exception_if_processing_interrupted"):
                    model_management.throw_exception_if_processing_interrupted()
                optimizer.zero_grad(set_to_none=True)

                total_loss_value = 0.0
                did_backward = False
                successful_micro_steps = 0
                batch_attempts = 0
                last_skip_reason = ""
                loss_value_sums = {}

                for micro_step in range(grad_accum_steps):
                    processed = None
                    supervision_stats = None

                    while batch_attempts < max_batch_attempts:
                        batch_attempts += 1
                        batch = get_next_batch()
                        candidate_processed = batch_processor(batch)
                        candidate_stats = _read_batch_supervision_stats(candidate_processed)
                        if candidate_stats["audio_tokens"] > 0 and candidate_stats["loss_tokens"] > 0:
                            processed = candidate_processed
                            supervision_stats = candidate_stats
                            break

                        skipped_batches += 1
                        last_skip_reason = (
                            "批次没有有效监督信号，"
                            f"audio_tokens={candidate_stats['audio_tokens']}, "
                            f"loss_tokens={candidate_stats['loss_tokens']}, "
                            f"text_shape={candidate_stats['text_shape']}, "
                            f"audio_shape={candidate_stats['audio_shape']}"
                        )
                        if skipped_batches <= 3 or skipped_batches % 10 == 0:
                            print(f"[VoxCPM][训练] 跳过无效批次：{last_skip_reason}")

                    if processed is None:
                        break

                    is_last_micro_step = micro_step == grad_accum_steps - 1
                    sync_context = contextlib.nullcontext() if is_last_micro_step else accelerator.no_sync()

                    with sync_context:
                        with accelerator.autocast(dtype=torch.bfloat16):
                            outputs = model(
                                processed["text_tokens"],
                                processed["text_mask"],
                                processed["audio_feats"],
                                processed["audio_mask"],
                                processed["loss_mask"],
                                processed["position_ids"],
                                processed["labels"],
                                progress=step / max(1, total_steps),
                            )

                        total_loss = None
                        for key, value in outputs.items():
                            if not key.startswith("loss/"):
                                continue
                            if value.numel() > 1:
                                value = value.mean()
                            loss_value_sums[key] = loss_value_sums.get(key, 0.0) + float(value.detach().item())
                            loss_term = value / grad_accum_steps
                            total_loss = loss_term if total_loss is None else total_loss + loss_term

                        if total_loss is None:
                            last_skip_reason = f"模型前向没有返回 loss/* 输出：{_format_loss_state(outputs)}"
                            print(f"[VoxCPM][训练] 首步梯度诊断：{last_skip_reason}")

                        if total_loss is not None and total_loss.grad_fn is None and step == start_step and micro_step == 0:
                            last_skip_reason = (
                                "批次前向完成但没有得到可反向传播的 loss，"
                                f"{_format_loss_state(outputs)}，"
                                f"audio_tokens={supervision_stats['audio_tokens']}，"
                                f"loss_tokens={supervision_stats['loss_tokens']}"
                            )
                            print(f"[VoxCPM][训练] 首步梯度诊断：{last_skip_reason}")

                        if total_loss is not None and total_loss.grad_fn is not None:
                            accelerator.backward(total_loss)
                            total_loss_value += float(total_loss.item()) * grad_accum_steps
                            did_backward = True
                            successful_micro_steps += 1
                        elif False:
                            raise RuntimeError("首个训练步没有得到有效梯度，请检查 LoRA 配置或训练数据。")

                if not did_backward:
                    raise RuntimeError(
                        "当前训练步没有得到有效梯度。"
                        f"最后一次诊断：{last_skip_reason or '未拿到可用于训练的有效批次'}。"
                    )

                if did_backward:
                    scaler = getattr(accelerator, "scaler", None)
                    if scaler is not None:
                        scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(unwrapped_model.parameters(), max_norm=1.0)
                    accelerator.step(optimizer)
                    accelerator.update()
                    scheduler.step()

                progress_bar.update(1)

                if (step + 1) % 10 == 0 or step == start_step:
                    current_lr = optimizer.param_groups[0]["lr"]
                    avg_loss_diff = loss_value_sums.get("loss/diff", 0.0) / max(1, successful_micro_steps)
                    avg_loss_stop = loss_value_sums.get("loss/stop", 0.0) / max(1, successful_micro_steps)
                    print(
                        f"[VoxCPM][训练] Step {step + 1}/{total_steps}, "
                        f"loss/diff: {avg_loss_diff:.6f}, "
                        f"loss/stop: {avg_loss_stop:.6f}, "
                        f"lr: {current_lr:.8f}"
                    )

                if (step + 1) % save_every_steps == 0 or (step + 1) == total_steps:
                    _save_checkpoint_snapshot(
                        model=unwrapped_model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        output_dir=output_dir,
                        model_path=model_path,
                        architecture=architecture,
                        lora_config=lora_config,
                        step=step + 1,
                        save_safetensors_file=save_safetensors_file,
                        safetensors_available=safetensors_available,
                    )
    finally:
        del batch_processor
        del model
        del optimizer
        del scheduler
        del train_loader
        gc.collect()
        model_management.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    latest_weights_path = output_dir / ("lora_weights.safetensors" if safetensors_available else "lora_weights.ckpt")
    if safetensors_available:
        single_file_path = output_dir.parent / f"{output_dir.name}.safetensors"
        if single_file_path.is_file():
            latest_weights_path = single_file_path
    info = (
        f"LoRA 训练完成，模型：{model_name}，"
        f"总步数：{total_steps}，"
        f"输出目录：{output_dir}"
    )
    return str(output_dir), str(latest_weights_path), info
