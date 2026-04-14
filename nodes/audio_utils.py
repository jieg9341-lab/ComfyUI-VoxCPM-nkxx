import os
import logging
import torch
import torchaudio
import tempfile
import numpy as np
import soundfile as sf
import folder_paths

logger = logging.getLogger("VoxCPM.Tools")

SENSEVOICE_MODEL_TYPE = "SenseVoice"
VOXCPM_MODEL_TYPE = "voxcpm"
ZIPENHANCER_DIR_NAME = "speech_zipenhancer_ans_multiloss_16k_base"

folder_paths.add_model_folder_path(SENSEVOICE_MODEL_TYPE, os.path.join(folder_paths.models_dir, SENSEVOICE_MODEL_TYPE))

_asr_model = None
_denoiser = None

def _safe_save_wav(path, waveform, sample_rate):
    try:
        torchaudio.save(path, waveform, sample_rate)
    except Exception:
        sf.write(path, waveform.numpy().T, sample_rate)

def _save_audio_to_temp(audio_dict):
    waveform = audio_dict["waveform"]
    sr = audio_dict["sample_rate"]
    if waveform.dim() == 3: waveform = waveform.squeeze(0)
    if waveform.shape[0] > 1: waveform = waveform.mean(dim=0, keepdim=True)
    
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    _safe_save_wav(tmp.name, waveform.cpu(), sr)
    return tmp.name

def _get_asr_model():
    global _asr_model
    if _asr_model is not None: return _asr_model
    from funasr import AutoModel
    
    base_dirs = folder_paths.get_folder_paths(SENSEVOICE_MODEL_TYPE)
    model_path = next((os.path.join(b, "SenseVoiceSmall") for b in base_dirs if os.path.isdir(os.path.join(b, "SenseVoiceSmall"))), None)
    if not model_path: raise FileNotFoundError("SenseVoiceSmall model not found.")
    
    _asr_model = AutoModel(model=model_path, disable_update=True, device="cuda:0" if torch.cuda.is_available() else "cpu")
    return _asr_model

def _recognize_audio(wav_path):
    asr = _get_asr_model()
    res = asr.generate(input=wav_path, language="auto", use_itn=True)
    return res[0]["text"].split("|>")[-1]

def _get_denoiser():
    global _denoiser
    if _denoiser is not None: return _denoiser
    from voxcpm.zipenhancer import ZipEnhancer
    
    base_dirs = folder_paths.get_folder_paths(VOXCPM_MODEL_TYPE)
    model_path = next((os.path.join(b, ZIPENHANCER_DIR_NAME) for b in base_dirs if os.path.isdir(os.path.join(b, ZIPENHANCER_DIR_NAME))), None)
    if not model_path: raise FileNotFoundError(f"ZipEnhancer not found.")
    
    _denoiser = ZipEnhancer(model_path)
    return _denoiser

def _denoise_audio(input_path):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp.close()
    _get_denoiser().enhance(input_path, output_path=tmp.name)
    return tmp.name
