# ComfyUI VoxCPM Unified Suite

This is a fully-featured **VoxCPM** extension node designed specifically for ComfyUI. It not only provides powerful voice generation capabilities but also implements a one-stop, fully automated workflow from "long audio auto-slicing and tagging" to "LoRA model training" right inside ComfyUI.

👨‍💻 **Author's Bilibili**: [Click here to follow me for more tutorials and updates](https://space.bilibili.com/3546882187987924?spm_id_from=333.1007.0.0)

## ✨ Core Features & Advantages

* 🎨 **Extremely Elegant UI**: The node UI supports **seamless switching between English and Chinese**. It uses a clever dynamic design where **parameters automatically show or hide based on your selected model architecture and generation mode** (e.g., selecting "Voice Design" automatically hides the reference audio input). This keeps the interface clean and provides a foolproof, premium experience.

* 🔄 **Full Series Compatibility & Comprehensive Modes**: Perfectly backward compatible with VoxCPM 0.5B, 1.5, and the latest 2.0 series models. Built-in modes include **Voice Design, Ultimate Cloning, Controllable Cloning, Regular Cloning, and Multi-Speaker Dubbing (supports script-level parsing)**.

* 🛠️ **Foolproof Training Data Prep**: Built-in babysitter-level dataset generation node. Just input a long audio file, and the node will automatically use VAD for slicing and filtering, combined with SenseVoice (ASR) and ZipEnhancer for noise reduction and text transcription. **Generate a standard `train.jsonl` with one click**, saying goodbye to tedious manual preprocessing!

* 🚀 **One-Stop LoRA Training Friendly**: Supports resuming training from checkpoints, custom Rank, Learning Rate, and on-demand toggling of LM/DiT/Proj layers fine-tuning. Complex training code is encapsulated into one elegant ComfyUI node.

## 📦 Installation Guide

### 1. Node Installation

Switch your terminal path to the `custom_nodes` directory of your ComfyUI, then clone this repository (or drag the folder directly into it):

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/jieg9341-lab/ComfyUI-VoxCPM-nkxx
```

*Tip: Upon the first run, `__init__.py` will automatically detect and attempt to install the required Python dependencies.*

### 2. Model Download & Placement

To make the nodes work properly, you need to download the corresponding weight files. Please ensure your models are placed in the exact correct directories.

#### 🪟 Windows Users

**Method 1: One-Click Installation Script (Highly Recommended)**
We have prepared a **one-click installation script** for you! Just double-click the `install_models.bat` (One-Click Install Models.bat) located in this plugin's directory. The script will automatically detect if you are using a portable Python environment (like the Aki package), create the directory structure, and pull the full series of VoxCPM models along with the necessary ASR/Denoising auxiliary models, achieving **true one-click deployment**.

**Method 2: Manual Command Line Installation**
If you prefer not to use the script, you can manually open the command prompt (Portable package users should execute this in the provided console/Python environment). Make sure your current command line path is in the **ComfyUI root directory** (the folder containing `main.py`), and run the following commands:

*For users in China, downloading via ModelScope is recommended:*

```cmd
pip install -U modelscope
mkdir models\voxcpm\loras
mkdir models\SenseVoice

modelscope download --model openbmb/VoxCPM2 --local_dir models\voxcpm\VoxCPM2
modelscope download --model openbmb/VoxCPM1.5 --local_dir models\voxcpm\VoxCPM1.5
modelscope download --model openbmb/VoxCPM-0.5B --local_dir models\voxcpm\VoxCPM-0.5B
modelscope download --model iic/SenseVoiceSmall --local_dir models\SenseVoice\SenseVoiceSmall
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir models\voxcpm\speech_zipenhancer_ans_multiloss_16k_base
```

*For overseas users, downloading via HuggingFace is recommended:*

```cmd
huggingface-cli download openbmb/VoxCPM2 --local_dir models\voxcpm\VoxCPM2
huggingface-cli download openbmb/VoxCPM1.5 --local_dir models\voxcpm\VoxCPM1.5
huggingface-cli download openbmb/VoxCPM-0.5B --local_dir models\voxcpm\VoxCPM-0.5B
```

#### 🐧 Linux / Cloud Environment One-Click Deployment (Recommended)

If you are in a new cloud container (like AutoDL), directly copy and paste the entire command block below into your terminal (make sure to execute it in the **ComfyUI root directory**). It will automatically install tools, create directories, and download all models in the correct structure.

```bash
# 1. Install ModelScope
pip install -U modelscope && \
# 2. Create necessary directories
mkdir -p models/voxcpm/loras models/SenseVoice && \
# 3. Download full series of VoxCPM models
echo "Downloading VoxCPM 2.0..." && \
modelscope download --model openbmb/VoxCPM2 --local_dir models/voxcpm/VoxCPM2 && \
echo "Downloading VoxCPM 1.5..." && \
modelscope download --model openbmb/VoxCPM1.5 --local_dir models/voxcpm/VoxCPM1.5 && \
echo "Downloading VoxCPM 0.5B..." && \
modelscope download --model openbmb/VoxCPM-0.5B --local_dir models/voxcpm/VoxCPM-0.5B && \
# 4. Download ASR and Denoising models
echo "Downloading SenseVoiceSmall (ASR)..." && \
modelscope download --model iic/SenseVoiceSmall --local_dir models/SenseVoice/SenseVoiceSmall && \
echo "Downloading ZipEnhancer (Denoising)..." && \
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base && \
echo "✅ All models deployed successfully. Please refresh ComfyUI."
```

**Final Correct Directory Structure:**

```text
ComfyUI/models/
  ├── voxcpm/
  │   ├── VoxCPM2/
  │   ├── VoxCPM1.5/
  │   ├── VoxCPM-0.5B/
  │   ├── speech_zipenhancer_ans_multiloss_16k_base/
  │   └── loras/             <-- Your LoRA training output and loading directory
  └── SenseVoice/
      └── SenseVoiceSmall/
```

## 💡 LoRA Training Guide & VRAM Suggestions

If you want to fine-tune a custom model with your own voice, here are some practical tips:

### 💻 VRAM Requirements Reference

* **VoxCPM 0.5B**: Extremely accessible, takes off with just **6GB VRAM** 🛫.
* **VoxCPM 1.5**: **12GB+ VRAM** recommended for a smooth experience (8GB can barely run it, but it's slow).
* **VoxCPM 2.0**: **20GB VRAM** recommended for training.

### 📊 Training Data & Steps Suggestions

* **Data Volume**: Tests show that preparing about **10 minutes** of clean, single-speaker speech audio (simply auto-sliced using this project's "Dataset Preparer" node) offers the best cost-to-performance ratio.
* **Training Steps**: Usually, training for around **2000 to 3000 steps** yields excellent voice cloning results with extremely high fidelity.

## 🙏 Acknowledgements

This project is deeply inspired by and based on the official [VoxCPM](https://github.com/OpenBMB/VoxCPM) repository. We sincerely thank the OpenBMB team for their outstanding contributions to the open-source voice generation community.

## 📄 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).