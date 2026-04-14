# ComfyUI VoxCPM 全能语音与训练套件 (ComfyUI VoxCPM Unified Suite)

这是一个专为 ComfyUI 设计的 **VoxCPM** 全功能扩展节点。它不仅提供强大的语音生成功能，还在 ComfyUI 中实现了从“长音频自动切片打标”到“LoRA 模型训练”的一站式全自动工作流。

👨‍💻 **作者 B站主页**：[点击这里关注我，获取更多教程和更新](https://space.bilibili.com/3546882187987924?spm_id_from=333.1007.0.0)

## ✨ 核心优势与功能

* 🎨 **极其优雅的交互体验**：节点 UI 支持**中英文无缝切换**。采用了巧妙的动态设计，**参数会根据你选择的模型架构和生成模式自动显示或隐藏**（例如，选择“声音设计”模式会自动隐藏参考音频接口），让界面始终保持清爽，提供傻瓜式的高级体验。

* 🔄 **全系列模型兼容与全模式支持**：完美向下兼容 VoxCPM 0.5B、1.5，以及最新的voxcpm2模型。内置 **声音设计、极致克隆、可控克隆、常规克隆、多人配音（支持剧本级解析）** 等全套生成模式。

* 🛠️ **傻瓜式训练数据准备**：内置保姆级的数据集生成节点。只需输入一段长音频，节点会自动利用 VAD 进行切片过滤，并结合 SenseVoice (ASR) 和 ZipEnhancer 进行降噪和文字打标，**一键生成符合规范的 `train.jsonl`**，彻底告别繁琐的手动预处理！

* 🚀 **一站式 LoRA 训练友好**：支持断点续训，支持自定义秩、学习率，且可按需开启 LM/DiT/Proj 层的微调。将复杂的训练代码封装为了 ComfyUI 中优雅的一个节点。

## 📦 安装指南

### 1. 安装节点 (Node Installation)

请将终端路径切换到你的 ComfyUI 的 `custom_nodes` 目录下，然后克隆本仓库（或直接将文件夹拖入该目录）：

```bash
cd ComfyUI/custom_nodes
git clone <你的GitHub仓库链接>
```

*提示：初次运行节点时，`__init__.py` 会自动检测并尝试安装所需的 Python 依赖。*

### 2. 模型下载与存放 (Model Installation)

要使节点正常工作，你需要下载相应的权重文件。请确保你的模型存放在正确的目录下。

#### 🪟 Windows 用户

**方法一：一键安装脚本（强烈推荐）**
我们为你准备了**一键安装脚本**！只需双击运行本插件目录下的 `一键安装模型.bat`。脚本会自动识别你是否使用的是便携版 Python 环境（如秋叶整合包），并自动为你创建目录结构、拉取全系列 VoxCPM 模型以及所需的 ASR/降噪辅助模型，实现**真正的一键部署**。

**方法二：手动命令行安装**
如果你不想使用一键脚本，也可以手动打开命令提示符（便携包用户请在整合包提供的控制台/Python 环境下执行）。请确保命令行当前路径在 **ComfyUI 的根目录**（即包含 `main.py` 的文件夹）下，执行以下命令：

*国内推荐使用 ModelScope (魔搭) 下载：*

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

*海外推荐使用 HuggingFace 下载：*

```cmd
huggingface-cli download openbmb/VoxCPM2 --local_dir models\voxcpm\VoxCPM2
huggingface-cli download openbmb/VoxCPM1.5 --local_dir models\voxcpm\VoxCPM1.5
huggingface-cli download openbmb/VoxCPM-0.5B --local_dir models\voxcpm\VoxCPM-0.5B
```

#### 🐧 Linux / 云端环境一键部署（推荐）

如果你在新的云端容器中（如 AutoDL），直接复制并粘贴以下整段命令到终端（请确保在 **ComfyUI 根目录**下执行）。它会自动安装工具、创建目录并按正确结构下载所有模型。

```bash
# 1. 安装魔搭工具
pip install -U modelscope && \
# 2. 创建必要的存放目录
mkdir -p models/voxcpm/loras models/SenseVoice && \
# 3. 下载 VoxCPM 全系列模型
echo "正在下载 VoxCPM 2.0..." && \
modelscope download --model openbmb/VoxCPM2 --local_dir models/voxcpm/VoxCPM2 && \
echo "正在下载 VoxCPM 1.5..." && \
modelscope download --model openbmb/VoxCPM1.5 --local_dir models/voxcpm/VoxCPM1.5 && \
echo "正在下载 VoxCPM 0.5B..." && \
modelscope download --model openbmb/VoxCPM-0.5B --local_dir models/voxcpm/VoxCPM-0.5B && \
# 4. 下载 ASR 与 降噪模型
echo "正在下载 SenseVoiceSmall (ASR)..." && \
modelscope download --model iic/SenseVoiceSmall --local_dir models/SenseVoice/SenseVoiceSmall && \
echo "正在下载 ZipEnhancer (降噪)..." && \
modelscope download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir models/voxcpm/speech_zipenhancer_ans_multiloss_16k_base && \
echo "✅ 所有模型已部署完成，请刷新 ComfyUI。"
```

最终正确的目录结构应如下所示：

```text
ComfyUI/models/
  ├── voxcpm/
  │   ├── VoxCPM2/
  │   ├── VoxCPM1.5/
  │   ├── VoxCPM-0.5B/
  │   ├── speech_zipenhancer_ans_multiloss_16k_base/
  │   └── loras/             <-- 你的 LoRA 训练输出和加载目录
  └── SenseVoice/
      └── SenseVoiceSmall/
```

## 💡 LoRA 训练指南 & 显存建议

如果你想用自己的声音微调一个专属模型，这里有一些实战总结的建议：

### 💻 显存 (VRAM) 需求参考

* **VoxCPM 0.5B**: 极其亲民，**6GB 显存**即可起飞 🛫。
* **VoxCPM 1.5**: 建议 **12GB 以上显存**以获得良好体验（8GB 勉强能跑，但速度较慢）。
* **VoxCPM 2.0**: 建议准备 **20GB 显存**进行训练。

### 📊 训练数据与步数建议

* **数据量**：实测准备大约 **10分钟** 的纯净单人讲话音频（通过本项目的“数据集准备节点”自动切片即可）是最优性价比的选择。
* **训练步数**：通常情况下，训练 **2000 到 3000 步**左右，即可获得非常出色且还原度极高的声音克隆效果。

## 🙏 Acknowledgements

This project is deeply inspired by and based on the official [VoxCPM](https://github.com/OpenBMB/VoxCPM) repository. We sincerely thank the OpenBMB team for their outstanding contributions to the open-source voice generation community.

## 📄 License

This project is licensed under the [Apache License 2.0](https://www.apache.org/licenses/LICENSE-2.0).