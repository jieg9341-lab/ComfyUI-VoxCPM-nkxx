@echo off
chcp 65001 >nul
title VoxCPM 模型一键安装工具
color 0A

echo =======================================================
echo          VoxCPM 模型一键安装 (极简纯净版)
echo =======================================================
echo.

:: 1. 自动定位你的 ComfyUI models 文件夹
:: 逻辑：从当前插件目录向上退两层，再进入 models 文件夹
set "NODE_DIR=%~dp0"
set "MODELS_DIR=%NODE_DIR%..\..\models"
echo [状态] 已精准定位模型文件夹：
echo %MODELS_DIR%
echo.

:: 2. 寻找 Python 环境 (优先找秋叶包或官方整合包)
set "PYTHON_EXE=python"
set "PIP_EXE=pip"
set "MODELSCOPE_EXE=modelscope"

if exist "%NODE_DIR%..\..\..\python_embeded\python.exe" (
    echo [状态] 检测到便携版独立环境，优先使用！
    set "PYTHON_EXE=%NODE_DIR%..\..\..\python_embeded\python.exe"
    :: 直接调用独立环境的工具
    set "PIP_EXE="%NODE_DIR%..\..\..\python_embeded\python.exe" -m pip"
    set "MODELSCOPE_EXE="%NODE_DIR%..\..\..\python_embeded\python.exe" -m modelscope"
) else (
    echo [状态] 未检测到独立环境，使用系统全局环境。
)
echo.

:: 3. 安装下载工具
echo [1/3] 正在准备魔搭 (ModelScope) 下载环境...
%PIP_EXE% install -U modelscope
echo.

:: 4. 创建必要的空文件夹
echo [2/3] 正在创建文件夹结构...
if not exist "%MODELS_DIR%\voxcpm\loras" mkdir "%MODELS_DIR%\voxcpm\loras"
if not exist "%MODELS_DIR%\SenseVoice" mkdir "%MODELS_DIR%\SenseVoice"
echo 文件夹就绪。
echo.

:: 5. 开始直接下载
echo [3/3] 开始下载模型，文件较大请耐心等待...
echo.

echo ---- 正在下载 VoxCPM 2.0 ----
%MODELSCOPE_EXE% download --model openbmb/VoxCPM2 --local_dir "%MODELS_DIR%\voxcpm\VoxCPM2"

echo ---- 正在下载 VoxCPM 1.5 ----
%MODELSCOPE_EXE% download --model openbmb/VoxCPM1.5 --local_dir "%MODELS_DIR%\voxcpm\VoxCPM1.5"

echo ---- 正在下载 VoxCPM 0.5B ----
%MODELSCOPE_EXE% download --model openbmb/VoxCPM-0.5B --local_dir "%MODELS_DIR%\voxcpm\VoxCPM-0.5B"

echo ---- 正在下载 SenseVoice (ASR) ----
%MODELSCOPE_EXE% download --model iic/SenseVoiceSmall --local_dir "%MODELS_DIR%\SenseVoice\SenseVoiceSmall"

echo ---- 正在下载 ZipEnhancer (降噪模型) ----
%MODELSCOPE_EXE% download --model iic/speech_zipenhancer_ans_multiloss_16k_base --local_dir "%MODELS_DIR%\voxcpm\speech_zipenhancer_ans_multiloss_16k_base"

echo.
echo =======================================================
echo ✅ 所有模型下载完成！请重新启动 ComfyUI。
echo =======================================================
pause