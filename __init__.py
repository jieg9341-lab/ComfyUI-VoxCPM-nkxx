import importlib
import importlib.util
import re
import subprocess
import sys
from pathlib import Path


WEB_DIRECTORY = "./web"
PLUGIN_ROOT = Path(__file__).resolve().parent
REQUIREMENTS_PATH = PLUGIN_ROOT / "requirements.txt"


def _parse_requirement_spec(line):
    raw_line = str(line or "").strip()
    if not raw_line or raw_line.startswith("#"):
        return None

    raw_line = raw_line.split("#", 1)[0].strip()
    if not raw_line:
        return None

    match = re.match(r"^([A-Za-z0-9_.-]+)", raw_line)
    if not match:
        return None

    package_name = match.group(1)
    import_name = package_name.replace("-", "_")
    return {
        "install_spec": raw_line,
        "package_name": package_name,
        "import_name": import_name,
    }


def _read_requirement_specs():
    if not REQUIREMENTS_PATH.is_file():
        return []

    specs = []
    for line in REQUIREMENTS_PATH.read_text(encoding="utf-8").splitlines():
        parsed = _parse_requirement_spec(line)
        if parsed is not None:
            specs.append(parsed)
    return specs


def _find_missing_dependency_specs(specs):
    missing = []
    for spec in specs:
        if importlib.util.find_spec(spec["import_name"]) is None:
            missing.append(spec)
    return missing


def _format_pip_output(output_text, max_lines=24):
    lines = [line.rstrip() for line in str(output_text or "").splitlines() if line.strip()]
    if len(lines) <= max_lines:
        return "\n".join(lines)
    return "\n".join(lines[-max_lines:])


def _install_missing_dependencies(missing_specs):
    install_targets = [spec["install_spec"] for spec in missing_specs]
    display_names = ", ".join(spec["package_name"] for spec in missing_specs)

    print(f"[VoxCPM][依赖] 检测到缺少依赖，开始自动安装: {display_names}")

    command = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "--disable-pip-version-check",
        *install_targets,
    ]
    result = subprocess.run(
        command,
        cwd=str(PLUGIN_ROOT),
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
        check=False,
    )

    importlib.invalidate_caches()
    still_missing = _find_missing_dependency_specs(missing_specs)

    if result.returncode == 0 and not still_missing:
        print(f"[VoxCPM][依赖] 依赖安装完成: {display_names}")
        return True

    print(f"[VoxCPM][依赖] 自动安装失败: {display_names}")
    pip_output = _format_pip_output("\n".join(filter(None, [result.stdout, result.stderr])))
    if pip_output:
        print("[VoxCPM][依赖] pip 输出:")
        print(pip_output)
    if still_missing:
        print("[VoxCPM][依赖] 仍缺少依赖: " + ", ".join(spec["package_name"] for spec in still_missing))
    print("[VoxCPM][依赖] 请在当前 ComfyUI 的 Python 环境里手动执行 requirements.txt 安装。")
    return False


def _ensure_plugin_dependencies():
    requirement_specs = _read_requirement_specs()
    if not requirement_specs:
        return True

    missing_specs = _find_missing_dependency_specs(requirement_specs)
    if not missing_specs:
        return True

    return _install_missing_dependencies(missing_specs)


if _ensure_plugin_dependencies():
    from .nodes.training_nodes import VoxCPM_Dataset_Preparer
    from .nodes.training_nodes import VoxCPM_Lora_Trainer
    from .nodes.unified_generate import VoxCPM_Unified_Generator

    NODE_CLASS_MAPPINGS = {
        "voxcpm_nkxx_unified_generator": VoxCPM_Unified_Generator,
        "voxcpm_nkxx_dataset_preparer": VoxCPM_Dataset_Preparer,
        "voxcpm_nkxx_lora_trainer": VoxCPM_Lora_Trainer,
    }

    NODE_DISPLAY_NAME_MAPPINGS = {
        "voxcpm_nkxx_unified_generator": "VoxCPM 全能生成",
        "voxcpm_nkxx_dataset_preparer": "VoxCPM 数据集准备",
        "voxcpm_nkxx_lora_trainer": "VoxCPM LoRA 训练",
    }
else:
    NODE_CLASS_MAPPINGS = {}
    NODE_DISPLAY_NAME_MAPPINGS = {}


__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
