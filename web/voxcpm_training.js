import { app } from "../../scripts/app.js";

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function toggleWidget(widget, show, defaultType = "custom") {
    if (!widget) return;

    widget.hidden = !show;

    if (show) {
        if (widget.type === "hidden" || widget.type === "converted-widget") {
            widget.type = widget.origType || defaultType;
        }
        if (widget.origComputeSize !== undefined) {
            widget.computeSize = widget.origComputeSize;
        } else {
            delete widget.computeSize;
        }
        if (widget.inputEl) widget.inputEl.style.display = "";
        if (widget.element) widget.element.style.display = "";
        return;
    }

    if (widget.type !== "hidden" && widget.type !== "converted-widget") {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.type = "hidden";
    }
    widget.computeSize = () => [0, -4];

    if (widget.inputEl) widget.inputEl.style.display = "none";
    if (widget.element) widget.element.style.display = "none";
}

function setWidgetText(widget, labelText) {
    if (!widget) return;

    if (widget._voxcpmOriginalLabel === undefined) {
        widget._voxcpmOriginalLabel = widget.label;
    }
    if (widget._voxcpmOriginalPlaceholder === undefined) {
        widget._voxcpmOriginalPlaceholder = widget.inputEl?.placeholder;
    }

    if (labelText) {
        widget.label = labelText;
        if (widget.inputEl && "placeholder" in widget.inputEl && (widget.name === "来源路径" || widget.name === "输出目录")) {
            widget.inputEl.placeholder = labelText;
        }
        return;
    }

    if (widget._voxcpmOriginalLabel !== undefined) {
        widget.label = widget._voxcpmOriginalLabel;
    }
    if (widget.inputEl && "placeholder" in widget.inputEl && widget._voxcpmOriginalPlaceholder !== undefined) {
        widget.inputEl.placeholder = widget._voxcpmOriginalPlaceholder;
    }
}

// ==================== 翻译字典 ====================

const PREPARER_ZH = {
    "来源模式": "来源模式",
    "来源路径": "来源路径",
    "输出目录": "输出目录",
    "训练清单文件名": "训练清单文件名",
    "递归扫描": "递归扫描",
    "启用参考音频": "启用参考音频",
    "参考音频比例": "参考音频比例",
    "最短片段秒数": "最短片段秒数",
    "最长片段秒数": "最长片段秒数",
    "目标片段秒数": "目标片段秒数",
    "前后保留静音毫秒": "前后保留静音 (ms)",
    "静音阈值分贝": "静音阈值 (dB)",
    "ui_language": "UI 语言",
};

const PREPARER_EN = {
    "来源模式": "Source Mode",
    "来源路径": "Source Path",
    "输出目录": "Output Dir",
    "训练清单文件名": "Manifest Filename",
    "递归扫描": "Recursive Scan",
    "启用参考音频": "Enable Ref Audio",
    "参考音频比例": "Ref Audio Ratio",
    "最短片段秒数": "Min Clip Secs",
    "最长片段秒数": "Max Clip Secs",
    "目标片段秒数": "Target Secs",
    "前后保留静音毫秒": "Padding Silence(ms)",
    "静音阈值分贝": "Silence Thresh(dB)",
    "ui_language": "UI Language",
};

const TRAINER_ZH = {
    "模型": "模型",
    "训练清单": "训练清单",
    "输出名称": "输出名称",
    "总步数": "总步数",
    "保存间隔": "保存间隔",
    "学习率": "学习率",
    "继续训练": "继续训练",
    "show_advanced": "高级选项",
    "ui_language": "UI 语言",
    "LoRA秩": "LoRA 秩 (Rank)",
    "LoRA缩放": "LoRA 缩放 (Alpha)",
    "LoRA丢弃": "LoRA 丢弃 (Dropout)",
    "预热步数": "预热步数",
    "梯度累积": "梯度累积",
    "批量大小": "批量大小",
    "最大批量Token": "最大批量 Token",
    "权重衰减": "权重衰减",
    "数据加载线程": "数据加载线程",
    "启用LM LoRA": "启用 LM LoRA",
    "启用DiT LoRA": "启用 DiT LoRA",
    "启用投影层LoRA": "启用投影层 LoRA",
};

const TRAINER_EN = {
    "模型": "Model",
    "训练清单": "Dataset Manifest",
    "输出名称": "Output Name",
    "总步数": "Total Steps",
    "保存间隔": "Save Every",
    "学习率": "Learning Rate",
    "继续训练": "Resume Training",
    "show_advanced": "Advanced Options",
    "ui_language": "UI Language",
    "LoRA秩": "LoRA Rank",
    "LoRA缩放": "LoRA Alpha",
    "LoRA丢弃": "LoRA Dropout",
    "预热步数": "Warmup Steps",
    "梯度累积": "Grad Accum Steps",
    "批量大小": "Batch Size",
    "最大批量Token": "Max Batch Tokens",
    "权重衰减": "Weight Decay",
    "数据加载线程": "Dataloader Workers",
    "启用LM LoRA": "Enable LM LoRA",
    "启用DiT LoRA": "Enable DiT LoRA",
    "启用投影层LoRA": "Enable Proj LoRA",
};

const PREPARER_MODES = {
    "长音频": { zh: "长音频", en: "Long Audio" },
    "数据集目录": { zh: "数据集目录", en: "Dataset Directory" },
    "批量音频": { zh: "批量音频", en: "Batch Audio" }, // 新增批量音频模式支持
};

// ==================== 数据集准备节点逻辑 ====================

function setupPreparerNode(nodeType, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);

        this._lastMode = null;
        this._lastLang = null;

        // 【关键修复】创建节点时立刻执行一次，防止引擎计算出全展开的臃肿高度
        this.applyVisibility();
        setTimeout(() => this.applyVisibility(), 100);

        setTimeout(() => {
            ["来源模式", "ui_language"].forEach(name => {
                const widget = getWidget(this, name);
                if (widget) {
                    const originalCallback = widget.callback;
                    widget.callback = (value) => {
                        if (originalCallback) originalCallback.apply(widget, [value]);
                        this.applyVisibility(true);
                    };
                }
            });
        }, 150);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        this._voxcpmIsRestored = true;
        if (onConfigure) onConfigure.apply(this, arguments);
        setTimeout(() => this.applyVisibility(), 100);
    };

    nodeType.prototype.applyVisibility = function (interactive = false) {
        if (!this.widgets) return;

        const wMode = getWidget(this, "来源模式");
        const wLang = getWidget(this, "ui_language");
        if (!wMode || !wLang) return;

        const currentLang = wLang.value || "中文";
        const isZh = currentLang === "中文";

        // 将模式转回标准内部标记
        let internalMode = "长音频";
        if (wMode.value === "数据集目录" || wMode.value === "Dataset Directory") {
            internalMode = "数据集目录";
        } else if (wMode.value === "批量音频" || wMode.value === "Batch Audio") { // 处理批量音频
            internalMode = "批量音频";
        }

        if (this._lastMode === internalMode && this._lastLang === currentLang) return;

        const oldMinHeight = this.computeSize()[1];

        this._lastMode = internalMode;
        this._lastLang = currentLang;

        this.title = isZh ? "VoxCPM 数据集准备" : "VoxCPM Dataset Preparer";

        // 添加批量音频到下拉菜单选项中
        wMode.options.values = isZh ? ["长音频", "数据集目录", "批量音频"] : ["Long Audio", "Dataset Directory", "Batch Audio"];
        wMode.value = isZh ? PREPARER_MODES[internalMode].zh : PREPARER_MODES[internalMode].en;

        this.widgets.forEach(widget => {
            const label = isZh ? PREPARER_ZH[widget.name] : PREPARER_EN[widget.name];
            if (label) setWidgetText(widget, label);
        });

        // 长音频模式下隐藏递归扫描，数据集和批量音频显示
        const wRecursive = getWidget(this, "递归扫描");
        const showRecursive = internalMode === "数据集目录" || internalMode === "批量音频";
        toggleWidget(wRecursive, showRecursive, "toggle");

        const newMinHeight = this.computeSize()[1];
        const heightDiff = newMinHeight - oldMinHeight;
        let useHeightDiff = true;
        if (this._voxcpmIsRestored && !interactive) {
            useHeightDiff = false;
        }

        if (useHeightDiff) {
            this.setSize([this.size[0], Math.max(newMinHeight, this.size[1] + heightDiff)]);
        } else {
            this.setSize([this.size[0], Math.max(newMinHeight, this.size[1])]);
        }
        app.graph.setDirtyCanvas(true, true);
    };
}

// ==================== LoRA训练节点逻辑 ====================

function setupTrainerNode(nodeType, app) {
    const onNodeCreated = nodeType.prototype.onNodeCreated;
    nodeType.prototype.onNodeCreated = function () {
        if (onNodeCreated) onNodeCreated.apply(this, arguments);

        this._lastLang = null;
        this._lastAdv = null;

        // 【关键修复】创建节点时立刻执行一次，防止引擎计算出全展开的臃肿高度
        this.applyVisibility();
        setTimeout(() => this.applyVisibility(), 100);

        setTimeout(() => {
            ["show_advanced", "ui_language"].forEach(name => {
                const widget = getWidget(this, name);
                if (widget) {
                    const originalCallback = widget.callback;
                    widget.callback = (value) => {
                        if (originalCallback) originalCallback.apply(widget, [value]);
                        this.applyVisibility(true);
                    };
                }
            });
        }, 150);
    };

    const onConfigure = nodeType.prototype.onConfigure;
    nodeType.prototype.onConfigure = function () {
        this._voxcpmIsRestored = true;
        if (onConfigure) onConfigure.apply(this, arguments);
        setTimeout(() => this.applyVisibility(), 100);
    };

    nodeType.prototype.applyVisibility = function (interactive = false) {
        if (!this.widgets) return;

        const wLang = getWidget(this, "ui_language");
        const wAdv = getWidget(this, "show_advanced");
        if (!wLang || !wAdv) return;

        const currentLang = wLang.value || "中文";
        const isZh = currentLang === "中文";
        const isAdv = wAdv.value === true;

        if (this._lastLang === currentLang && this._lastAdv === isAdv) return;

        const oldMinHeight = this.computeSize()[1];

        this._lastLang = currentLang;
        this._lastAdv = isAdv;

        this.title = isZh ? "VoxCPM LoRA 训练" : "VoxCPM LoRA Trainer";

        this.widgets.forEach(widget => {
            const label = isZh ? TRAINER_ZH[widget.name] : TRAINER_EN[widget.name];
            if (label) setWidgetText(widget, label);
        });

        // 整理高级参数名单并进行动态隐藏
        const advancedNames = [
            "LoRA秩", "LoRA缩放", "LoRA丢弃", "预热步数", "梯度累积",
            "批量大小", "最大批量Token", "权重衰减", "数据加载线程",
            "启用LM LoRA", "启用DiT LoRA", "启用投影层LoRA"
        ];

        const widgetTypes = {
            "LoRA秩": "number", "LoRA缩放": "number", "LoRA丢弃": "number",
            "预热步数": "number", "梯度累积": "number", "批量大小": "number",
            "最大批量Token": "number", "权重衰减": "number", "数据加载线程": "number",
            "启用LM LoRA": "toggle", "启用DiT LoRA": "toggle", "启用投影层LoRA": "toggle"
        };

        advancedNames.forEach(name => {
            const w = getWidget(this, name);
            toggleWidget(w, isAdv, widgetTypes[name]);
        });

        const newMinHeight = this.computeSize()[1];
        const heightDiff = newMinHeight - oldMinHeight;
        let useHeightDiff = true;
        if (this._voxcpmIsRestored && !interactive) {
            useHeightDiff = false;
        }

        if (useHeightDiff) {
            this.setSize([this.size[0], Math.max(newMinHeight, this.size[1] + heightDiff)]);
        } else {
            this.setSize([this.size[0], Math.max(newMinHeight, this.size[1])]);
        }
        app.graph.setDirtyCanvas(true, true);
    };
}

// 统一注册扩展
app.registerExtension({
    name: "VoxCPM.Training.ModelAwareUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name === "voxcpm_nkxx_dataset_preparer") {
            setupPreparerNode(nodeType, app);
        } else if (nodeData.name === "voxcpm_nkxx_lora_trainer") {
            setupTrainerNode(nodeType, app);
        }
    }
});