import { app } from "../../scripts/app.js";

const MODE_TRANSLATIONS = {
    "声音设计": { zh: "声音设计", en: "Voice Design" },
    "极致克隆": { zh: "极致克隆", en: "Ultimate Cloning" },
    "可控克隆": { zh: "可控克隆", en: "Controllable Cloning" },
    "常规克隆": { zh: "常规克隆", en: "Regular Cloning" },
    "多人配音": { zh: "多人配音", en: "Multi-Speaker Dubbing" },
};

const MODE_ALIASES = {
    "Voice Design": "声音设计",
    "Ultimate Cloning": "极致克隆",
    "Controllable Cloning": "可控克隆",
    "Regular Cloning": "常规克隆",
    "Multi-Speaker Dubbing": "多人配音",
    "Multi-Speaker Dialog": "多人配音",
    "多角色对话": "多人配音",
};

const V2_MODES_ZH = ["声音设计", "极致克隆", "可控克隆", "多人配音"];
const LEGACY_MODES_ZH = ["常规克隆", "多人配音"];

function getWidget(node, name) {
    return node.widgets?.find((widget) => widget.name === name);
}

function parseModelProfiles(rawValue) {
    try {
        return JSON.parse(rawValue || "{}");
    } catch {
        return {};
    }
}

function normalizeMode(modeValue) {
    return MODE_ALIASES[modeValue] || modeValue || "声音设计";
}

function toDisplayMode(modeZh, isZh) {
    const translation = MODE_TRANSLATIONS[modeZh] || MODE_TRANSLATIONS["声音设计"];
    return isZh ? translation.zh : translation.en;
}

function inferModelArchitecture(modelName, modelProfiles) {
    const profile = modelProfiles?.[modelName];
    if (profile?.architecture === "voxcpm2" || profile?.architecture === "voxcpm") {
        return profile.architecture;
    }

    const normalizedName = String(modelName || "").toLowerCase();
    if (normalizedName.includes("voxcpm2")) {
        return "voxcpm2";
    }
    if (
        normalizedName.includes("voxcpm1.5") ||
        normalizedName.includes("voxcpm-0.5b") ||
        normalizedName.includes("voxcpm_0.5b") ||
        normalizedName.includes("0.5b") ||
        normalizedName.includes("1.5")
    ) {
        return "voxcpm";
    }
    return "voxcpm2";
}

function getSupportedModes(modelArch) {
    return modelArch === "voxcpm2" ? V2_MODES_ZH : LEGACY_MODES_ZH;
}

function coerceModeForModel(modeZh, modelArch) {
    if (modelArch === "voxcpm2") {
        if (modeZh === "常规克隆") {
            return "极致克隆";
        }
        return V2_MODES_ZH.includes(modeZh) ? modeZh : "声音设计";
    }

    if (modeZh === "声音设计" || modeZh === "极致克隆" || modeZh === "可控克隆") {
        return "常规克隆";
    }
    return LEGACY_MODES_ZH.includes(modeZh) ? modeZh : "常规克隆";
}

function getModeOptions(modelArch, isZh) {
    return getSupportedModes(modelArch).map((modeZh) => toDisplayMode(modeZh, isZh));
}

// 恢复为最基础稳定的显隐逻辑，不再干预输入类型和进行 Hack 伪装
function toggleWidget(widget, show, defaultType = "custom") {
    if (!widget) {
        return;
    }

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

        if (widget.inputEl) {
            widget.inputEl.style.display = "";
        }
        if (widget.element) {
            widget.element.style.display = "";
        }
        return;
    }

    if (widget.type !== "hidden" && widget.type !== "converted-widget") {
        widget.origType = widget.type;
        widget.origComputeSize = widget.computeSize;
        widget.type = "hidden";
    }
    widget.computeSize = () => [0, -4];

    if (widget.inputEl) {
        widget.inputEl.style.display = "none";
    }
    if (widget.element) {
        widget.element.style.display = "none";
    }
}

function setWidgetText(widget, labelText) {
    if (!widget) {
        return;
    }

    if (widget._voxcpmOriginalLabel === undefined) {
        widget._voxcpmOriginalLabel = widget.label;
    }
    if (widget._voxcpmOriginalPlaceholder === undefined) {
        widget._voxcpmOriginalPlaceholder = widget.inputEl?.placeholder;
    }

    if (labelText) {
        widget.label = labelText;
        if (widget.inputEl && "placeholder" in widget.inputEl && (widget.name === "target_text" || widget.name === "control_instruction" || widget.name === "reference_text")) {
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

app.registerExtension({
    name: "VoxCPM.Unified.Generator.V6.ModelAwareUI",
    async beforeRegisterNodeDef(nodeType, nodeData, app) {
        if (nodeData.name !== "voxcpm_nkxx_unified_generator") {
            return;
        }

        const onNodeCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            if (onNodeCreated) {
                onNodeCreated.apply(this, arguments);
            }

            this._lastMode = null;
            this._lastSpkCount = null;
            this._lastAutoAsr = null;
            this._lastAdv = null;
            this._lastLang = null;
            this._lastModelName = null;
            this._lastModelArch = null;

            this.applyVisibility();
            setTimeout(() => this.applyVisibility(), 100);

            setTimeout(() => {
                const watchedNames = [
                    "model_name",
                    "work_mode",
                    "speaker_count",
                    "show_advanced",
                    "ui_language",
                    "auto_asr",
                ];

                watchedNames.forEach((name) => {
                    const widget = getWidget(this, name);
                    if (!widget) {
                        return;
                    }
                    const originalCallback = widget.callback;
                    widget.callback = (value) => {
                        if (originalCallback) {
                            originalCallback.apply(widget, [value]);
                        }
                        this.applyVisibility(true);
                    };
                });
            }, 150);
        };

        const onConfigure = nodeType.prototype.onConfigure;
        nodeType.prototype.onConfigure = function () {
            this._voxcpmIsRestored = true;
            if (onConfigure) {
                onConfigure.apply(this, arguments);
            }
            setTimeout(() => this.applyVisibility(), 100);
        };

        nodeType.prototype.applyVisibility = function (interactive) {
            if (interactive !== true) interactive = false;
            if (!this.widgets) {
                return;
            }

            const wModel = getWidget(this, "model_name");
            const wMode = getWidget(this, "work_mode");
            const wSpkCount = getWidget(this, "speaker_count");
            const wAutoAsr = getWidget(this, "auto_asr");
            const wShowAdv = getWidget(this, "show_advanced");
            const wLang = getWidget(this, "ui_language");
            const wModelProfiles = getWidget(this, "model_profiles_json");

            if (!wMode) {
                return;
            }

            const modelProfiles = parseModelProfiles(wModelProfiles?.value);
            const modelName = wModel ? wModel.value : "";
            const modelArch = inferModelArchitecture(modelName, modelProfiles);
            
            let currentSpkCount = 2;
            if (wSpkCount) {
                currentSpkCount = parseInt(wSpkCount.value, 10);
                if (isNaN(currentSpkCount) || currentSpkCount < 1) {
                    currentSpkCount = 2;
                }
            }

            const autoAsrVal = wAutoAsr ? wAutoAsr.value : true;
            const isAdv = wShowAdv ? wShowAdv.value : false;
            const currentLang = wLang ? wLang.value : "中文";
            const isZh = currentLang === "中文";

            const normalizedMode = normalizeMode(wMode.value);
            const mappedMode = coerceModeForModel(normalizedMode, modelArch);

            if (
                this._lastMode === mappedMode &&
                this._lastSpkCount === currentSpkCount &&
                this._lastAutoAsr === autoAsrVal &&
                this._lastAdv === isAdv &&
                this._lastLang === currentLang &&
                this._lastModelName === modelName &&
                this._lastModelArch === modelArch
            ) {
                return;
            }

            const oldMinHeight = this.computeSize()[1];

            this._lastMode = mappedMode;
            this._lastSpkCount = currentSpkCount;
            this._lastAutoAsr = autoAsrVal;
            this._lastAdv = isAdv;
            this._lastLang = currentLang;
            this._lastModelName = modelName;
            this._lastModelArch = modelArch;

            wMode.options.values = getModeOptions(modelArch, isZh);
            wMode.value = toDisplayMode(mappedMode, isZh);

            this.title = isZh ? "VoxCPM 全能生成" : "VoxCPM Unified";

            const ZH_LABELS = {
                "model_name": "模型",
                "work_mode": "模式",
                "control_instruction": "控制指令",
                "target_text": "目标文本",
                "show_advanced": "高级选项",
                "ui_language": "UI 语言",
                "cfg_value": "CFG 强度",
                "inference_steps": "推理步数",
                "seed": "随机种子",
                "control_after_generate": "生成前控制",
                "speaker_count": "配音人数",
                "lora_name": "LoRA",
                "force_offload": "强制释放显存",
                "auto_asr": "自动识别参考文本",
                "reference_text": "参考文本",
                "denoise_reference": "参考音频降噪",
                "normalize_text": "文本规范化",
                "normalize_loudness": "响度标准化",
            };

            const EN_LABELS = {
                "model_name": "Model",
                "work_mode": "Mode",
                "control_instruction": "Control",
                "target_text": "Text",
                "show_advanced": "Advanced",
                "ui_language": "UI Language",
                "cfg_value": "CFG",
                "inference_steps": "Steps",
                "seed": "Seed",
                "control_after_generate": "Control After Gen",
                "speaker_count": "Cast Count",
                "lora_name": "LoRA",
                "force_offload": "Force Offload",
                "auto_asr": "Auto ASR",
                "reference_text": "Reference Text",
                "denoise_reference": "Denoise Ref Audio",
                "normalize_text": "Normalize Text",
                "normalize_loudness": "Normalize Loudness",
            };

            this.widgets.forEach((widget) => {
                if (widget.name === "model_profiles_json") {
                    return;
                }
                const translatedLabel = isZh ? ZH_LABELS[widget.name] : EN_LABELS[widget.name];
                setWidgetText(widget, translatedLabel);
            });

            const wControlInstr = getWidget(this, "control_instruction");
            const wTarget = getWidget(this, "target_text");
            const wRefText = getWidget(this, "reference_text");
            const wDenoise = getWidget(this, "denoise_reference");
            const wCfg = getWidget(this, "cfg_value");
            const wSteps = getWidget(this, "inference_steps");
            const wSeed = getWidget(this, "seed");
            const wControlAfterGen = getWidget(this, "control_after_generate");
            const wLora = getWidget(this, "lora_name");
            const wOffload = getWidget(this, "force_offload");
            const wNormText = getWidget(this, "normalize_text");
            const wNorm = getWidget(this, "normalize_loudness");

            const isMulti = mappedMode === "多人配音";
            const needsRef = mappedMode === "极致克隆" || mappedMode === "可控克隆" || mappedMode === "常规克隆";
            const needsPromptText = mappedMode === "极致克隆" || mappedMode === "常规克隆";
            const usesControl = modelArch === "voxcpm2" && (mappedMode === "声音设计" || mappedMode === "可控克隆");

            toggleWidget(wModelProfiles, false, "hidden");
            toggleWidget(wControlInstr, usesControl, "customtext");
            toggleWidget(wSpkCount, isMulti, "number");

            toggleWidget(wLang, isAdv, "combo");
            toggleWidget(wCfg, isAdv, "number");
            toggleWidget(wSteps, isAdv, "number");
            toggleWidget(wSeed, isAdv, "number");
            toggleWidget(wControlAfterGen, isAdv, "combo");
            toggleWidget(wLora, isAdv, "combo");
            toggleWidget(wOffload, isAdv, "toggle");
            toggleWidget(wNormText, isAdv, "toggle");
            toggleWidget(wNorm, isAdv, "toggle");

            toggleWidget(wAutoAsr, isAdv && !isMulti && needsPromptText, "toggle");
            toggleWidget(wRefText, isAdv && !isMulti && needsPromptText && !autoAsrVal, "customtext");
            toggleWidget(wDenoise, isAdv && (needsRef || isMulti), "toggle");

            const targetInputs = [];
            if (isMulti) {
                for (let i = 1; i <= currentSpkCount; i += 1) {
                    targetInputs.push(`audio_${i}`);
                }
            } else if (needsRef) {
                targetInputs.push("reference_audio");
            }

            const currentInputs = this.inputs ? this.inputs.map((input) => input.name) : [];
            for (let i = currentInputs.length - 1; i >= 0; i -= 1) {
                const name = currentInputs[i];
                if ((name.startsWith("audio_") || name === "reference_audio") && !targetInputs.includes(name)) {
                    this.removeInput(i);
                }
            }

            targetInputs.forEach((name) => {
                if (!this.inputs?.some((input) => input.name === name)) {
                    this.addInput(name, "AUDIO");
                }
            });

            if (this.inputs) {
                this.inputs.forEach((input) => {
                    if (input.name === "reference_audio") {
                        input.label = isZh ? "参考音频" : "Reference Audio";
                    } else if (input.name.startsWith("audio_")) {
                        const speakerIndex = input.name.split("_")[1];
                        input.label = isZh ? `角色音频 ${speakerIndex}` : `Speaker ${speakerIndex} Audio`;
                    }
                });
            }

            const newMinHeight = this.computeSize()[1];
            const heightDiff = newMinHeight - oldMinHeight;

            let useHeightDiff = true;
            if (this._voxcpmIsRestored && !interactive) {
                useHeightDiff = false;
            }

            if (useHeightDiff) {
                this.setSize([this.size[0], Math.max(newMinHeight, this.size[1] + heightDiff)]);
            } else {
                // 【修复】：在初次加载/切换工作流时，不仅托底最小高度，同时尊重并保留保存下来的更高尺寸
                this.setSize([this.size[0], Math.max(newMinHeight, this.size[1])]);
            }

            app.graph.setDirtyCanvas(true, true);
        };

        const onDrawBackground = nodeType.prototype.onDrawBackground;
        nodeType.prototype.onDrawBackground = function (ctx) {
            if (onDrawBackground) {
                onDrawBackground.apply(this, arguments);
            }

            // 恢复最原始的稳定检测逻辑，去除了所有画板操作检测
            const autoAsr = getWidget(this, "auto_asr");
            if (autoAsr && this._lastAutoAsr !== autoAsr.value) {
                this.applyVisibility(true);
            }
        };
    },
});