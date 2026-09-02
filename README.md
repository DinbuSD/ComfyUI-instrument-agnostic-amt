# ComfyUI-instrument-agnostic-amt

中文 | [English](README_EN.md)

将任意乐器音频（钢琴、吉他、贝斯、人声、弦乐、合成器等）转录为 MIDI 的 ComfyUI 节点集，基于 [instrument-agnostic-amt](https://github.com/anime-song/instrument-agnostic-amt)（Neural Semi-CRF 架构，MIT 许可，源码已 vendor 在本节点内）。上游项目已于 2026-09 更名为 **tsumugi**（代码包名 `instrument_agnostic_amt` 不变）。

## 功能

- **音频 → MIDI 转录**：单轨 / 多模型联合 / 分轨工作流
- **音轨分离**：BS-RoFormer 分离 6 轨（vocals / guitar / bass / drums / piano / other）
- **后处理**：力度预测、乐器精修、节拍/和弦/调性
- **进度条**：转写、分离、力度、精修节点均带节点级进度条
- **模型自动下载**：缺失的模型在执行时自动拉取（官方 HF → hf-mirror 回退）

## 节点一览（分类 Audio/AMT）

| 节点 | 输入 → 输出 | 说明 |
|---|---|---|
| **Instrument Agnostic Amt** | AUDIO → MIDI / note_count / duration | 音频转 MIDI（精简参数：model + 静音检测 + 精度 + 设备）；model 缺失自动下载 |
| **Instrument Agnostic Amt Advanced** | AUDIO → MIDI / note_count / duration | 同上，另暴露全部高级参数（window/stride/merge/镲合并） |
| **Stem Separate** | AUDIO → 6×AUDIO | 音轨分离；`save_outputs` 开关可保存分离产物到 output/stems/ |
| **Merge MIDI** | MIDI×8 → MIDI | 多轨 MIDI 合并（乐器按 program/名称归并） |
| **Predict Velocity** | MIDI + AUDIO → MIDI | 从分离音频预测音符力度（stem_name 下拉） |
| **Refine Instrument** | MIDI + AUDIO → MIDI | 按音色重分配乐器标签（mode: cluster/single） |
| **Beat Chord Key** | MIDI → MIDI | 预测节拍/和弦/调性并写回；`fix_leading_tempo` 默认开，修正开头空白段导致的虚高初始 tempo |
| **Save MIDI** | MIDI → 文件 | 保存到 output/midi/（SaveImage 风格命名） |

## 安装

要求：新版 ComfyUI（含内置 LoadAudio / AUDIO 类型）。

```bash
# 1. 在 custom_nodes 目录下克隆本仓库（将 <你的ComfyUI路径> 换成实际安装位置）
cd <你的ComfyUI路径>/custom_nodes
git clone https://github.com/DinbuSD/ComfyUI-instrument-agnostic-amt

# 2. 安装依赖（使用 ComfyUI 的 Python 环境，在节点目录下执行）
#    portable 版:
ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt

# 3. 重启 ComfyUI
```

模型**无需手动下载**：首次执行节点时自动从 Hugging Face 拉取（先官方，失败回退 hf-mirror）。全部模型约 1GB，缓存于 `ComfyUI/models/instrument_agnostic_amt/`。

网络受限时可强制指定下载端点：

```bash
set AMT_HF_ENDPOINT=https://hf-mirror.com   # Windows
export AMT_HF_ENDPOINT=https://hf-mirror.com # Linux/macOS
```

## 快速开始

`example_workflows/` 目录内提供可直接拖入 ComfyUI 画布的工作流模板：

- **simple-transcription.json**：简易单轨工作流（LoadAudio → 转写 → 保存）
- **stem-separation.json**：单纯音轨分离（LoadAudio → 分离 → 6 轨存 FLAC 到 output/stems/<轨名>/）
- **stem-separated-transcription.json**：完整分轨工作流（分离 → 6 轨转写 → guitar/bass/piano/other 精修 → 6 轨力度 → 合并 → 节拍/和弦 → 保存）

**单轨转录**：

```
LoadAudio(音频) → Instrument Agnostic Amt(model=default) → Save MIDI
```

**完整分轨工作流**（官方 Colab 同款；官方说明：通常比整曲直接转写效果更好，尤其密集编曲）：

```
LoadAudio
  → Stem Separate
      ├─ vocals → Amt(vocal_harmony_v1_5) → Velocity ─┐
      ├─ guitar → Amt(guitar_v1_5) → Refine → Velocity ─┤
      ├─ bass   → Amt(bass_v2)     → Refine → Velocity ─┼→ Merge MIDI → Beat Chord Key → Save MIDI
      ├─ drums  → Amt(drums_v1_5)  → Velocity ─┤
      ├─ piano  → Amt(default)     → Refine → Velocity ─┤
      └─ other  → Amt(other_v1_5)  → Refine → Velocity ─┘
```

> vocals / drums 不做乐器精修（上游 `REFINEMENT_EXCLUDED_STEM_GROUPS` 官方设定，鼓类候选与角色类轨道不适合重标）；guitar / bass / piano / other 精修后接力度预测，与官方管线一致。

可选后处理：每轨 `Predict Velocity`（需要该轨音频）；guitar / bass / piano / other 建议接 `Refine Instrument`（分类易乱的轨道）；`Merge MIDI` 最多 8 路输入。

## 模型清单（自动下载）

| 文件 | 用途 | 大小 |
|---|---|---|
| best_model.pth | 通用全乐器 | 53.5 MB |
| best_model_bass_v2.pth | 贝斯 v2 | 54.5 MB |
| best_model_vocal.pth | 人声 | 53.5 MB |
| best_model_guitar_v1_5.pth | 吉他 v1.5 | 54.5 MB |
| best_model_vocal_harmony.pth | 人声和声 | 54.5 MB |
| best_model_vocal_harmony_v1_5.pth | 人声和声 v1.5（分轨推荐） | 54.5 MB |
| best_model_drums.pth | 鼓（实验性） | 54.5 MB |
| best_model_drums_v1_5.pth | 鼓 v1.5（分轨推荐） | 54.5 MB |
| best_model_other.pth | 其他乐器 | 54.5 MB |
| best_model_other_v1_5.pth | 其他乐器 v1.5（分轨推荐） | 54.5 MB |
| best_velocity_model.pth | 力度预测 | 55.0 MB |
| best_instrument_refinement.pth | 乐器精修 | 56.2 MB |
| best_beat_chord_key.pth | 节拍/和弦/调性 | 86.9 MB |
| stem_splitter.pt | 音轨分离（BS-RoFormer） | 350 MB |

> 2026-09 上游（tsumugi）起，分轨工作流对 vocals / drums / other 改用 `*_v1_5` 新模型（`vocal_harmony_v1_5` / `drums_v1_5` / `other_v1_5`）；旧模型保留可选。`*_v1_5` 缺失时执行中自动下载。

> **转写节点新增参数**：`collapse_crash_cymbals`（默认开）——把鼓的 Crash Cymbal 2（pitch 57）合并到 Crash Cymbal 1（pitch 49），对齐上游 2026-09 起的默认行为；关掉则保留两个镲独立输出。

## 已知限制

- **drums 模型为 Experimental**：鼓点完整度有限，官方标注精度会随模型演进变化
- **分轨转写时间偏移**：分离 stem 各自独立转写可能引入微小时间偏移（官方已知问题），合并后可在 DAW 中整轨对齐
- **人声轨道音色**：vocal 模型输出 `melody` 角色标签，MIDI 播放时映射为旋律性乐器（管乐类）——音高/节奏正确，音色可在 DAW 中改为 choir（GM 52-54）
- **单声道输入**：转录/分离要求双声道输入，单声道音频会自动复制为双声道

## 许可

节点代码为 MIT 许可。vendor 的 `instrument_agnostic_amt/` 目录来自 [anime-song/instrument-agnostic-amt](https://github.com/anime-song/instrument-agnostic-amt)（2026-09 起更名 [tsumugi](https://github.com/anime-song/tsumugi)，MIT 许可，Copyright (c) 2026 [anime-song](https://github.com/anime-song)，vendor 说明见 `instrument_agnostic_amt/VENDORED_README.txt`）。模型权重与分离器权重版权归原作者（anime-song）。

## 开发声明

本节点使用 DeepSeek-V4-Flash（模型）与 DeepSeek Harness（前端）开发。
