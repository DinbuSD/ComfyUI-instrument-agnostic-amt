# ComfyUI-instrument-agnostic-amt

[中文](README.md) | English

A ComfyUI node pack for transcribing arbitrary instrument audio (piano, guitar, bass, vocals, strings, synthesizers, etc.) into MIDI, based on [instrument-agnostic-amt](https://github.com/anime-song/instrument-agnostic-amt) (Neural Semi-CRF architecture, MIT licensed, source vendored inside this node pack).

## Features

- **Audio → MIDI transcription**: single-track, multi-model ensemble, or per-stem workflows
- **Stem separation**: BS-RoFormer separates 6 stems (vocals / guitar / bass / drums / piano / other)
- **Post-processing**: velocity prediction, instrument refinement, beat / chord / key estimation
- **Progress bars**: node-level progress bars on transcription, separation, velocity, and refinement nodes
- **Automatic model download**: missing models are fetched on the fly at execution time (official HF → hf-mirror fallback)

## Node Overview (Category Audio/AMT)

| Node | Input → Output | Description |
|---|---|---|
| **Instrument Agnostic Amt** | AUDIO → MIDI / note_count / duration | Audio to MIDI transcription; pick a model from the dropdown (auto-downloads when missing) |
| **Stem Separate** | AUDIO → 6×AUDIO | Stem separation; the `save_outputs` toggle saves separated stems to output/stems/ |
| **Merge MIDI** | MIDI×8 → MIDI | Merge multi-track MIDI (instruments grouped by program/name) |
| **Predict Velocity** | MIDI + AUDIO → MIDI | Predict note velocities from the separated audio (stem_name dropdown) |
| **Refine Instrument** | MIDI + AUDIO → MIDI | Reassign instrument labels by timbre (mode: cluster/single) |
| **Beat Chord Key** | MIDI → MIDI | Predict beat/chord/key and write back; `fix_leading_tempo` on by default, corrects the inflated leading tempo caused by silence at the song start |
| **Save MIDI** | MIDI → file | Save to output/midi/ (SaveImage-style naming) |

## Installation

Requirements: a recent ComfyUI build (with built-in LoadAudio / AUDIO type).

```bash
# 1. Clone this repository inside custom_nodes (replace <your-comfyui-path> with your actual install location)
cd <your-comfyui-path>/custom_nodes
git clone https://github.com/DinbuSD/ComfyUI-instrument-agnostic-amt

# 2. Install dependencies (use ComfyUI's Python environment, run inside the node directory)
#    Portable edition:
ComfyUI_windows_portable\python_embeded\python.exe -m pip install -r requirements.txt

# 3. Restart ComfyUI
```

No **manual model download** is needed: models are pulled automatically from Hugging Face on the first node execution (official endpoint first, falls back to hf-mirror). All models total ~960MB and are cached in `ComfyUI/models/instrument_agnostic_amt/`.

If you are behind a restricted network, you can force a specific download endpoint:

```bash
set AMT_HF_ENDPOINT=https://hf-mirror.com   # Windows
export AMT_HF_ENDPOINT=https://hf-mirror.com # Linux/macOS
```

## Quick Start

Workflow templates ready to drag-and-drop into the ComfyUI canvas are provided in the `example_workflows/` directory:

- **simple-transcription.json**: simple single-track workflow (LoadAudio → transcribe → save)
- **stem-separation.json**: plain stem separation (LoadAudio → separate → save 6 stems as FLAC to output/stems/<stem>/)
- **stem-separated-transcription.json**: complete per-stem workflow (separate → transcribe 6 stems → per-stem velocity → refine → merge → beat/chord/key → save)

**Single-track transcription**:

```
LoadAudio(audio) → Instrument Agnostic Amt(model=default) → Save MIDI
```

**Full per-stem workflow** (the official Colab workflow — usually better than transcribing the full mix, especially for dense arrangements):

```
LoadAudio
  → Stem Separate
      ├─ vocals → Amt(vocal_harmony)─┐
      ├─ guitar → Amt(guitar_v1_5)  ─┤
      ├─ bass   → Amt(bass_v2)      ─┼→ Merge MIDI → Beat Chord Key → Save MIDI
      ├─ drums  → Amt(drums)        ─┤
      ├─ piano  → Amt(default)      ─┤
      └─ other  → Amt(other)        ─┘
```

Optional post-processing: `Predict Velocity` per track (requires that track's audio); the "other" track is recommended to go through `Refine Instrument` (the stem most prone to misclassification); `Merge MIDI` accepts up to 8 inputs.

## Model List (Auto-Downloaded)

| File | Purpose | Size |
|---|---|---|
| best_model.pth | General all-instrument | 53.5 MB |
| best_model_bass_v2.pth | Bass v2 | 54.5 MB |
| best_model_vocal.pth | Vocals | 53.5 MB |
| best_model_guitar_v1_5.pth | Guitar v1.5 | 54.5 MB |
| best_model_vocal_harmony.pth | Vocal harmony | 54.5 MB |
| best_model_drums.pth | Drums (experimental) | 54.5 MB |
| best_model_other.pth | Other instruments | 54.5 MB |
| best_velocity_model.pth | Velocity prediction | 55.0 MB |
| best_instrument_refinement.pth | Instrument refinement | 56.2 MB |
| best_beat_chord_key.pth | Beat/chord/key | 86.9 MB |
| stem_splitter.pt | Stem separation (BS-RoFormer) | 350 MB |

## Known Limitations

- **The drums model is Experimental**: note completeness is limited, and the official accuracy may change as the model evolves
- **Time offsets in per-stem transcription**: separately transcribing individual stems may introduce slight time offsets (a known upstream issue); align the merged result as a whole in your DAW
- **Vocal track timbre**: the vocal model outputs a `melody` role label, which maps to a melodic instrument (wind family) on MIDI playback — pitch/rhythm are correct; change the timbre to choir (GM 52-54) in your DAW if desired
- **Mono input**: transcription/separation require stereo input; mono audio is automatically duplicated to stereo

## License

The node code is MIT licensed. The vendored `instrument_agnostic_amt/` directory is from [anime-song/instrument-agnostic-amt](https://github.com/anime-song/instrument-agnostic-amt) (MIT License, Copyright (c) 2026 [anime-song](https://github.com/anime-song); see `instrument_agnostic_amt/VENDORED_README.txt` for the vendoring notes). Model weights and the stem splitter weights belong to their original author (anime-song).

## Development Note

This node pack was developed entirely with DeepSeek-V4-Flash (model) and ZCode (frontend).
