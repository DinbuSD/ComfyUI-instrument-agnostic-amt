"""ComfyUI nodes for instrument-agnostic-amt: audio -> MIDI transcription.

Design:
- Checkpoints are NOT auto-downloaded: place .pth files in
  ComfyUI/models/instrument_agnostic_amt/ and pick from the node dropdown;
- The transcription node outputs a MIDI data object; "Save MIDI" writes it to
  disk and optionally renders a preview audio (its AUDIO output is synthesized
  only when the output is connected downstream, e.g. to the official
  PreviewAudio node);
- AMT source code is referenced via sys.path (import only, never modified);
- All AMT imports are lazy to avoid breaking ComfyUI startup when the source
  or dependencies are not ready.
"""

import sys
from collections import OrderedDict
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace

NODE_DIR = Path(__file__).resolve().parent

# Checkpoint directory (dropdown source): ComfyUI/models/instrument_agnostic_amt/
try:
    import folder_paths  # noqa: F401  ComfyUI environment

    _CHECKPOINT_DIRS = [Path(folder_paths.models_dir) / "instrument_agnostic_amt"]
except Exception:
    # Fallback for standalone testing without ComfyUI
    _CHECKPOINT_DIRS = [NODE_DIR.parent.parent / "models" / "instrument_agnostic_amt"]

# 完整模型目录（下拉框显示全部；缺失时执行中自动下载）
# 转写模型（主节点）10 个：上游 2026-09 重构（tsumugi）后 drums/other/vocal_harmony
# 分轨走 *_v1_5 新模型（旧版保留，下拉仍可选）
_MODEL_CATALOG = [
    "best_model.pth",
    "best_model_bass_v2.pth",
    "best_model_vocal.pth",
    "best_model_guitar_v1_5.pth",
    "best_model_vocal_harmony.pth",
    "best_model_vocal_harmony_v1_5.pth",
    "best_model_drums.pth",
    "best_model_drums_v1_5.pth",
    "best_model_other.pth",
    "best_model_other_v1_5.pth",
    "best_velocity_model.pth",
    "best_instrument_refinement.pth",
    "best_beat_chord_key.pth",
]

_MISSING_HINT = "(No checkpoint found. Place .pth files in ComfyUI/models/instrument_agnostic_amt/)"

_AMT = {}          # lazy import cache

# 模型缓存：LRU 限制条目数，避免多模型工作流把显存占满
_MODEL_CACHE_MAX = 3
_MODEL_CACHE = OrderedDict()  # key: (checkpoint_str, device_str) -> (model, config, training_args)


def _cache_get(key):
    if key in _MODEL_CACHE:
        _MODEL_CACHE.move_to_end(key)
        return _MODEL_CACHE[key]
    return None


def _cache_put(key, value):
    _MODEL_CACHE[key] = value
    _MODEL_CACHE.move_to_end(key)
    while len(_MODEL_CACHE) > _MODEL_CACHE_MAX:
        _MODEL_CACHE.popitem(last=False)


def _clear_model_caches():
    """Release all cached models (CPU/GPU memory) on demand."""
    _MODEL_CACHE.clear()
    try:
        StemSeparate._MODEL_CACHE.clear()
    except NameError:  # 模块尚未定义 StemSeparate（加载早期）
        pass
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def _hook_model_unload():
    """让前端"卸载模型"操作同时释放本节点的模型缓存。

    ComfyUI 的卸载按钮调用 model_management.unload_all_models；
    包装它，在原逻辑之后清空我们的缓存并回收显存。
    """
    try:
        import comfy.model_management as mm

        orig = mm.unload_all_models
        if not getattr(orig, "_amt_hooked", False):
            def wrapped(*args, **kwargs):
                try:
                    orig(*args, **kwargs)
                finally:
                    _clear_model_caches()
                    print("[AMT] Cached models released (unload)")
            wrapped._amt_hooked = True
            mm.unload_all_models = wrapped
    except Exception:
        pass


_hook_model_unload()


def _import_amt() -> dict:
    """Lazily import AMT inference modules from the vendored package
    (this node's own instrument_agnostic_amt/ folder); returns {name: callable}."""
    if _AMT:
        return _AMT
    if str(NODE_DIR) not in sys.path:
        sys.path.insert(0, str(NODE_DIR))

    from instrument_agnostic_amt.cli.infer import (  # noqa: WPS433
        load_model,
        resolve_inference_settings,
    )
    from instrument_agnostic_amt.inference.midi import build_midi
    from instrument_agnostic_amt.inference.windowed import decode_notes

    _AMT.update(
        load_model=load_model,
        resolve_inference_settings=resolve_inference_settings,
        build_midi=build_midi,
        decode_notes=decode_notes,
    )
    return _AMT


def _list_checkpoints(pattern: str = "best_model*.pth") -> list[str]:
    """Return all catalog filenames matching the pattern (the dropdown shows
    every model; missing ones are downloaded on execution)."""
    import fnmatch

    names = [f for f in _MODEL_CATALOG if fnmatch.fnmatch(f, pattern)]
    return names or [_MISSING_HINT]


def _download_checkpoint(filename: str) -> Path:
    """Download a missing checkpoint into the models dir.

    Tries the official Hugging Face endpoint first, falls back to hf-mirror,
    then raises. Set AMT_HF_ENDPOINT to force a single endpoint.
    """
    import os

    import torch

    target_dir = _CHECKPOINT_DIRS[0]
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / filename

    env = os.environ.get("AMT_HF_ENDPOINT", "").strip().rstrip("/")
    endpoints = [env] if env else ["https://huggingface.co", "https://hf-mirror.com"]

    errors = []
    for ep in endpoints:
        url = f"{ep}/anime-song/instrument_agnostic_amt/resolve/main/{filename}?download=true"
        try:
            print(f"[AMT] Downloading {filename} from {ep} ...")
            torch.hub.download_url_to_file(url, str(target))
            print(f"[AMT] Downloaded {filename} -> {target}")
            return target
        except Exception as exc:
            errors.append(f"{ep}: {exc}")
            print(f"[AMT] Download failed from {ep}: {exc}")
            # 清理半截文件，避免下次误判为已存在
            target.unlink(missing_ok=True)

    raise RuntimeError(
        f"[AMT] Failed to download {filename} from all endpoints:\n"
        + "\n".join(errors)
    )


def _resolve_checkpoint(filename: str) -> Path:
    """Locate a checkpoint in the scanned dirs; auto-download when missing."""
    if filename == _MISSING_HINT:
        raise RuntimeError(
            f"[AMT] No checkpoint found. Place .pth files in:\n"
            f"  {_CHECKPOINT_DIRS[0]}"
        )
    for d in _CHECKPOINT_DIRS:
        p = d / filename
        if p.is_file():
            return p
    return _download_checkpoint(filename)


def _resolve_device(requested: str):
    """Resolve device option to a torch.device.

    Options: auto / cuda / cpu.
    """
    import torch

    req = str(requested)
    if req == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        print("[AMT] CUDA not detected, using CPU (inference will be much slower)")
        return torch.device("cpu")
    if req == "cuda" and not torch.cuda.is_available():
        print("[AMT] CUDA unavailable, falling back to CPU (inference will be much slower)")
        return torch.device("cpu")
    return torch.device(req)


def _patch_tqdm_progress(module, node_id):
    """Replace `module.tqdm` with a silent variant that forwards progress to
    ComfyUI's PROGRESS_BAR_HOOK (with node_id, the frontend renders the bar on
    the node). Returns a restore() callable.

    Caller must pass disable_tqdm=False to the wrapped function.
    """
    import tqdm.auto as _tqdm_auto

    orig = getattr(module, "tqdm", None)
    if orig is None:
        # 该模块推理路径不使用 tqdm（如 beat_chord），无法转发进度
        return lambda: None
    _node_id = str(node_id)

    def _send(value: int, total: int) -> None:
        try:
            from comfy import utils as _cu

            hook = _cu.PROGRESS_BAR_HOOK
            if hook is not None:
                hook(int(value), max(1, int(total)), None, node_id=_node_id)
        except Exception:
            pass

    class _SilentTqdm(_tqdm_auto.tqdm):
        def __init__(self, *args, **kwargs):
            kwargs["disable"] = False  # keep counting; output is silenced
            super().__init__(*args, **kwargs)

        # tqdm's __iter__ does NOT call update() for performance
        def __iter__(self):
            for obj in self.iterable:
                yield obj
                self.n += 1
                _send(self.n, self.total)

        def update(self, n=1):
            self.n += n
            _send(self.n, self.total)

        def refresh(self, *args, **kwargs):
            pass

        def display(self, *args, **kwargs):
            pass

        def close(self):
            if self.total:
                _send(self.total, self.total)

    module.tqdm = _SilentTqdm

    def restore():
        module.tqdm = orig

    return restore


class InstrumentAgnosticAmt:
    """Audio -> MIDI transcription. Connect LoadAudio's AUDIO output;
    the MIDI result is written by the Save MIDI node."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                # AUDIO-typed inputs render as a pure port on the left side
                "audio": ("AUDIO",),
                # Checkpoint dropdown; rescanned when the UI refreshes
                "model": (_list_checkpoints(),),
            },
            "optional": {
                # 静音检测：RMS 低于阈值视为无音频，跳过转写（省时间）
                "skip_silent": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "skip transcription when the track RMS is below the threshold"},
                ),
                "silence_rms_threshold": (
                    "FLOAT",
                    {"default": 0.001, "min": 0.0, "max": 1.0, "step": 0.0001,
                     "tooltip": "RMS below this = silent track (separated empty stems ~0.00003, real stems ~0.1)"},
                ),
                "use_amp": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "mixed precision (CUDA only)"},
                ),
                # auto: use CUDA when available, otherwise CPU; or force one
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto",
                     "tooltip": "auto: pick automatically, fall back to CPU if CUDA unavailable"},
                ),
                # --- 高级参数 ---
                "window_ms": (
                    "INT",
                    {"default": -1, "min": -1, "max": 60000, "step": 500,
                     "tooltip": "-1 = use model default window (ms)"},
                ),
                "stride_ms": (
                    "INT",
                    {"default": -1, "min": -1, "max": 60000, "step": 500,
                     "tooltip": "-1 = half of window (ms)"},
                ),
                "window_batch_size": (
                    "INT",
                    {"default": 1, "min": 1, "max": 16,
                     "tooltip": "windows processed at once (higher = faster but more VRAM)"},
                ),
                "merge_gap_ms": (
                    "INT",
                    {"default": -1, "min": -1, "max": 500,
                     "tooltip": "-1 = default; merge threshold for small note gaps"},
                ),
                "merge_onset_ms": (
                    "INT",
                    {"default": 20, "min": 0, "max": 500,
                     "tooltip": "merge threshold for near-simultaneous onsets"},
                ),
            },
            "hidden": {
                # 节点 id：用于在节点上显示解码进度条
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AMT_MIDI", "INT", "FLOAT")
    RETURN_NAMES = ("midi", "note_count", "audio_duration_s")
    FUNCTION = "transcribe"
    CATEGORY = "Audio/AMT"

    def transcribe(
        self,
        audio,
        model,
        window_ms=-1,
        stride_ms=-1,
        window_batch_size=1,
        merge_gap_ms=-1,
        merge_onset_ms=20,
        use_amp=True,
        device="auto",
        skip_silent=True,
        silence_rms_threshold=0.001,
        prompt=None,
        unique_id=None,
    ):
        import torch
        import torchaudio.functional as TAF

        amt = _import_amt()
        checkpoint_path = _resolve_checkpoint(model)

        # audio must be LoadAudio's AUDIO output
        # ({"waveform": [1, C, T], "sample_rate": int})
        if not (isinstance(audio, dict) and "waveform" in audio):
            raise RuntimeError(
                f"[AMT] Invalid audio input: {type(audio).__name__}. "
                "Connect LoadAudio's AUDIO output."
            )

        dev = _resolve_device(device)
        amp_enabled = bool(use_amp and dev.type == "cuda")
        amp_dtype = torch.bfloat16 if dev.type == "cuda" else torch.float32

        # Model cache: LRU, at most _MODEL_CACHE_MAX models resident
        cache_key = (str(checkpoint_path), dev.type)
        cached = _cache_get(cache_key)
        if cached is None:
            print(f"[AMT] Loading model {checkpoint_path.name} ...")
            model_, config, training_args = amt["load_model"](checkpoint_path, device=dev)
            _cache_put(cache_key, (model_, config, training_args))
            cached = _cache_get(cache_key)
        model_, config, training_args = cached

        # Inference settings (aligned with the official infer.py defaults)
        args = SimpleNamespace(
            window_ms=window_ms if window_ms > 0 else None,
            stride_ms=stride_ms if stride_ms > 0 else None,
            semi_crf_track_batch_size=None,
            window_batch_size=window_batch_size,
            merge_gap_ms=None,
            merge_onset_ms=50.0,
            silence_gate_rms_dbfs=-72.0,
            note_bias=0.0,
            disable_tqdm=True,
            no_boundary_head=False,
            semi_crf_sparse_decode=False,
            semi_crf_sparse_topk_per_start=16,
            semi_crf_sparse_score_threshold=None,
            semi_crf_sparse_max_span_ms=None,
            instrument_pair_infer_topk=256,
            instrument_pair_gate_threshold=-3.0,
            instrument_pair_max_pairs=512,
            # 上游 2026-09 重构（tsumugi）新增：dense Semi-CRF 解码后端
            # （torch 即可；triton 需 CUDA JIT 编译，本节点不启用）
            semi_crf_backend="torch",
        )
        settings = amt["resolve_inference_settings"](config, training_args, args)
        settings = replace(
            settings,
            merge_gap_ms=None if merge_gap_ms < 0 else float(merge_gap_ms),
            merge_onset_ms=float(merge_onset_ms),
            disable_tqdm=True,
        )

        sample_rate = int(config.sample_rate)
        print(
            f"[AMT] Transcribing LoadAudio waveform input "
            f"(checkpoint={checkpoint_path.name}, device={dev.type}, amp={amp_enabled})"
        )
        waveform = audio["waveform"].squeeze(0)  # [1, C, T] -> [C, T]
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] == 1:
            waveform = waveform.repeat(2, 1)
        elif waveform.shape[0] > 2:
            waveform = waveform[:2]
        src_sr = int(audio["sample_rate"])
        if src_sr != sample_rate:
            waveform = TAF.resample(waveform, src_sr, sample_rate)
        waveform = waveform.float()

        # 静音检测：RMS 低于阈值视为无音频，跳过转写（省时间）
        if skip_silent:
            rms = float(waveform.pow(2).mean().sqrt())
            if rms < float(silence_rms_threshold):
                import pretty_midi

                duration = float(waveform.shape[-1]) / float(sample_rate)
                print(
                    f"[AMT] Skipped: audio too quiet (RMS={rms:.5f} < "
                    f"{silence_rms_threshold}), returning empty MIDI"
                )
                return (pretty_midi.PrettyMIDI(), 0, duration)

        # --- Node progress bar: forward the per-window tqdm progress inside
        # decode_notes to ComfyUI's progress hook (with node_id, the frontend
        # renders the bar on this node). Sends directly via PROGRESS_BAR_HOOK
        # to bypass ProgressBar's time throttling (window batches complete
        # faster than its min interval). ---
        import tqdm.auto as _tqdm_auto
        import instrument_agnostic_amt.inference.windowed as _windowed_mod

        # v1 模型走 v1_windowed 模块（v1/v2 各自模块级绑定 tqdm，需一并 patch）
        _patched_mods = [_windowed_mod]
        try:
            import instrument_agnostic_amt.inference.v1_windowed as _v1_mod

            _patched_mods.append(_v1_mod)
        except Exception:
            pass
        _orig_tqdms = {m: m.tqdm for m in _patched_mods}

        bar_total = None
        if unique_id:
            try:
                from comfy import utils as _comfy_utils

                _node_id = str(unique_id)
                n_windows = len(
                    _windowed_mod._build_window_starts(
                        total_audio_frames=int(waveform.shape[-1]),
                        window_audio_frames=int(
                            round(float(settings.window_ms) * sample_rate / 1000.0)
                        ),
                        stride_audio_frames=int(
                            round(float(settings.stride_ms) * sample_rate / 1000.0)
                        ),
                    )
                )
                bar_total = n_windows

                def _send_progress(value: int, total: int) -> None:
                    hook = _comfy_utils.PROGRESS_BAR_HOOK
                    if hook is not None:
                        hook(int(value), int(total), None, node_id=_node_id)

                class _SilentTqdm(_tqdm_auto.tqdm):
                    def __init__(self, *args, **kwargs):
                        kwargs["disable"] = False  # keep counting; output is silenced
                        super().__init__(*args, **kwargs)

                    # tqdm's __iter__ does NOT call update() for performance;
                    # forward progress here (one update per window batch)
                    def __iter__(self):
                        for obj in self.iterable:
                            yield obj
                            self.n += 1
                            _send_progress(self.n, bar_total)

                    def update(self, n=1):
                        self.n += n
                        _send_progress(self.n, bar_total)

                    def refresh(self, *args, **kwargs):
                        pass

                    def display(self, *args, **kwargs):
                        pass

                    def close(self):
                        _send_progress(bar_total, bar_total)

                for _mod in _patched_mods:
                    _mod.tqdm = _SilentTqdm
            except Exception as exc:
                print(f"[AMT] Progress bar disabled: {exc}")

        try:
            with torch.no_grad():
                notes, stats = amt["decode_notes"](
                    model_,
                    config,
                    waveform,
                    instrument_filter_id=None,
                    device=dev,
                    amp_enabled=amp_enabled,
                    amp_dtype=amp_dtype,
                    settings=settings,
                    velocity=100,
                )
        finally:
            for _mod in _patched_mods:
                _mod.tqdm = _orig_tqdms[_mod]
            if bar_total is not None:
                _send_progress(bar_total, bar_total)

        midi, _ = amt["build_midi"](
            notes,
            sample_rate=sample_rate,
            instrument_id=None,
            min_midi_note_ms=5.0,
            max_midi_melodic_instruments=15,
            instrument_volumes=None,
            return_stats=True,
        )

        duration = float(waveform.shape[-1]) / float(sample_rate)
        print(
            f"[AMT] Done: {len(notes)} notes "
            f"(windows={stats.get('window_count')}, "
            f"skipped_silent={stats.get('skipped_silent_window_count')})"
        )
        return (midi, len(notes), float(duration))


def _merge_midi_objects(midi_objects: list, max_melodic_instruments: int = 15):
    """把多个 pretty_midi 对象合并为一个（对齐官方 infer_stem.merge_midis_logic）。

    按 (program, is_drum, name) 分组乐器；超过 15 个旋律乐器时
    多余的归并到 "Other / Merged" 轨道；超长音符（>15s）被过滤。
    """
    from collections import defaultdict

    import pretty_midi

    all_notes: dict = defaultdict(list)
    all_cc: dict = defaultdict(list)
    all_pb: dict = defaultdict(list)
    names: dict = {}

    for obj in midi_objects:
        for inst in obj.instruments:
            key = (inst.program, inst.is_drum, inst.name)
            all_notes[key].extend(n for n in inst.notes if (n.end - n.start) < 15.0)
            all_cc[key].extend(inst.control_changes)
            all_pb[key].extend(inst.pitch_bends)
            names.setdefault(key, inst.name)

    melodic_keys = [k for k in all_notes if not k[1]]
    drum_keys = [k for k in all_notes if k[1]]
    melodic_keys.sort(key=lambda k: len(all_notes[k]), reverse=True)

    final_instruments: list = []
    if len(melodic_keys) > max_melodic_instruments:
        kept = melodic_keys[: max_melodic_instruments - 1]
        overflow = melodic_keys[max_melodic_instruments - 1:]
        for key in kept:
            inst = pretty_midi.Instrument(program=key[0], is_drum=key[1], name=names[key])
            inst.notes = all_notes[key]
            inst.control_changes = all_cc[key]
            inst.pitch_bends = all_pb[key]
            final_instruments.append(inst)
        base = overflow[0]
        ovf = pretty_midi.Instrument(program=base[0], is_drum=base[1], name="Other / Merged")
        for key in overflow:
            ovf.notes.extend(all_notes[key])
            ovf.control_changes.extend(all_cc[key])
            ovf.pitch_bends.extend(all_pb[key])
        final_instruments.append(ovf)
    else:
        for key in melodic_keys:
            inst = pretty_midi.Instrument(program=key[0], is_drum=key[1], name=names[key])
            inst.notes = all_notes[key]
            inst.control_changes = all_cc[key]
            inst.pitch_bends = all_pb[key]
            final_instruments.append(inst)
    for key in drum_keys:
        inst = pretty_midi.Instrument(program=key[0], is_drum=True, name=names[key])
        inst.notes = all_notes[key]
        inst.control_changes = all_cc[key]
        inst.pitch_bends = all_pb[key]
        final_instruments.append(inst)

    master = pretty_midi.PrettyMIDI()
    if midi_objects:
        first = midi_objects[0]
        master.tempo = list(first.tempo) if hasattr(first, "tempo") else [120.0]
        master.time_signature_changes = list(first.time_signature_changes)
    master.instruments = final_instruments
    return master


class MergeMidi:
    """把多个 AMT MIDI 合并成一个（多模型分轨转录后合并成单个文件）。"""

    MAX_INPUTS = 8

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi_1": ("AMT_MIDI",),
            },
            "optional": {
                f"midi_{i}": ("AMT_MIDI",) for i in range(2, cls.MAX_INPUTS + 1)
            },
        }

    RETURN_TYPES = ("AMT_MIDI",)
    RETURN_NAMES = ("midi",)
    FUNCTION = "merge"
    CATEGORY = "Audio/AMT"

    def merge(self, **kwargs):
        objs = [v for v in kwargs.values() if v is not None]
        if not objs:
            raise RuntimeError("[AMT] Merge MIDI: no inputs connected")
        merged = _merge_midi_objects(objs)
        n_notes = sum(len(i.notes) for i in merged.instruments)
        print(
            f"[AMT] Merged {len(objs)} MIDIs into {len(merged.instruments)} tracks "
            f"({n_notes} notes)"
        )
        return (merged,)


class VelocityPredict:
    """Predict per-note velocity from the separated stem audio.

    Inputs: the AMT MIDI of one stem plus that stem's audio.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": ("AMT_MIDI",),
                "audio": ("AUDIO",),
                "stem_name": (
                    ["vocals", "guitar", "bass", "drums", "piano", "other"],
                    {"default": "other",
                     "tooltip": "which separated stem this audio/MIDI belongs to"},
                ),
            },
            "optional": {
                "checkpoint": (_list_checkpoints("best_velocity_model.pth"),),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto",
                     "tooltip": "auto: pick automatically, fall back to CPU if CUDA unavailable"},
                ),
            },
            "hidden": {
                # 节点 id：用于在节点上显示进度条
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AMT_MIDI",)
    RETURN_NAMES = ("midi",)
    FUNCTION = "predict"
    CATEGORY = "Audio/AMT"

    def predict(self, midi, audio, stem_name="other", checkpoint="", device="auto",
                prompt=None, unique_id=None):
        import soundfile as sf

        amt = _import_amt()
        import instrument_agnostic_amt.velocity.cli.infer_velocity as _vcli

        if not (isinstance(audio, dict) and "waveform" in audio):
            raise RuntimeError(
                f"[AMT] Invalid audio input: {type(audio).__name__}. "
                "Connect the separated stem audio."
            )
        if checkpoint in (None, ""):
            checkpoint = _list_checkpoints("best_velocity_model.pth")[0]
        checkpoint_path = _resolve_checkpoint(checkpoint)

        dev = _resolve_device(device)

        # 空 MIDI 直接透传（官方对 0 音符的 MIDI 不写输出文件）
        if sum(len(i.notes) for i in midi.instruments) == 0:
            print("[AMT] Velocity skipped: input MIDI has no notes")
            return (midi,)

        # Temporary files in ComfyUI/temp
        try:
            import folder_paths

            temp_dir = Path(folder_paths.get_temp_directory())
        except Exception:
            temp_dir = NODE_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        midi_path = temp_dir / "velocity_in.mid"
        wav_path = temp_dir / "velocity_stem.wav"
        out_path = temp_dir / "velocity_out.mid"

        try:
            midi.write(str(midi_path))
            waveform = audio["waveform"].squeeze(0).numpy()  # [C, T]
            sf.write(str(wav_path), waveform.T, int(audio["sample_rate"]))

            print(f"[AMT] Predicting velocity (stem={stem_name}, device={dev.type})")
            restore = None
            if unique_id:
                restore = _patch_tqdm_progress(_vcli, unique_id)
            try:
                # 显式配对 stem→MIDI/音频，绕开官方按轨道名猜测的入口
                # （vocal 模型输出的轨道名 "melody" 不在 STEM_NAMES 里，
                # 走 predict_velocity_for_midi 会落回 "other" 导致音频失配、预测塌缩）
                _vcli.predict_velocity_for_stem_midis(
                    stem_midis={stem_name: midi_path},
                    stem_audios={stem_name: wav_path},
                    output_midi_path=out_path,
                    template_midi_path=midi_path,
                    checkpoint_path=checkpoint_path,
                    device=dev,
                    disable_tqdm=False,
                )
            finally:
                if restore is not None:
                    restore()

            import pretty_midi

            # 兜底：官方未产出文件时（如预测异常）透传原 MIDI
            if not out_path.exists():
                print("[AMT] Velocity produced no output; returning input MIDI unchanged")
                return (midi,)
            result = pretty_midi.PrettyMIDI(str(out_path))
            n = sum(len(i.notes) for i in result.instruments)
            print(f"[AMT] Velocity done: {n} notes")
            return (result,)
        finally:
            for p in (midi_path, wav_path, out_path):
                p.unlink(missing_ok=True)


class RefineInstrument:
    """Re-assign instrument classes of AMT notes using the stem audio."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": ("AMT_MIDI",),
                "audio": ("AUDIO",),
                "stem_name": (
                    ["vocals", "guitar", "bass", "drums", "piano", "other"],
                    {"default": "other",
                     "tooltip": "which separated stem this audio/MIDI belongs to"},
                ),
            },
            "optional": {
                "checkpoint": (_list_checkpoints("best_instrument_refinement.pth"),),
                "mode": (
                    ["cluster", "single"],
                    {"default": "cluster",
                     "tooltip": "cluster: group notes by timbre; single: one instrument per stem"},
                ),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto",
                     "tooltip": "auto: pick automatically, fall back to CPU if CUDA unavailable"},
                ),
            },
            "hidden": {
                # 节点 id：用于在节点上显示进度条
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AMT_MIDI",)
    RETURN_NAMES = ("midi",)
    FUNCTION = "refine"
    CATEGORY = "Audio/AMT"

    def refine(self, midi, audio, stem_name="other", checkpoint="", mode="cluster",
               device="auto", prompt=None, unique_id=None):
        import soundfile as sf

        _import_amt()
        from instrument_agnostic_amt.instrument_refinement.inference.refine import (
            refine_midi_instruments,
        )

        if not (isinstance(audio, dict) and "waveform" in audio):
            raise RuntimeError(
                f"[AMT] Invalid audio input: {type(audio).__name__}. "
                "Connect the separated stem audio."
            )
        if checkpoint in (None, ""):
            checkpoint = _list_checkpoints("best_instrument_refinement.pth")[0]
        checkpoint_path = _resolve_checkpoint(checkpoint)

        dev = _resolve_device(device)

        # 空 MIDI 直接透传（无音符可精修）
        if sum(len(i.notes) for i in midi.instruments) == 0:
            print("[AMT] Refine skipped: input MIDI has no notes")
            return (midi,)

        try:
            import folder_paths

            temp_dir = Path(folder_paths.get_temp_directory())
        except Exception:
            temp_dir = NODE_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        midi_path = temp_dir / "refine_in.mid"
        wav_path = temp_dir / "refine_stem.wav"
        out_path = temp_dir / "refine_out.mid"

        try:
            midi.write(str(midi_path))
            waveform = audio["waveform"].squeeze(0).numpy()  # [C, T]
            sf.write(str(wav_path), waveform.T, int(audio["sample_rate"]))

            print(f"[AMT] Refining instruments (stem={stem_name}, mode={mode})")
            restore = None
            if unique_id:
                import instrument_agnostic_amt.instrument_refinement.inference.refine as _rmod

                restore = _patch_tqdm_progress(_rmod, unique_id)
            try:
                refine_midi_instruments(
                    wav_path,
                    midi_path,
                    checkpoint_path=checkpoint_path,
                    output_midi_path=out_path,
                    stem_name=stem_name,
                    device=dev,
                    mode=mode,
                    disable_tqdm=False,
                )
            finally:
                if restore is not None:
                    restore()

            import pretty_midi

            # 兜底：官方未产出文件时（如预测异常）透传原 MIDI
            if not out_path.exists():
                print("[AMT] Refine produced no output; returning input MIDI unchanged")
                return (midi,)
            result = pretty_midi.PrettyMIDI(str(out_path))
            print(f"[AMT] Refinement done: {len(result.instruments)} tracks")
            return (result,)
        finally:
            for p in (midi_path, wav_path, out_path):
                p.unlink(missing_ok=True)


class BeatChordKey:
    """Predict beat / chord / key for a MIDI and write the beat-mapped MIDI."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": ("AMT_MIDI",),
            },
            "optional": {
                "checkpoint": (_list_checkpoints("best_beat_chord_key.pth"),),
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto",
                     "tooltip": "auto: pick automatically, fall back to CPU if CUDA unavailable"},
                ),
                "fix_leading_tempo": (
                    "BOOLEAN",
                    {"default": True,
                     "tooltip": "fix the inflated leading tempo the official model writes when the song starts with silence/pickup"},
                ),
            },
            "hidden": {
                # 节点 id：用于在节点上显示进度条
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AMT_MIDI",)
    RETURN_NAMES = ("midi",)
    FUNCTION = "infer"
    CATEGORY = "Audio/AMT"

    def infer(self, midi, checkpoint="", device="auto", fix_leading_tempo=True,
              prompt=None, unique_id=None):
        _import_amt()
        from instrument_agnostic_amt.beat_chord.cli.infer import (
            predict_beat_chord_for_midi,
        )

        if checkpoint in (None, ""):
            checkpoint = _list_checkpoints("best_beat_chord_key.pth")[0]
        checkpoint_path = _resolve_checkpoint(checkpoint)
        dev = _resolve_device(device)

        # 空 MIDI 直接透传（无音符可预测节拍/和弦）
        if sum(len(i.notes) for i in midi.instruments) == 0:
            print("[AMT] Beat/chord/key skipped: input MIDI has no notes")
            return (midi,)

        try:
            import folder_paths

            temp_dir = Path(folder_paths.get_temp_directory())
        except Exception:
            temp_dir = NODE_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        midi_path = temp_dir / "beat_chord_in.mid"
        out_path = temp_dir / "beat_chord_out.mid"

        try:
            midi.write(str(midi_path))
            # 官方 _load_midi_events 按文件路径 lru_cache：temp 文件名固定，
            # 同进程连着跑多首时第二次会命中第一首的缓存（BPM/调性错乱）。
            import instrument_agnostic_amt.beat_chord.midi_roll as _mr

            _mr._load_midi_events.cache_clear()
            print(f"[AMT] Predicting beat/chord/key (device={dev.type})")
            predict_beat_chord_for_midi(
                midi_path,
                out_path,
                checkpoint_path=checkpoint_path,
                device=dev,
                disable_tqdm=False,
            )

            import pretty_midi

            # 兜底：官方未产出文件时（如预测异常）透传原 MIDI
            if not out_path.exists():
                print("[AMT] Beat/chord/key produced no output; returning input MIDI unchanged")
                return (midi,)
            result = pretty_midi.PrettyMIDI(str(out_path))
            # 官方对开头空白段外推虚高的初始 tempo（t=0 处可能是正常值的数倍）。
            # 补丁：首 tempo 超过第二个 1.5 倍时，用第二个替换（默认开，可关）。
            # pretty_midi 无 tempo setter，直接改内部 _tick_scales（tempo = 60/(scale*resolution)）
            if fix_leading_tempo:
                _times, _bpms = result.get_tempo_changes()
                if len(_bpms) >= 2 and _bpms[0] > _bpms[1] * 1.5:
                    _tick, _ = result._tick_scales[0]
                    result._tick_scales[0] = (
                        _tick,
                        60.0 / (_bpms[1] * result.resolution),
                    )
                    print(f"[AMT] Leading tempo fixed: {_bpms[0]:.1f} -> {_bpms[1]:.1f} BPM")
            n_tracks = len(result.instruments)
            n_notes = sum(len(i.notes) for i in result.instruments)
            print(f"[AMT] Beat/chord/key done: {n_tracks} tracks, {n_notes} notes")
            return (result,)
        finally:
            for p in (midi_path, out_path):
                p.unlink(missing_ok=True)


def _separate_with_progress(
    input_wav_path: Path,
    output_directory: Path,
    config,
    model,
    device,
    send_progress,
):
    """stem-splitter 分离循环（带进度回调）。

    与官方 _separate_one_file 同逻辑（overlap-add），每处理一个 chunk 调用
    send_progress(done, total)。窗口张量强制 float32（float16 的 hann 窗会 NaN）。
    """
    import math

    import numpy as np
    import soundfile as sf
    import torch
    import torchaudio.functional as AF

    waveform, sr = sf.read(str(input_wav_path), dtype="float32", always_2d=True)
    waveform = torch.from_numpy(waveform.T.copy())  # [C, T]
    if sr != int(config.target_sample_rate):
        waveform = AF.resample(waveform, orig_freq=sr, new_freq=config.target_sample_rate)
    audio = waveform.numpy().astype(np.float32, copy=False)

    channels, total_length = audio.shape
    chunk_size = int(config.chunk_size)
    hop_size = int(config.hop_size) if config.hop_size else chunk_size // 2
    if total_length <= chunk_size:
        padded_length = chunk_size
    else:
        steps = math.ceil((total_length - chunk_size) / hop_size)
        padded_length = steps * hop_size + chunk_size
    if padded_length > total_length:
        audio = np.pad(audio, ((0, 0), (0, padded_length - total_length)), mode="constant")

    use_amp = bool(config.enable_autocast and config.use_half_precision and device.type == "cuda")
    tensor_dtype = torch.float32  # window 必须 float32（float16 hann 窗 NaN）
    base_window = torch.hann_window(chunk_size, periodic=False, dtype=tensor_dtype, device=device)
    base_window_np = base_window.to(torch.float32).cpu().numpy()

    num_stems = len(config.stem_names)
    accum = np.zeros((num_stems, channels, padded_length), dtype=np.float32)
    weight_sum = np.zeros(padded_length, dtype=np.float64)

    starts = list(range(0, padded_length - chunk_size + 1, hop_size))
    total_chunks = len(starts)

    with torch.inference_mode():
        for i in range(0, total_chunks, int(config.batch_size)):
            batch_starts = starts[i : i + int(config.batch_size)]
            input_chunks = []
            for start in batch_starts:
                chunk_np = audio[:, start : start + chunk_size]
                if chunk_np.shape[1] < chunk_size:
                    chunk_np = np.pad(
                        chunk_np, ((0, 0), (0, chunk_size - chunk_np.shape[1])), mode="constant"
                    )
                input_chunks.append(chunk_np)
            input_batch = torch.from_numpy(np.stack(input_chunks)).to(device=device, dtype=tensor_dtype)

            with torch.amp.autocast(
                device_type=device.type, enabled=use_amp, dtype=torch.float16
            ):
                output_batch = model(input_batch)

            for b_idx, start in enumerate(batch_starts):
                output_chunk = output_batch[b_idx : b_idx + 1]
                _, _, _, t_out = output_chunk.shape
                if t_out == chunk_size:
                    window_vec = base_window
                    window_np = base_window_np
                else:
                    window_vec = torch.hann_window(
                        t_out, periodic=False, dtype=tensor_dtype, device=device
                    )
                    window_np = window_vec.to(torch.float32).cpu().numpy()
                windowed = (output_chunk * window_vec.view(1, 1, 1, -1)).squeeze(0)
                out_np = windowed.to(torch.float32).cpu().numpy()
                accum[:, :, start : start + t_out] += out_np
                weight_sum[start : start + t_out] += window_np

            send_progress(min(i + len(batch_starts), total_chunks), total_chunks)

    eps = 1e-8
    weight_sum = np.maximum(weight_sum, eps)
    accum /= weight_sum[None, None, :]

    base_name = Path(input_wav_path).stem
    out_root = Path(output_directory) / base_name
    out_root.mkdir(parents=True, exist_ok=True)
    saved: dict = {}
    for stem_index, stem_name in enumerate(config.stem_names):
        out_path = out_root / f"{base_name}_{stem_name}.flac"
        sf.write(str(out_path), accum[stem_index, :, :total_length].T, int(config.target_sample_rate))
        saved[stem_name] = out_path
    return saved


class StemSeparate:
    """Separate audio into stems with the official stem-splitter (BS-RoFormer).

    Outputs 6 AUDIO stems: vocals / bass / drums / other / guitar / piano.
    The model weight (stem_splitter.pt) is expected at
    ComfyUI/models/instrument_agnostic_amt/ — never auto-downloaded.
    """

    _MODEL_CACHE = {}  # key: device.type -> model

    STEM_ORDER = ("vocals", "guitar", "bass", "drums", "piano", "other")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio": ("AUDIO",),
            },
            "optional": {
                "device": (
                    ["auto", "cuda", "cpu"],
                    {"default": "auto",
                     "tooltip": "auto: pick automatically, fall back to CPU if CUDA unavailable"},
                ),
                "use_half": (
                    "BOOLEAN",
                    {"default": True, "tooltip": "half precision (CUDA only)"},
                ),
                # 临时开关：打开时把 6 轨分离 flac 保存到 output/stems/<音频名>_<时间戳>/
                "save_outputs": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "temporary: save stem flacs to output/stems/"},
                ),
            },
            "hidden": {
                # 节点 id：用于在节点上显示分离进度条
                "prompt": "PROMPT",
                "unique_id": "UNIQUE_ID",
            },
        }

    RETURN_TYPES = ("AUDIO",) * 6
    RETURN_NAMES = tuple(STEM_ORDER)
    FUNCTION = "separate"
    CATEGORY = "Audio/AMT"

    def separate(self, audio, device="auto", use_half=True, save_outputs=False,
                 prompt=None, unique_id=None):
        import soundfile as sf
        import torch

        if not (isinstance(audio, dict) and "waveform" in audio):
            raise RuntimeError(
                f"[AMT] Invalid audio input: {type(audio).__name__}. "
                "Connect LoadAudio's AUDIO output."
            )

        from stem_splitter.inference import SeparationConfig, load_mss_model

        dev = _resolve_device(device)
        # Window tensors MUST stay float32: torch.hann_window with a long
        # window (588800 samples) produces NaN in float16 on CUDA.
        dtype = torch.float32

        # Weight cache dir: ComfyUI/models/instrument_agnostic_amt/
        # (weight pre-downloaded there; never auto-download)
        try:
            import folder_paths

            cache_dir = Path(folder_paths.models_dir) / "instrument_agnostic_amt"
        except Exception:
            cache_dir = NODE_DIR.parent.parent / "models" / "instrument_agnostic_amt"

        # use_half_precision + enable_autocast (default True) => model runs in
        # fp16 via autocast while window tensors stay float32
        config = SeparationConfig(
            cache_dir=cache_dir,
            device_preference=dev.type,
            use_half_precision=bool(use_half and dev.type == "cuda"),
        )

        if dev.type not in StemSeparate._MODEL_CACHE:
            print(f"[AMT] Loading stem separator (bs_roformer) ...")
            StemSeparate._MODEL_CACHE[dev.type] = load_mss_model(config, dev)
        model = StemSeparate._MODEL_CACHE[dev.type]

        # Temporary files in ComfyUI/temp
        try:
            import folder_paths as _fp

            temp_dir = Path(_fp.get_temp_directory())
        except Exception:
            temp_dir = NODE_DIR / "temp"
        temp_dir.mkdir(parents=True, exist_ok=True)
        in_path = temp_dir / "stem_separate_in.wav"
        out_dir = temp_dir / "stem_separate_out"

        try:
            import numpy as np

            waveform = audio["waveform"].squeeze(0).numpy()  # [C, T]
            if waveform.ndim == 1:
                waveform = waveform[None]
            if waveform.shape[0] == 1:
                waveform = np.repeat(waveform, 2, axis=0)  # 单声道 → 双声道
            sf.write(str(in_path), waveform.T, int(audio["sample_rate"]))

            # 进度发送：直连 PROGRESS_BAR_HOOK（带 node_id，前端在节点上显示进度条）
            def _send_progress(value: int, total: int) -> None:
                if unique_id:
                    try:
                        from comfy import utils as _cu

                        hook = _cu.PROGRESS_BAR_HOOK
                        if hook is not None:
                            hook(int(value), int(total), None, node_id=str(unique_id))
                    except Exception:
                        pass

            print(f"[AMT] Separating stems (device={dev.type}, half={config.use_half_precision})")
            paths = _separate_with_progress(
                in_path, out_dir, config, model, dev, _send_progress
            )

            results = []
            for stem in self.STEM_ORDER:
                p = paths.get(stem) if paths else None
                if p is not None and Path(p).exists():
                    w, sr = sf.read(str(p), dtype="float32", always_2d=True)
                    results.append(
                        {
                            "waveform": torch.from_numpy(w.T.copy()).unsqueeze(0),
                            "sample_rate": sr,
                        }
                    )
                else:
                    results.append(None)
            done = [s for s, r in zip(self.STEM_ORDER, results) if r is not None]
            print(f"[AMT] Separation done: {done} tracks")

            # 临时开关：保存分离 flac 到 output/stems/<音频名>_<时间戳>/
            if save_outputs:
                import re
                import shutil
                import time as _time

                # 溯源上游 LoadAudio 的音频文件名（AUDIO 数据本身不带文件名）
                audio_name = None
                if prompt and unique_id:
                    for nid, node in prompt.items():
                        if str(nid) == str(unique_id):
                            src = node.get("inputs", {}).get("audio")
                            if isinstance(src, list) and len(src) >= 2:
                                src_node = prompt.get(str(src[0]), {})
                                if src_node.get("class_type") == "LoadAudio":
                                    audio_name = src_node.get("inputs", {}).get("audio")
                            break
                if audio_name:
                    audio_name = re.sub(r'[<>:"/\\|?*]', "_", Path(audio_name).stem)

                try:
                    import folder_paths

                    out_stems = Path(folder_paths.get_output_directory()) / "stems"
                except Exception:
                    out_stems = Path.cwd() / "stems"
                stamp = _time.strftime("%Y%m%d_%H%M%S")
                stem_dir = out_stems / (f"{audio_name}_{stamp}" if audio_name else stamp)
                stem_dir.mkdir(parents=True, exist_ok=True)
                for stem in done:
                    p = paths.get(stem)
                    if p is not None and Path(p).exists():
                        shutil.copy2(p, stem_dir / f"{stem}.flac")
                print(f"[AMT] Stems saved to {stem_dir}")

            return tuple(results)
        finally:
            in_path.unlink(missing_ok=True)


class SaveAmtMidi:
    """Save the AMT MIDI output to disk (terminal output node)."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "midi": ("AMT_MIDI",),
            },
            "optional": {
                # 默认 output/midi/ 文件夹、ComfyUI_ 前缀（SaveImage 风格）；
                # 前缀支持子文件夹，如 "midi/song1" -> output/midi/song1/
                "filename_prefix": (
                    "STRING",
                    {
                        "default": "midi/ComfyUI",
                        "tooltip": "subfolder/filename prefix (default midi/ComfyUI)",
                    },
                ),
            },
        }

    RETURN_TYPES = ()
    RETURN_NAMES = ()
    FUNCTION = "save"
    CATEGORY = "Audio/AMT"
    OUTPUT_NODE = True  # save nodes must be marked as output nodes

    def save(self, midi, filename_prefix="midi/ComfyUI"):
        try:
            import folder_paths

            out_dir = Path(folder_paths.get_output_directory())
        except Exception:
            out_dir = Path.cwd()

        # SaveImage-style: last prefix segment becomes the filename start,
        # remaining segments become subfolders. ".." is stripped.
        parts = [
            p
            for p in filename_prefix.strip().replace("\\", "/").split("/")
            if p not in ("", ".", "..")
        ]
        name = parts.pop() if parts else "midi"
        sub = Path(*parts) if parts else Path()

        save_dir = out_dir / sub
        save_dir.mkdir(parents=True, exist_ok=True)
        # Incrementing counter naming (SaveImage style), never overwrites
        counter = 1
        while (save_dir / f"{name}_{counter:05d}_.mid").exists():
            counter += 1
        midi_path = save_dir / f"{name}_{counter:05d}_.mid"
        midi.write(str(midi_path))

        print(f"[AMT] MIDI saved: {midi_path}")
        return {}


NODE_CLASS_MAPPINGS = {
    "InstrumentAgnosticAmt": InstrumentAgnosticAmt,
    "StemSeparate": StemSeparate,
    "MergeMidi": MergeMidi,
    "VelocityPredict": VelocityPredict,
    "RefineInstrument": RefineInstrument,
    "BeatChordKey": BeatChordKey,
    "SaveMidi": SaveAmtMidi,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "InstrumentAgnosticAmt": "Instrument Agnostic Amt",
    "StemSeparate": "Stem Separate",
    "MergeMidi": "Merge MIDI",
    "VelocityPredict": "Predict Velocity",
    "RefineInstrument": "Refine Instrument",
    "BeatChordKey": "Beat Chord Key",
    "SaveMidi": "Save MIDI",
}
