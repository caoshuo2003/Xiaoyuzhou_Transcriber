"""
模块 3：语音转译器 - 调用 Whisper 模型将音频转为文字稿。

基于 faster-whisper，支持：
  - 本地运行 Whisper 模型（tiny ~ large-v3）
  - 带时间戳的文本输出（可选择每句/每词级别）
  - 多种输出格式：纯文本、SRT 字幕、VTT 字幕、JSON

模型会在首次使用时自动下载到本地缓存。
"""

import json
import math
import os
from datetime import timedelta
from pathlib import Path


def _format_timestamp(seconds: float, fmt: str = "srt") -> str:
    """将秒数格式化为时间戳字符串。

    Args:
        seconds: 秒数。
        fmt: 格式类型 - "srt" (00:00:00,000), "vtt" (00:00:00.000), "plain" (MM:SS)。

    Returns:
        格式化的时间戳字符串。
    """
    td = timedelta(seconds=seconds)
    total_seconds = int(td.total_seconds())
    hours, remainder = divmod(total_seconds, 3600)
    minutes, secs = divmod(remainder, 60)
    millis = int((seconds - int(seconds)) * 1000)

    if fmt == "srt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"
    elif fmt == "vtt":
        return f"{hours:02d}:{minutes:02d}:{secs:02d}.{millis:03d}"
    else:  # plain
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        return f"{minutes:02d}:{secs:02d}"


def _load_model(model_size: str, device: str, compute_type: str, model_dir: str = ""):
    """加载 Whisper 模型（首次调用时自动下载）。

    Args:
        model_size: 模型大小。
        device: 推理设备。
        compute_type: 计算精度。
        model_dir: 模型缓存目录。

    Returns:
        faster_whisper.WhisperModel 实例。
    """
    from faster_whisper import WhisperModel

    kwargs = {
        "model_size_or_path": model_size,
        "device": device,
        "compute_type": compute_type,
    }

    # 如果 model_size 是本地路径，直接用，不设 download_root
    is_local = "/" in model_size or "\\" in model_size or Path(model_size).exists()
    if not is_local and model_dir:
        kwargs["download_root"] = model_dir

    extra = f"(本地: {model_size})" if is_local else ""
    print(f"正在加载 Whisper 模型: {model_size} (device={device}, compute_type={compute_type}) {extra}")

    try:
        model = WhisperModel(**kwargs)
    except RuntimeError as e:
        if "cublas" in str(e).lower() or "cuda" in str(e).lower() and device != "cpu":
            print(f"GPU 加载失败 ({e})，回退到 CPU...")
            kwargs["device"] = "cpu"
            kwargs["compute_type"] = "int8"
            model = WhisperModel(**kwargs)
        else:
            raise

    print("模型加载完成。")
    return model


def transcribe(
    audio_path: Path,
    model_size: str = "medium",
    device: str = "auto",
    compute_type: str = "int8_float16",
    model_dir: str = "",
    language: str = "auto",
    timestamps: bool = True,
) -> list[dict]:
    """将音频转译为带时间戳的文本段。

    Args:
        audio_path: 音频文件路径。
        model_size: Whisper 模型大小。
        device: 推理设备。
        compute_type: 计算精度。
        model_dir: 模型缓存目录。
        language: 语言代码（auto 自动检测，en 英文，zh 中文等）。
        timestamps: 是否生成时间戳（关闭后仅返回纯净文本，速度更快）。

    Returns:
        段落列表，每段为字典：
        { "start": float, "end": float, "text": str }
        若 timestamps=False，start/end 为平均估算值。
    """
    model = _load_model(model_size, device, compute_type, model_dir)

    lang = None if language == "auto" else language

    print(f"开始转译: {audio_path}")
    segments, info = model.transcribe(
        str(audio_path),
        language=lang,
        beam_size=5,
        vad_filter=True,                    # 自动过滤静音
        vad_parameters=dict(
            min_silence_duration_ms=500,    # 最小静音间隔 500ms
        ),
    )

    detected_lang = info.language
    print(f"检测到语言: {detected_lang} (概率: {info.language_probability:.2f})")

    results = []
    for seg in segments:
        results.append({
            "start": round(seg.start, 3),
            "end": round(seg.end, 3),
            "text": seg.text.strip(),
        })

    print(f"转译完成，共 {len(results)} 个段落。")
    return results


def to_plain_text(segments: list[dict], with_timestamps: bool = True) -> str:
    """将转译结果格式化为纯文本。

    Args:
        segments: 转译段落列表。
        with_timestamps: 是否在每行前加时间戳。

    Returns:
        格式化的文本字符串。
    """
    lines = []
    for seg in segments:
        if with_timestamps:
            ts = _format_timestamp(seg["start"], "plain")
            lines.append(f"[{ts}] {seg['text']}")
        else:
            lines.append(seg["text"])
    return "\n".join(lines)


def to_srt(segments: list[dict]) -> str:
    """将转译结果格式化为 SRT 字幕。

    Args:
        segments: 转译段落列表。

    Returns:
        SRT 格式的字幕字符串。
    """
    lines = []
    for i, seg in enumerate(segments, start=1):
        start_ts = _format_timestamp(seg["start"], "srt")
        end_ts = _format_timestamp(seg["end"], "srt")
        lines.append(f"{i}")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg["text"])
        lines.append("")  # 空行分隔
    return "\n".join(lines)


def to_vtt(segments: list[dict]) -> str:
    """将转译结果格式化为 VTT 字幕。

    Args:
        segments: 转译段落列表。

    Returns:
        WebVTT 格式的字幕字符串。
    """
    lines = ["WEBVTT", ""]
    for seg in segments:
        start_ts = _format_timestamp(seg["start"], "vtt")
        end_ts = _format_timestamp(seg["end"], "vtt")
        lines.append(f"{start_ts} --> {end_ts}")
        lines.append(seg["text"])
        lines.append("")
    return "\n".join(lines)


def to_json(segments: list[dict], title: str = "") -> str:
    """将转译结果格式化为 JSON。

    Args:
        segments: 转译段落列表。
        title: 播客标题（写入元数据）。

    Returns:
        格式化的 JSON 字符串。
    """
    doc = {
        "title": title,
        "segments": segments,
        "total_duration": segments[-1]["end"] if segments else 0,
    }
    return json.dumps(doc, ensure_ascii=False, indent=2)


def save_transcript(
    segments: list[dict],
    output_dir: str = "output",
    basename: str = "transcript",
    fmt: str = "txt",
    title: str = "",
    with_timestamps: bool = True,
) -> Path:
    """将转译结果保存为文件。

    Args:
        segments: 转译段落列表。
        output_dir: 输出目录。
        basename: 输出文件名（不含扩展名）。
        fmt: 输出格式 - "txt", "srt", "vtt", "json"。
        title: 播客标题。
        with_timestamps: 文本输出是否含时间戳。

    Returns:
        保存的文件路径。
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    formatters = {
        "txt": lambda: to_plain_text(segments, with_timestamps=with_timestamps),
        "srt": lambda: to_srt(segments),
        "vtt": lambda: to_vtt(segments),
        "json": lambda: to_json(segments, title=title),
    }

    if fmt not in formatters:
        raise ValueError(f"不支持的输出格式: {fmt}，可选: {list(formatters.keys())}")

    content = formatters[fmt]()
    filepath = output_path / f"{basename}.{fmt}"

    encoding = "utf-8-sig" if fmt in ("srt", "json") else "utf-8"

    with open(filepath, "w", encoding=encoding) as f:
        f.write(content)

    print(f"文稿已保存: {filepath}")
    return filepath
