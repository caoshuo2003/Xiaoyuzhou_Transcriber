"""
模块 2：音频处理器 - 调用 ffmpeg 对音频进行格式转换和预处理。

主要功能：
  - 将任意格式音频转换为 Whisper 所需的 16kHz 单声道 WAV
  - 可选裁剪音频片段
  - 音量标准化
"""

import subprocess
from pathlib import Path


def ensure_ffmpeg() -> bool:
    """检查 ffmpeg 是否可用。"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            check=True,
        )
        return True
    except (FileNotFoundError, subprocess.CalledProcessError):
        return False


def convert_for_whisper(
    input_path: Path,
    output_path: Path | None = None,
    sample_rate: int = 16000,
    channels: int = 1,
    trim_start: float = 0,
    trim_end: float | None = None,
) -> Path:
    """将音频转换为 Whisper 模型所需格式（16kHz 单声道 WAV）。

    Args:
        input_path: 输入文件路径。
        output_path: 输出文件路径（不指定则自动生成 .wav 文件）。
        sample_rate: 目标采样率。
        channels: 目标声道数。
        trim_start: 裁剪起始时间（秒）。
        trim_end: 裁剪结束时间（秒）。

    Returns:
        转换后的 WAV 文件路径。
    """
    if output_path is None:
        output_path = input_path.with_suffix(".wav")

    cmd = [
        "ffmpeg",
        "-y",                # 覆盖已存在的输出文件
        "-i", str(input_path),
    ]

    # 可选的时间裁剪
    if trim_start > 0:
        cmd.extend(["-ss", str(trim_start)])
    if trim_end is not None:
        cmd.extend(["-to", str(trim_end)])

    # 格式转换参数
    cmd.extend([
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-sample_fmt", "s16",     # 16-bit PCM
        "-acodec", "pcm_s16le",
        str(output_path),
    ])

    subprocess.run(cmd, check=True, capture_output=True)
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"音频预处理完成: {output_path} ({size_mb:.1f} MB)")
    return output_path


def trim_audio(
    input_path: Path,
    start: float,
    end: float,
    output_path: Path | None = None,
) -> Path:
    """裁剪音频片段。

    Args:
        input_path: 输入文件路径。
        start: 起始时间（秒）。
        end: 结束时间（秒）。
        output_path: 输出文件路径。

    Returns:
        裁剪后的音频文件路径。
    """
    return convert_for_whisper(
        input_path,
        output_path=output_path,
        trim_start=start,
        trim_end=end,
    )
