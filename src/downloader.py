"""
模块 1：音频下载器 - 下载小宇宙播客原始音频。

支持两种输入：
  1. 直接传入 MP3 URL（如 media.xyzcdn.net 的链接）
  2. 传入小宇宙 episode 链接，自动提取音频 URL
"""

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def _slugify(name: str, max_len: int = 60) -> str:
    """将标题转换为安全的文件名。"""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "-", name)
    return name[:max_len].strip("-")


def _extract_audio_url(episode_url: str) -> str | None:
    """从小宇宙 episode 页面提取音频 URL。

    目前需要手动传入音频 URL；后续可集成 API 自动提取。
    如果你通过浏览器开发者工具拿到了音频直链，可以直接传直链给 download_audio()。
    """
    # 暂时返回 None，保留接口将来扩展
    return None


def download_audio(
    audio_url: str,
    save_dir: str = "downloads",
    title: str = "",
    timeout: int = 600,
) -> Path:
    """下载音频文件到本地。

    Args:
        audio_url: 音频文件的直链 URL。
        save_dir: 保存目录。
        title: 播客标题（用作文件名前缀）。
        timeout: 下载超时时间（秒）。

    Returns:
        下载完成后的本地文件路径。

    Raises:
        requests.HTTPError: 下载失败时抛出。
    """
    save_path = Path(save_dir)
    save_path.mkdir(parents=True, exist_ok=True)

    # 从 URL 提取原始文件名
    parsed = urlparse(audio_url)
    filename = Path(parsed.path).name or "audio.mp3"

    if title:
        filename = f"{_slugify(title)}-{filename}"

    output_path = save_path / filename

    if output_path.exists():
        print(f"文件已存在，跳过下载: {output_path}")
        return output_path

    print(f"开始下载: {audio_url}")
    start = time.time()

    response = requests.get(audio_url, stream=True, timeout=timeout)
    response.raise_for_status()

    total = int(response.headers.get("content-length", 0))
    downloaded = 0

    with open(output_path, "wb") as f:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded / total * 100
                mb = downloaded / 1024 / 1024
                total_mb = total / 1024 / 1024
                print(f"\r进度: {pct:.1f}% ({mb:.1f}/{total_mb:.1f} MB)", end="")

    elapsed = time.time() - start
    size_mb = output_path.stat().st_size / 1024 / 1024
    print(f"\n下载完成: {output_path} ({size_mb:.1f} MB, 耗时 {elapsed:.0f}s)")

    return output_path
