"""
模块 1：音频下载器 - 下载小宇宙播客原始音频。

支持两种输入：
  1. 直接传入 MP3 URL（如 media.xyzcdn.net 的链接）
  2. 传入小宇宙 episode 链接，自动提取音频 URL 和标题
"""

import hashlib
import re
import time
from pathlib import Path
from urllib.parse import urlparse

import requests


def slugify(name: str, max_len: int = 60) -> str:
    """将标题转换为安全的文件名。"""
    name = re.sub(r"[^\w\s-]", "", name).strip()
    name = re.sub(r"[-\s]+", "-", name)
    return name[:max_len].strip("-")


def extract_episode_info(episode_url: str, timeout: int = 30) -> dict:
    """从小宇宙 episode 页面提取音频 URL 和标题。

    通过解析页面 HTML 中的 og:audio 和 og:title meta 标签获取信息。

    Args:
        episode_url: 小宇宙 episode 页面链接（如 https://www.xiaoyuzhoufm.com/episode/xxx）。
        timeout: 请求超时时间（秒）。

    Returns:
        {"audio_url": str, "title": str} 或 None（提取失败时）。

    Raises:
        requests.HTTPError: 页面请求失败时抛出。
    """
    print(f"正在访问页面: {episode_url}")

    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        )
    }
    resp = requests.get(episode_url, headers=headers, timeout=timeout)
    resp.raise_for_status()
    html = resp.text

    # 提取音频 URL（og:audio meta 标签）
    audio_match = re.search(
        r'<meta\s+property="og:audio"\s+content="([^"]+)"', html
    )
    if not audio_match:
        print("错误: 未在页面中找到音频资源链接。")
        return None

    audio_url = audio_match.group(1)
    print(f"找到音频资源: {audio_url}")

    # 提取标题（og:title meta 标签）
    title_match = re.search(
        r'<meta\s+property="og:title"\s+content="([^"]+)"', html
    )
    title = title_match.group(1) if title_match else "episode"
    print(f"播客标题: {title}")

    return {"audio_url": audio_url, "title": title}


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
        filename = f"{slugify(title)}-{filename}"

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
