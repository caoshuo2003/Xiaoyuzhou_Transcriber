"""一站式转译脚本 - 提取信息 → 下载 → 预处理 → 转译 → 导出。

支持两种输入方式：
  1. 命令行参数：python run.py https://www.xiaoyuzhoufm.com/episode/xxx
  2. 配置文件：在 config.toml 的 [xiaoyuzhou] 段中设置 episode_url

程序会自动从 episode 页面提取 MP3 资源链接和播客标题，无需手动查找。
"""

from src.downloader import download_audio, extract_episode_info, slugify
from src.audio_processor import convert_for_whisper
from src.transcriber import transcribe, save_transcript
import os
import sys
import time
from pathlib import Path

# ============================================================
# 0. 初始化：GPU DLL 补丁 + 获取输入 URL
# ============================================================

# 核心补丁：强行把虚拟环境里藏着的 nvidia dll 路径塞进 Windows 的动态库搜寻范围中
venv_site_packages = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages")
nvidia_paths = [
    os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
    os.path.join(venv_site_packages, "nvidia", "cudnn", "bin"),
]

for path in nvidia_paths:
    if os.path.exists(path):
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        try:
            os.add_dll_directory(path)
        except AttributeError:
            pass

# 获取输入 URL
input_url = None
if len(sys.argv) > 1:
    input_url = sys.argv[1].strip()
else:
    # 尝试从配置文件读取
    try:
        if sys.version_info >= (3, 11):
            import tomllib
        else:
            import tomli as tomllib
        with open("config.toml", "rb") as f:
            config = tomllib.load(f)
        input_url = config.get("xiaoyuzhou", {}).get("episode_url", "").strip()
    except Exception:
        pass

# 如果没有配置，交互式输入
if not input_url:
    print("请输入小宇宙 episode 页面链接（或直接输入 MP3 链接）：")
    print("  例: https://www.xiaoyuzhoufm.com/episode/6a154c95463ccdb07096d1ce")
    input_url = input("> ").strip()

if not input_url:
    print("错误：未提供输入链接。")
    sys.exit(1)

# ============================================================
# 步骤 1/4：提取信息（如果是 episode 页面则自动获取 MP3 链接）
# ============================================================

pipe_start = time.time()
title = "episode"
audio_url = input_url

print("=" * 50)
print("[1/4] 提取信息")

if "xiaoyuzhoufm.com/episode/" in input_url:
    info = extract_episode_info(input_url)
    if info is None:
        print("无法从页面提取音频信息，请确认链接是否正确。")
        sys.exit(1)
    audio_url = info["audio_url"]
    title = info["title"]
else:
    print(f"使用直链: {audio_url}")
    print(f"标题: {title} (使用默认)")

# ============================================================
# 步骤 2/4：下载音频
# ============================================================

print("=" * 50)
print("[2/4] 下载音频")

mp3 = download_audio(audio_url, title=title)

# ============================================================
# 步骤 3/4：音频预处理（MP3 → 16kHz WAV）
# ============================================================

print("=" * 50)
print("[3/4] 音频预处理 (MP3 → WAV)")

wav = convert_for_whisper(mp3)

# ============================================================
# 步骤 4/4：转译 + 导出
# ============================================================

print("=" * 50)
print("[4/4] 语音转译")

segments = transcribe(
    wav,
    model_size="models/large-v3",
    language="auto",
    device="auto",
    compute_type="int8_float16",
)

# 导出结果
print("=" * 50)
print("导出文稿")

basename = slugify(title)

save_transcript(segments, fmt="txt", with_timestamps=True, basename=basename, title=title)

# ============================================================
# 完成
# ============================================================

total_elapsed = time.time() - pipe_start
duration = segments[-1]["end"] if segments else 0
print("=" * 50)
print(f"全部完成！")
print(f"  播客: {title}")
print(f"  时长: {duration:.0f} 秒 ({duration/60:.1f} 分钟)")
print(f"  段落: {len(segments)} 段")
print(f"  总耗时: {total_elapsed:.0f} 秒")
