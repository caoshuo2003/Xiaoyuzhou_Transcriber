"""一站式转译脚本 - 下载 → 预处理 → 转译 → 导出。"""
from src.downloader import download_audio
from src.audio_processor import convert_for_whisper
from src.transcriber import transcribe, save_transcript
import os
import sys

# 【核心补丁】强行把虚拟环境里藏着的 nvidia dll 路径塞进 Windows 的动态库搜寻范围中
venv_site_packages = os.path.join(os.path.dirname(__file__), ".venv", "Lib", "site-packages")
nvidia_paths = [
    os.path.join(venv_site_packages, "nvidia", "cublas", "bin"),
    os.path.join(venv_site_packages, "nvidia", "cudnn", "bin")
]

for path in nvidia_paths:
    if os.path.exists(path):
        # 让 Windows 的 os.path 认识它
        os.environ["PATH"] = path + os.pathsep + os.environ["PATH"]
        # 给 Python 3.8+ 专门用的底层 DLL 导入机制引路
        try:
            os.add_dll_directory(path)
        except AttributeError:
            pass
            
# 1. 下载音频
mp3 = download_audio(
    "https://media.xyzcdn.net/5e2aadff418a84a046540982/lhk73V5TjzGhXhaFr8XF_5hzwQGX.mp3",
    title="Round-Table-China",
)

# 2. 转为 Whisper 格式（需要 ffmpeg）
wav = convert_for_whisper(mp3)

# 3. 转译（使用本地已下载的模型，不会联网）
segments = transcribe(
    wav,
    model_size="models/large-v3",
    language="auto",
    device="auto",
    compute_type="int8_float16",
)

# 4. 同时导出多种格式
save_transcript(segments, fmt="txt", with_timestamps=True)   # 带时间戳文本
save_transcript(segments, fmt="srt")                          # SRT 字幕
save_transcript(segments, fmt="json")                         # JSON 结构化
