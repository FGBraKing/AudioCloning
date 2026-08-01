#!/usr/bin/env python3
"""
ASR 功能端到端测试脚本
验证 llama-server + Qwen3-ASR-0.6B-GGUF 语音识别流程
"""

import sys
import os
import logging
import time

# 确保项目根目录在 sys.path 中
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("test_asr")


def main():
    from src.asr import ASREngine

    test_audio = os.path.join(ROOT_DIR, "asset", "zero_shot_prompt.wav")

    print("=" * 60)
    print("  ASR 功能端到端测试")
    print("=" * 60)

    # 1. 初始化引擎
    print("\n[1/4] 初始化 ASR 引擎...")
    engine = ASREngine()
    status = engine.get_status()
    print(f"  模型可用: {status['model_available']}")
    print(f"  主模型: {status['model_path']}")
    print(f"  mmproj: {status['mmproj_path']}")

    if not engine.is_model_available:
        print("\n[ERROR] 模型文件不完整，请先运行: python download_asr_model.py")
        sys.exit(1)

    # 2. 启动 llama-server
    print("\n[2/4] 启动 llama-server...")
    start = time.time()
    if not engine.start_server(timeout=90.0):
        print("\n[ERROR] llama-server 启动失败")
        sys.exit(1)
    print(f"  服务就绪，耗时 {time.time() - start:.1f}s")
    print(f"  地址: {engine.base_url}")

    # 3. 执行语音识别
    print(f"\n[3/4] 识别音频: {test_audio}")
    if not os.path.exists(test_audio):
        print(f"\n[ERROR] 测试音频不存在: {test_audio}")
        engine.stop_server()
        sys.exit(1)

    try:
        start = time.time()
        text = engine.transcribe(test_audio)
        elapsed = time.time() - start
        print(f"\n  识别结果: {text}")
        print(f"  耗时: {elapsed:.2f}s")
    except Exception as e:
        print(f"\n[ERROR] 识别失败: {e}")
        engine.stop_server()
        sys.exit(1)

    # 4. 停止服务
    print("\n[4/4] 停止 llama-server...")
    engine.stop_server()

    print("\n" + "=" * 60)
    print("  ASR 测试通过!")
    print("=" * 60)


if __name__ == "__main__":
    main()
