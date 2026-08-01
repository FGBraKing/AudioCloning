@echo off
REM ============================================================
REM AudioCloning Windows 一键安装脚本 (MLX 版)
REM 注意: MLX 仅支持 Apple Silicon，此脚本仅供参考
REM ============================================================
setlocal enabledelayedexpansion

set ENV_NAME=agent

echo ============================================================
echo   AudioCloning 安装脚本 (MLX)
echo   注意: MLX 仅支持 Apple Silicon Mac
echo ============================================================

REM 1. 激活 conda 环境
echo [1/3] 激活 conda 环境: %ENV_NAME% ...
call conda activate %ENV_NAME%

REM 2. 安装 Python 依赖
echo [2/3] 安装 Python 依赖...
pip install -r requirements.txt

REM 3. 下载模型
echo [3/3] 下载 MLX 格式模型...
python scripts\download_tts_model.py
python scripts\download_asr_model.py

echo.
echo ============================================================
echo   安装完成!
echo ============================================================
echo   启动 WebUI:  python src\app.py
echo   启动 API:    python src\api\server.py
echo   访问地址:    http://localhost:7860
echo ============================================================

endlocal
