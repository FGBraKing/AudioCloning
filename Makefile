# AudioCloning 常用命令（uv 管理）
PY ?= uv run

.PHONY: setup lint type test cov webui api download-tts download-asr benchmark eval clean

setup:          ## 安装依赖（Python 3.13, 生成 .venv + uv.lock）
	uv sync

lint:           ## 静态检查（ruff）
	$(PY) ruff check .

type:           ## 类型检查（mypy）
	$(PY) mypy src/audioclone

test:           ## 单元/架构/契约/API 测试
	$(PY) pytest -m "not integration" -q

cov:            ## 覆盖率（门槛 60%）
	$(PY) pytest -m "not integration" --cov=audioclone --cov-report=term-missing

webui:          ## 启动 WebUI（dummy-omni 快速体验，无模型）
	$(PY) audioclone-webui --tts-backend dummy-omni --asr-backend dummy-omni

api:            ## 启动 REST API（默认仅本机）
	$(PY) audioclone-api

download-tts:   ## 下载 TTS 模型
	$(PY) audioclone-download-tts

download-asr:   ## 下载 ASR 模型
	$(PY) audioclone-download-asr

benchmark:      ## TTS 性能基准（冷启动/单次合成）
	$(PY) python -m tools.benchmark --backend dummy-omni --iterations 3

eval:           ## 合成质量指标（频谱距离）
	$(PY) python -m tools.eval_quality --ref asset/zero_shot_prompt.wav --synth outputs/test.wav

clean:          ## 清理缓存与产物
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov __pycache__ \
		$(shell find . -name __pycache__ -type d 2>/dev/null)
