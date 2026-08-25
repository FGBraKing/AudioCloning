"""Gradio WebUI 组件定义。"""

from __future__ import annotations

from typing import Any

import gradio as gr

from audioclone.config import MAX_TEXT_LENGTH

# 各 Tab 暴露给外层绑定的组件 key（Gradio 组件类型不统一，用 Any 便于事件绑定）
ComponentGroup = dict[str, Any]


def _char_count_html(text: str) -> str:
    count = len(text) if text else 0
    color = "red" if count > MAX_TEXT_LENGTH else "black"
    return f"字符数: <span style='color:{color}'>{count}</span> / {MAX_TEXT_LENGTH}"


def build_zero_shot_tab() -> ComponentGroup:
    """构建 Tab1: 声音克隆 (零样本)。"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 参考音频")
            prompt_wav_upload = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="上传参考音频 (wav/mp3/flac, 1~30秒)",
            )
            prompt_wav_record = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="或录制参考音频",
            )
            prompt_text = gr.Textbox(
                label="参考文本 (可选，提升克隆精度)",
                lines=1,
                placeholder="请输入参考音频对应的文字内容...",
                value="",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 合成文本")
            tts_text = gr.Textbox(
                label="输入合成文本",
                lines=3,
                placeholder=f"请输入需要合成的文本 (最大 {MAX_TEXT_LENGTH} 字)...",
                value="你好，欢迎使用个人声音克隆系统，我是由AI生成的语音助手。",
            )
            char_count = gr.Markdown(_char_count_html(""))

            with gr.Row():
                speed = gr.Number(value=1.0, label="语速", minimum=0.5, maximum=2.0, step=0.1)

            with gr.Row():
                generate_btn = gr.Button("开始合成", variant="primary", scale=2)
                clear_btn = gr.Button("清空", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 合成结果")
            audio_output = gr.Audio(label="合成音频", autoplay=True)

    tts_text.change(fn=_char_count_html, inputs=[tts_text], outputs=[char_count])

    return {
        "tts_text": tts_text,
        "prompt_wav_upload": prompt_wav_upload,
        "prompt_wav_record": prompt_wav_record,
        "prompt_text": prompt_text,
        "speed": speed,
        "generate_btn": generate_btn,
        "clear_btn": clear_btn,
        "audio_output": audio_output,
    }


def build_cross_lingual_tab() -> ComponentGroup:
    """构建 Tab2: 跨语言合成。"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 参考音频")
            cl_prompt_wav = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="上传参考音频 (源语言音色)",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 合成文本 (目标语言)")
            cl_tts_text = gr.Textbox(
                label="输入合成文本 (使用目标语言书写)",
                lines=3,
                placeholder="Enter the text in target language...",
                value="Hello, this is a cross-lingual voice cloning demonstration.",
            )

            with gr.Row():
                cl_speed = gr.Number(value=1.0, label="语速", minimum=0.5, maximum=2.0, step=0.1)

            with gr.Row():
                cl_generate_btn = gr.Button("开始合成", variant="primary", scale=2)
                cl_clear_btn = gr.Button("清空", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 合成结果")
            cl_audio_output = gr.Audio(label="合成音频", autoplay=True)

    return {
        "tts_text": cl_tts_text,
        "prompt_wav": cl_prompt_wav,
        "speed": cl_speed,
        "generate_btn": cl_generate_btn,
        "clear_btn": cl_clear_btn,
        "audio_output": cl_audio_output,
    }


def build_instruct_tab() -> ComponentGroup:
    """构建 Tab3: 自然语言控制。"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 参考音频")
            inst_prompt_wav = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="上传参考音频 (音色来源)",
            )
            inst_instruct = gr.Textbox(
                label="自然语言指令",
                lines=2,
                placeholder="例如: 请用开心的语气说 / 请用广东话表达 / 请尽可能快地说",
                value="请用开心的语气说",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 合成文本")
            inst_tts_text = gr.Textbox(
                label="输入合成文本",
                lines=3,
                placeholder="请输入需要合成的文本...",
                value="今天天气真好，适合出去走走，感受一下阳光和微风。",
            )

            with gr.Row():
                inst_speed = gr.Number(value=1.0, label="语速", minimum=0.5, maximum=2.0, step=0.1)

            with gr.Row():
                inst_generate_btn = gr.Button("开始合成", variant="primary", scale=2)
                inst_clear_btn = gr.Button("清空", scale=1)

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 合成结果")
            inst_audio_output = gr.Audio(label="合成音频", autoplay=True)

    return {
        "tts_text": inst_tts_text,
        "prompt_wav": inst_prompt_wav,
        "instruct_text": inst_instruct,
        "speed": inst_speed,
        "generate_btn": inst_generate_btn,
        "clear_btn": inst_clear_btn,
        "audio_output": inst_audio_output,
    }


def build_system_status_panel() -> ComponentGroup:
    """构建系统状态面板。"""
    with gr.Accordion("⚙️ 系统状态", open=True):
        with gr.Row():
            tts_status = gr.Markdown("🔄 TTS 模型: 加载中...")
            asr_status = gr.Markdown("🔄 ASR 服务: 检测中...")
        with gr.Row():
            device_info = gr.Markdown("框架: 检测中...")
            server_info = gr.Markdown("服务: 初始化...")

    return {
        "tts_status": tts_status,
        "asr_status": asr_status,
        "device_info": device_info,
        "server_info": server_info,
    }


def build_asr_tab() -> ComponentGroup:
    """构建 Tab4: 语音识别 (ASR)。"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 音频输入")
            asr_audio_upload = gr.Audio(
                sources=["upload"],
                type="filepath",
                label="上传音频文件 (wav/mp3/flac/ogg/m4a)",
            )
            asr_audio_record = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="或录制音频",
            )

        with gr.Column(scale=2):
            gr.Markdown("### 识别结果")
            asr_text_output = gr.Textbox(
                label="识别文本",
                lines=6,
                placeholder="点击“开始识别”后显示结果...",
                value="",
            )
            asr_info = gr.Markdown("")

            with gr.Row():
                asr_generate_btn = gr.Button("开始识别", variant="primary", scale=2)
                asr_clear_btn = gr.Button("清空", scale=1)

    return {
        "audio_upload": asr_audio_upload,
        "audio_record": asr_audio_record,
        "text_output": asr_text_output,
        "info": asr_info,
        "generate_btn": asr_generate_btn,
        "clear_btn": asr_clear_btn,
    }


def build_multitask_tab() -> ComponentGroup:
    """构建 Tab5: 多任务实验台（T2A / A2A / 音频分类 / VAD）。

    各子任务是否可用取决于当前后端是否具备对应能力；
    使用 ``dummy-omni`` 全能力后端可完整体验。
    """
    gr.Markdown(
        "### 🧪 多任务实验台\n"
        "以下任务按后端能力动态可用；使用 `dummy-omni` 全能力后端可全部体验，"
        "否则将提示“当前后端不支持该能力”。"
    )
    with gr.Row():
        with gr.Column():
            gr.Markdown("### 文本转音频 (T2A)")
            t2a_text = gr.Textbox(
                label="文本提示（描述声音/音乐/音效）",
                lines=2,
                value="轻柔的钢琴背景音乐",
            )
            t2a_duration = gr.Number(value=2.0, label="时长(秒)", minimum=0.5, maximum=30)
            t2a_generate = gr.Button("生成", variant="primary")
            t2a_output = gr.Audio(label="生成音频")

        with gr.Column():
            gr.Markdown("### 音频到音频 (A2A)")
            a2a_audio = gr.Audio(sources=["upload"], type="filepath", label="上传音频")
            a2a_instruct = gr.Textbox(label="转换指令（如“改成男声”）", value="改成男声", lines=1)
            a2a_generate = gr.Button("转换", variant="primary")
            a2a_output = gr.Audio(label="转换结果")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### 音频分类")
            cls_audio = gr.Audio(sources=["upload"], type="filepath", label="上传音频")
            cls_generate = gr.Button("分类", variant="primary")
            cls_output = gr.Textbox(label="分类结果", lines=4, interactive=False)

        with gr.Column():
            gr.Markdown("### 语音活动检测 (VAD)")
            vad_audio = gr.Audio(sources=["upload"], type="filepath", label="上传音频")
            vad_generate = gr.Button("检测", variant="primary")
            vad_output = gr.Textbox(label="语音片段 (start-end)", lines=4, interactive=False)

    return {
        "t2a_text": t2a_text,
        "t2a_duration": t2a_duration,
        "t2a_generate": t2a_generate,
        "t2a_output": t2a_output,
        "a2a_audio": a2a_audio,
        "a2a_instruct": a2a_instruct,
        "a2a_generate": a2a_generate,
        "a2a_output": a2a_output,
        "cls_audio": cls_audio,
        "cls_generate": cls_generate,
        "cls_output": cls_output,
        "vad_audio": vad_audio,
        "vad_generate": vad_generate,
        "vad_output": vad_output,
    }