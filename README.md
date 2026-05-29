# dd-song-miner-llm

基于 Whisper ASR + LLM 的歌曲提取工具。

## 原理

1. **Whisper ASR** 转写视频音频，得到带时间戳的文本
2. **LLM** 分析完整转写文本，识别完整歌曲及其 ASR 段落范围
3. 根据 ASR 时间戳确定歌曲时间范围
4. 前后加 padding，导出报告和片段

## 安装

```bash
pip install -r requirements.txt

# NVIDIA GPU 支持 (可选，faster-whisper / CTranslate2 使用 CUDA 12 + cuDNN 9)
pip install -r requirements-cu12.txt
```

## 使用

```bash
# 生成配置文件
python -m dd_song_miner_llm init-config --out config.yaml

# 编辑 config.yaml，填入 LLM API key

# 运行
python -m dd_song_miner_llm run 视频.mp4 --config config.yaml

# 按目录批量运行，处理成功的源视频文件夹会写入完成标记
python -m dd_song_miner_llm batch-run D:\ddtv2 --config config.yaml --result-root D:\song-results

# 手动调整 songs.csv 的 start/end 后重新切片
python -m dd_song_miner_llm manual-cut runs\某次运行 --config config.yaml --csv runs\某次运行\04_reports\songs.csv
```

## CLI 参数

```bash
python -m dd_song_miner_llm run 视频.mp4 [选项]

# 常用选项
--config config.yaml        # 配置文件
--asr-model medium          # Whisper 模型
--asr-language zh            # 语言
--llm-model gpt-4o           # LLM 模型
--llm-api-key sk-xxx         # API key
--llm-base-url https://...   # 自定义 API 地址
--padding-before 5.0         # 开始前 padding
--padding-after 8.0          # 结束后 padding
--no-video-clips             # 不导出视频
--video-codec auto           # auto=nv > intel > amd > cpu；copy=不重编码
--audio-bitrate-kbps 320     # 需要转码时的音频码率
```

## 批量运行

`batch-run` 会递归扫描指定目录下的视频文件。每个含视频的子文件夹作为一个处理单元；如果该子文件夹里已有 `.dd_song_miner_done.json`，以后会跳过这个文件夹。

```bash
python -m dd_song_miner_llm batch-run \\server\share\ddtv2 \
  --config config.yaml \
  --work-root runs\batch \
  --result-root D:\song-results
```

网络目录、中文/日文路径会先 staging 到运行目录的 `00_input`，避免 Windows 版 FFmpeg 路径编码问题。处理完成后，运行结果会复制到 `--result-root`。

常用参数：

```bash
--marker .dd_song_miner_done.json # 源视频子文件夹里的完成标记
--extensions mp4,mkv,flv          # 限定扫描的视频格式
--work-root runs\batch            # 本地工作目录，建议放本机磁盘
--result-root D:\song-results     # 最终结果归档目录
```

对网络盘建议把 `--work-root` 放在本机 SSD，减少 ASR 和 FFmpeg 对网络文件夹的反复读写；流程结束后会把完整运行目录复制到 `--result-root`。

## Windows 路径兼容

Windows 下中文、日文、空格路径或 UNC 网络路径可能在 Python/FFmpeg 之间发生编码问题。运行时会自动判断输入路径：

- ASCII 本地路径：直接使用原文件。
- 非 ASCII 路径或 `\\server\share` 网络路径：在运行目录 `00_input` 下创建 ASCII 文件名的 staging 输入。
- staging 会优先创建硬链接；如果跨盘、网络盘或权限不允许硬链接，则复制文件。

因此命令可以直接传原始路径：

```bash
python -m dd_song_miner_llm run "C:\Users\bakaz\OneDrive - bakazdev\ddtv2\綾音Aya\video.mp4" --config config.yaml
```

如果旧版终端把命令行参数本身传乱码，建议使用 PowerShell 7 / Windows Terminal，或用 `batch-run` 扫描目录，让程序从文件系统读取路径。

## FFmpeg 编码策略

视频导出默认使用：

```yaml
output:
  video_codec: auto
```

`auto` 会按 `NVIDIA NVENC > Intel QSV > AMD AMF > CPU libx264` 选择 FFmpeg 支持的编码器，并在失败时继续回退。也可以显式指定：

```bash
--video-codec nv      # h264_nvenc
--video-codec intel   # h264_qsv
--video-codec amd     # h264_amf
--video-codec cpu     # libx264
--video-codec copy    # 不重编码，直接复制原视频流
```

音频默认导出 `m4a` 并复制源 AAC 音轨，不降低音质。需要转码到 `mp3` 等格式时，使用 `audio_bitrate_kbps` 控制码率，默认 `320k`。

```yaml
output:
  audio_extension: m4a
  audio_bitrate_kbps: 320
```

## 手动重切

先打开某次运行的 `04_reports/songs.csv`，手动调整 `start` / `end`，可用 `HH:MM:SS` 或秒数。然后执行：

```bash
python -m dd_song_miner_llm manual-cut runs\某次运行 --config config.yaml
```

手动输出默认写到 `05_manual`，不会覆盖原始 LLM 结果。

手动 CSV 至少需要这些列：

```csv
index,start,end,title,artist
1,00:23:02,00:28:31,一半一半,洛天依
```

`start` / `end` 支持 `HH:MM:SS`、`MM:SS` 或秒数。`manual-cut` 会读取手改后的时间，重新导出音频、视频和一份新的报告。

## Match 上下文

每次 LLM 识别后，`02_asr/llm` 下会生成：

- `matches.json`：LLM 返回的原始歌曲 match。
- `match_context.json`：每个 match 的 `segment_indices`、起止秒数、命中的 Whisper 段落，以及前后上下文。
- `match_context.csv`：同样内容的表格版本，方便人工筛查和微调 prompt。

默认每个 match 前后各带 10 条 Whisper segment，可在配置里调整：

```yaml
output:
  match_context_segments: 10
```

## 配置兼容

支持任何兼容 OpenAI API 格式的 LLM：

```yaml
llm:
  api_key: your-key
  base_url: https://api.your-provider.com/v1
  model: your-model-name
  batch_size: null  # 默认整段提交；正整数表示按 ASR 段数分批
```

### DeepSeek 示例

歌曲抽取任务建议使用非推理模型，避免把输出预算消耗在 reasoning 内容上：

```yaml
llm:
  api_key: null
  api_key_env: DEEPSEEK_API_KEY
  base_url: https://api.deepseek.com
  model: deepseek-chat
  temperature: 0.1
  max_tokens: 4096
  max_completion_tokens: null
  retry_empty_with_reasoning: true
  reasoning_followup_rounds: 2
  reasoning_followup_max_tokens: 8192
  batch_size: null
```

## 免费 API 资源

以下平台提供**永久免费**的 OpenAI 兼容模型（非试用额度）：

| 平台 | 免费模型 | 限制 | 说明 |
|------|----------|------|------|
| **OpenRouter** | 25+ 免费模型 | 每天 50 请求 | 聚合多个提供商，OpenAI 兼容格式 |
| **Groq** | Llama、Mixtral 等 | 每分钟 30 请求 | 超快推理速度 |
| **Together AI** | 部分免费模型 | 有限制 | 支持多种开源模型 |
| **智谱 AI** | GLM-4-Flash | 有限制 | 国内平台，中文效果好 |
| **小米 MiMo** | mimo-v2.5 | Token 计费 | 国内平台，性价比高 |

### 使用方法

在配置文件中设置对应的 `base_url` 和 `api_key`：

```yaml
# OpenRouter 免费模型示例
llm:
  api_key: your-openrouter-key
  base_url: https://openrouter.ai/api/v1
  model: meta-llama/llama-3.1-8b-instruct:free

# Groq 免费模型示例
llm:
  api_key: your-groq-key
  base_url: https://api.groq.com/openai/v1
  model: llama-3.1-8b-instant

# 小米 MiMo 示例
llm:
  api_key: your-mimo-key
  base_url: https://token-plan-cn.xiaomimimo.com/v1
  model: mimo-v2.5
```

## 多 Provider 回退机制

本工具支持配置多个 LLM 提供商，当主提供商失败时自动回退到备用提供商，提高可用性。

### 配置示例

```yaml
# 多 provider 配置
llm:
  # 主提供商
  api_key: your-primary-key
  base_url: https://openrouter.ai/api/v1
  model: meta-llama/llama-3.1-8b-instruct:free
  
  # 备用提供商列表（可选）
  fallbacks:
    - api_key: your-groq-key
      base_url: https://api.groq.com/openai/v1
      model: llama-3.1-8b-instant
    - api_key: your-backup-key
      base_url: https://api.together.xyz/v1
      model: meta-llama/Llama-3-8b-chat-hf
```

### 回退策略

1. **自动切换**：当主提供商请求失败时，自动尝试下一个备用提供商
2. **负载均衡**：可在多个提供商之间轮询请求
3. **错误重试**：单个提供商失败时自动重试，超过次数后切换

### CLI 参数

```bash
# 使用多个 API key
python -m dd_song_miner_llm run 视频.mp4 \
  --llm-api-key key1,key2,key3 \
  --llm-base-url url1,url2,url3 \
  --llm-model model1,model2,model3
```

## 输出结构

```
runs/
├── 00_input/
│   └── input_xxx.mp4        # 非 ASCII 或网络路径的 staging 输入
├── 01_audio/
│   └── source.wav
├── 02_asr/
│   ├── transcript.json
│   └── llm/
│       ├── matches.json
│       ├── match_context.csv
│       └── match_context.json
├── 03_clips/
│   ├── audio/
│   └── video/
├── 04_reports/
│   ├── songs.csv
│   └── songs.json
└── manifest.json
```

## License

MIT
