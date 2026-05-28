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
├── 01_audio/
│   └── source.wav
├── 02_asr/
│   └── transcript.json
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
