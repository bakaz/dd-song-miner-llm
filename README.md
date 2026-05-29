# dd-song-miner-llm

基于 Whisper ASR + LLM 的直播录像歌曲提取工具。它会把视频转写成带时间戳的文本，让 LLM 从完整转写中识别歌曲段落，再按时间切出音频、视频片段，并输出可人工校正的报告。

## 工作流程

1. 用 FFmpeg 从视频提取 16 kHz 单声道 WAV。
2. 用 faster-whisper 生成带时间戳的 ASR segment。
3. 把完整 transcript 或分批 transcript 交给兼容 OpenAI API 的 LLM。
4. LLM 返回歌曲 `segment_indices`，程序按 ASR 时间戳生成候选歌曲。
5. 对歌曲前后加 padding，但不会超过相邻 ASR 句子的边界。
6. 导出 `songs.csv`、`songs.json`、match 上下文、音频片段和视频片段。
7. 可手动修改 `songs.csv` 的起止时间，再用 `manual-cut` 重新切出最终版本。

## 安装

### 1. 准备 Python

建议使用 Python 3.10 到 3.12。Windows 下推荐在项目目录创建虚拟环境：

```powershell
cd C:\Users\bakaz\Documents\dd-song-miner-llm
py -3.12 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
pip install -r requirements.txt
```

如果使用可编辑安装：

```powershell
pip install -e .
```

### 2. 准备 FFmpeg

程序会优先使用系统 PATH 里的 `ffmpeg` / `ffprobe`。如果没有系统 FFmpeg，会回退到 `imageio-ffmpeg` 自带的 FFmpeg。

建议 Windows 上安装完整 FFmpeg：

```powershell
winget install Gyan.FFmpeg
```

安装后重新打开终端，检查：

```powershell
ffmpeg -version
ffprobe -version
```

### 3. NVIDIA CUDA 可选加速

如果想用 GPU 跑 faster-whisper，安装 CUDA 12 相关 pip 包：

```powershell
pip install -r requirements-cu12.txt
```

如果本机只有 CUDA 13，仍然需要 CUDA 12 的运行时 DLL，因为当前 CTranslate2 / faster-whisper 依赖 `cublas64_12.dll`。程序检测到 CUDA 缺失时会自动回退 CPU int8，但速度会慢很多。

### 4. 准备 LLM Key

生成配置文件：

```powershell
python -m dd_song_miner_llm init-config --out config.yaml
```

推荐不要把真实 key 写进仓库文件，使用环境变量：

```powershell
$env:DEEPSEEK_API_KEY="sk-..."
```

或写入用户环境变量后重新打开终端：

```powershell
[Environment]::SetEnvironmentVariable("DEEPSEEK_API_KEY", "sk-...", "User")
```

## 快速开始

单个视频：

```powershell
python -m dd_song_miner_llm run "D:\videos\live.mp4" --config config.yaml
```

指定输出目录：

```powershell
python -m dd_song_miner_llm run "D:\videos\live.mp4" --config config.yaml --out "D:\song-runs\live_001"
```

批量处理某个目录下所有视频：

```powershell
python -m dd_song_miner_llm batch-run "D:\ddtv2" --config config.yaml --work-root "D:\song-work" --result-root "D:\song-results"
```

手动重切：

```powershell
python -m dd_song_miner_llm manual-cut "D:\song-results\某次运行" --config config.yaml
```

## 配置文件

完整配置结构如下。`config.example.yaml` 和 `config.deepseek.example.yaml` 也可以直接复制修改。

```yaml
audio:
  sample_rate: 16000
  channels: 1

asr:
  model: small
  device: auto
  compute_type: default
  language: null
  beam_size: 5
  vad_filter: true
  initial_prompt: null

llm:
  api_key: null
  api_key_env: DEEPSEEK_API_KEY
  base_url: https://api.deepseek.com
  model: deepseek-chat
  temperature: 0.1
  max_tokens: 8192
  max_completion_tokens: null
  retry_empty_with_reasoning: true
  reasoning_followup_rounds: 5
  reasoning_followup_max_tokens: 32768
  batch_size: null
  use_tools: true
  verify_with_search: true
  fallbacks: []

padding:
  before_seconds: 3.0
  after_seconds: 5.0
  min_song_seconds: 15.0
  merge_gap_seconds: 30.0

output:
  video_clips: true
  audio_segments: true
  audio_extension: m4a
  audio_bitrate_kbps: 320
  video_extension: mp4
  video_codec: auto
  match_context_segments: 10
```

### audio

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `sample_rate` | `16000` | Whisper 输入音频采样率。 |
| `channels` | `1` | Whisper 输入声道数，默认单声道。 |

### asr

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `model` | `small` | faster-whisper 模型，常用 `tiny`、`base`、`small`、`medium`、`large-v3`。 |
| `device` | `auto` | `auto`、`cuda` 或 `cpu`。CUDA 不可用时会自动回退 CPU int8。 |
| `compute_type` | `default` | faster-whisper compute type，例如 `float16`、`int8`。 |
| `language` | `null` | 语言提示。可设 `zh`、`ja`、`en`，`null` 表示自动识别。 |
| `beam_size` | `5` | Whisper beam size。 |
| `vad_filter` | `true` | 是否启用 VAD 静音过滤。 |
| `initial_prompt` | `null` | Whisper 初始提示词，可用于提示中日混合、歌词场景。 |

### llm

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `api_key` | `null` | LLM API key。建议用 `api_key_env`，避免明文写入文件。 |
| `api_key_env` | `null` | 从环境变量读取 key，例如 `DEEPSEEK_API_KEY`。 |
| `base_url` | `null` | OpenAI 兼容 API 地址。DeepSeek 为 `https://api.deepseek.com`。 |
| `model` | `gpt-4o` | 模型名。歌曲提取建议优先用非 reasoning 模型。 |
| `temperature` | `0.3` | 温度，建议低一些。 |
| `max_tokens` | `4096` 或配置值 | 普通模型输出 token 上限。 |
| `max_completion_tokens` | `null` | 某些 reasoning 模型需要此字段；设置后优先于 `max_tokens`。 |
| `retry_empty_with_reasoning` | `true` | 当模型把有效内容放在 reasoning 或输出为空时，自动发 follow-up。 |
| `reasoning_followup_rounds` | `2` 或配置值 | reasoning follow-up 最大轮数。 |
| `reasoning_followup_max_tokens` | `8192` 或配置值 | follow-up 输出 token 上限。 |
| `batch_size` | `null` | `null` 表示整段 transcript 一次提交；正整数表示按 ASR segment 数分批。 |
| `use_tools` | `true` | 是否允许 LLM 调用歌词搜索工具。 |
| `verify_with_search` | `true` | 是否用搜索结果辅助校验歌曲名。 |
| `fallbacks` | `[]` | 主 provider 失败时的备用 provider 列表。 |

DeepSeek 推荐配置：

```yaml
llm:
  api_key: null
  api_key_env: DEEPSEEK_API_KEY
  base_url: https://api.deepseek.com
  model: deepseek-chat
  temperature: 0.1
  max_tokens: 8192
  batch_size: null
  use_tools: true
  verify_with_search: true
```

### padding

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `before_seconds` | `3.0` | 歌曲开始前补多少秒。不会超过上一条 ASR segment 的结束时间。 |
| `after_seconds` | `5.0` | 歌曲结束后补多少秒。不会超过下一条 ASR segment 的开始时间。 |
| `min_song_seconds` | `15.0` | 低于这个时长的候选会被丢弃。 |
| `merge_gap_seconds` | `30.0` | 同名歌曲间隔小于该值时合并。 |

### output

| 字段 | 默认值 | 说明 |
| --- | --- | --- |
| `video_clips` | `true` | 是否导出视频片段。 |
| `audio_segments` | `true` | 是否导出音频片段。 |
| `audio_extension` | `m4a` | 音频格式。`m4a/aac` 默认复制源 AAC 音轨；`mp3` 会转码。 |
| `audio_bitrate_kbps` | `320` | 音频转码码率。只在需要转码时使用。 |
| `video_extension` | `mp4` | 视频片段格式。 |
| `video_codec` | `auto` | `auto`、`copy`、`nv`、`intel`、`amd`、`cpu`。 |
| `match_context_segments` | `10` | 每个 match 前后额外输出多少条 Whisper segment。 |

## 单视频运行

```powershell
python -m dd_song_miner_llm run "D:\videos\live.mp4" --config config.yaml
```

常用覆盖参数：

```powershell
--asr-model medium
--asr-language zh
--llm-model deepseek-chat
--llm-base-url https://api.deepseek.com
--padding-before 5
--padding-after 8
--export-audio m4a
--export-video mp4
--video-codec auto
--audio-bitrate-kbps 320
--no-video-clips
```

## 批量运行

`batch-run` 用于按天或定时处理目录。它会递归扫描 `input_root` 下的视频文件，并按“视频所在子文件夹”写完成标记。

```powershell
python -m dd_song_miner_llm batch-run "\\server\share\ddtv2" `
  --config config.yaml `
  --work-root "D:\song-work" `
  --result-root "D:\song-results" `
  --marker ".dd_song_miner_done.json" `
  --extensions "mp4,mkv,flv"
```

处理规则：

- 每个含视频的子文件夹是一个处理单元。
- 如果子文件夹里已有 `.dd_song_miner_done.json`，这个子文件夹会跳过。
- 只有该子文件夹内所有视频都处理成功，才写完成标记。
- `--work-root` 是实际运行目录，建议放本机 SSD。
- `--result-root` 是归档目录，流程结束后会复制完整 run 结果过去。
- 网络目录或中文/日文路径会自动 staging 到 run 目录的 `00_input`，降低 FFmpeg 路径编码风险。

适合 ddtv2 这类结构：

```text
D:\ddtv2\
└── 92450_綾音Aya\
    └── 2026_05_27\
        ├── 12_03_48_xxx.mp4
        ├── 13_33_49_xxx.mp4
        └── .dd_song_miner_done.json
```

## Windows 定时运行

最稳的做法是写一个 PowerShell 脚本，再用 Windows 任务计划每天调用。

### 1. 创建脚本

例如保存为 `D:\song-miner-jobs\daily-run.ps1`：

```powershell
$ErrorActionPreference = "Stop"

$Repo = "C:\Users\bakaz\Documents\dd-song-miner-llm"
$InputRoot = "\\server\share\ddtv2"
$WorkRoot = "D:\song-work"
$ResultRoot = "D:\song-results"

Set-Location $Repo
& "$Repo\.venv\Scripts\python.exe" -m dd_song_miner_llm batch-run $InputRoot `
  --config "$Repo\config.yaml" `
  --work-root $WorkRoot `
  --result-root $ResultRoot `
  --marker ".dd_song_miner_done.json" `
  --extensions "mp4,mkv,flv"
```

### 2. 手动试跑脚本

```powershell
powershell.exe -ExecutionPolicy Bypass -File "D:\song-miner-jobs\daily-run.ps1"
```

确认能跑通后再加定时任务。

### 3. 创建每天凌晨运行的任务

```powershell
schtasks /Create /TN "dd-song-miner-daily" /SC DAILY /ST 03:30 /F /TR "powershell.exe -NoProfile -ExecutionPolicy Bypass -File D:\song-miner-jobs\daily-run.ps1"
```

查看任务：

```powershell
schtasks /Query /TN "dd-song-miner-daily" /V /FO LIST
```

手动触发一次：

```powershell
schtasks /Run /TN "dd-song-miner-daily"
```

删除任务：

```powershell
schtasks /Delete /TN "dd-song-miner-daily" /F
```

网络路径注意事项：

- 任务计划运行账号必须能访问 `\\server\share`。
- 如果网络盘是映射盘符，比如 `Z:\`，任务计划里可能不可见；优先使用 UNC 路径。
- 建议 `--work-root` 放本地磁盘，避免 ASR/FFmpeg 长时间占用网络盘。
- `--result-root` 可以是本地盘或网络盘，但网络盘写入失败会导致该视频处理失败，源目录不会写完成标记。

## Windows 路径兼容

Windows 下中文、日文、空格路径或 UNC 网络路径可能在 Python/FFmpeg 之间发生编码问题。程序会自动判断输入路径：

- ASCII 本地路径：直接使用原文件。
- 非 ASCII 路径或 `\\server\share` 网络路径：在运行目录 `00_input` 下创建 ASCII 文件名 staging 输入。
- staging 优先创建硬链接；跨盘、网络盘或权限不允许硬链接时复制文件。

如果旧版终端把命令行参数本身传乱码，建议使用 PowerShell 7 / Windows Terminal，或用 `batch-run` 扫描目录，让程序从文件系统读取路径。

## FFmpeg 编码策略

视频导出配置：

```yaml
output:
  video_codec: auto
```

`auto` 会按以下顺序尝试：

1. NVIDIA NVENC: `h264_nvenc`
2. Intel Quick Sync: `h264_qsv`
3. AMD AMF: `h264_amf`
4. CPU: `libx264`

显式指定：

```powershell
--video-codec nv      # h264_nvenc
--video-codec intel   # h264_qsv
--video-codec amd     # h264_amf
--video-codec cpu     # libx264
--video-codec copy    # 不重编码，直接复制原视频流
```

如果你只想最快切片且不要求重新编码，使用：

```powershell
--video-codec copy
```

音频默认 `m4a`，会复制源 AAC 音轨，不降低音质。导出 `mp3` 等需要转码的格式时，使用 `audio_bitrate_kbps` 控制码率：

```yaml
output:
  audio_extension: mp3
  audio_bitrate_kbps: 320
```

## Match 上下文

每次 LLM 识别后，`02_asr\llm` 下会生成：

- `matches.json`：LLM 返回的原始歌曲 match。
- `match_context.json`：每个 match 的 `segment_indices`、起止秒数、命中的 Whisper 段落，以及前后上下文。
- `match_context.csv`：表格版上下文，适合人工筛查、对照歌词、继续微调 prompt。

`match_context.csv` 里每行包含：

- `match_index`
- `title`
- `artist`
- `confidence`
- `match_start`
- `match_end`
- `segment_index`
- `start`
- `end`
- `start_timecode`
- `end_timecode`
- `is_match`
- `text`

上下文条数由配置控制：

```yaml
output:
  match_context_segments: 10
```

## 手动重切

先复制或直接编辑某次运行的 `04_reports\songs.csv`，调整 `start` / `end`。

最小 CSV 示例：

```csv
index,start,end,title,artist
1,00:23:02,00:28:31,一半一半,洛天依
```

时间格式支持：

- `HH:MM:SS`
- `MM:SS`
- 秒数，例如 `1382.14`

重新切片：

```powershell
python -m dd_song_miner_llm manual-cut "D:\song-results\某次运行" --config config.yaml
```

指定手改 CSV 和输出目录：

```powershell
python -m dd_song_miner_llm manual-cut "D:\song-results\某次运行" `
  --config config.yaml `
  --csv "D:\song-results\某次运行\04_reports\songs.manual.csv" `
  --out "D:\song-results\某次运行\05_manual_v2"
```

`manual-cut` 默认使用 run 目录里的 `manifest.json` 找原始输入视频。如果 manifest 里是 staging 路径或文件已移动，可以手动指定：

```powershell
python -m dd_song_miner_llm manual-cut "D:\song-results\某次运行" --config config.yaml --video "D:\videos\source.mp4"
```

## 输出结构

```text
runs\xxx\
├── 00_input\
│   └── input_xxx.mp4
├── 01_audio\
│   └── source.wav
├── 02_asr\
│   ├── transcript.json
│   └── llm\
│       ├── llm_batch_000000.json
│       ├── matches.json
│       ├── match_context.csv
│       └── match_context.json
├── 03_clips\
│   ├── audio\
│   └── video\
├── 04_reports\
│   ├── songs.csv
│   └── songs.json
├── 05_manual\
│   ├── audio\
│   ├── video\
│   └── reports\
└── manifest.json
```

关键文件：

| 文件 | 说明 |
| --- | --- |
| `01_audio\source.wav` | Whisper 使用的中间音频。 |
| `02_asr\transcript.json` | ASR 原始分段结果。 |
| `02_asr\llm\llm_batch_*.json` | 每批 LLM 原始输出和调试信息。 |
| `02_asr\llm\matches.json` | LLM 识别出的歌曲 match。 |
| `02_asr\llm\match_context.csv/json` | 每个 match 的前后 ASR 上下文。 |
| `03_clips\audio` | 自动切出的音频片段。 |
| `03_clips\video` | 自动切出的视频片段。 |
| `04_reports\songs.csv/json` | 自动识别报告。 |
| `05_manual` | 手动重切输出。 |

## 常见问题

### `Binary not found: ffprobe`

安装系统 FFmpeg，并确认 `ffprobe -version` 可用。没有 ffprobe 时程序会尝试用 `ffmpeg -i` 解析时长，但推荐安装完整 FFmpeg。

### `cublas64_12.dll is not found`

安装 CUDA 12 运行时依赖：

```powershell
pip install -r requirements-cu12.txt
```

如果仍失败，程序会自动回退 CPU int8，只是速度会慢。

### FFmpeg 找不到中文/日文路径

优先使用 PowerShell 7 / Windows Terminal。程序会自动 staging 非 ASCII 路径，但如果参数在进入 Python 前已经乱码，建议改用 `batch-run` 扫描目录。

### LLM 返回 0 首

先看：

```text
02_asr\llm\llm_batch_000000.json
02_asr\llm\matches.json
02_asr\llm\match_context.csv
```

如果 `raw_response` 为空或有 `Connection error`，优先检查 API key、网络和 `base_url`。如果有 matches 但最终报告少，检查 `min_song_seconds` 和 `merge_gap_seconds`。

### `songs.csv` 被 Excel 占用导致 PermissionError

程序会尝试写带时间戳的备用报告文件。建议运行时关闭 Excel 中打开的 `songs.csv`。

## License

MIT
