from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .models import SongMatch, TranscriptSegment


@dataclass
class LLMProvider:
    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096


def _build_prompt(segments: list[TranscriptSegment], batch_start: int) -> str:
    lines = []
    for i, seg in enumerate(segments):
        idx = batch_start + i
        lines.append(f"[{idx}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}")

    transcript_text = "\n".join(lines)

    return f"""你是一个音乐识别专家。下面是一段视频的 ASR 转写结果，每行格式为 [序号] (时间段) 文本。

请识别其中的**完整歌曲**，返回 JSON 数组。每个歌曲对象包含：
- title: 歌名（尽量准确）
- artist: 歌手（如果能识别）
- segment_indices: 属于这首歌的所有连续 ASR 段落序号列表
- confidence: 置信度 0-1

重要规则：
1. 一首完整的歌曲通常持续 2-5 分钟，包含多段歌词和副歌
2. 将**连续的唱歌段落**合并为一首歌，不要把同一首歌拆成多段
3. 只识别唱歌的部分，忽略说话、串场、聊天、感谢、告别
4. 如果无法确定歌名，用歌词前几个字作为 title
5. 如果完全无法识别，返回空数组 []
6. 返回纯 JSON，不要其他文字

判断唱歌的特征：
- 歌词押韵、有节奏感
- 连续多段相似的旋律结构（主歌-副歌）
- 不是日常对话或互动聊天

ASR 转写结果：
{transcript_text}"""


def _parse_llm_response(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except json.JSONDecodeError:
        pass
    return []


def _parse_providers(config: dict[str, Any]) -> list[LLMProvider]:
    llm_config = config["llm"]
    providers: list[LLMProvider] = []

    api_keys = llm_config.get("api_key", "")
    base_urls = llm_config.get("base_url", "")
    models = llm_config.get("model", "gpt-4o")
    temperature = float(llm_config.get("temperature", 0.3))
    max_tokens = int(llm_config.get("max_tokens", 4096))

    if isinstance(api_keys, str):
        api_keys = [k.strip() for k in api_keys.split(",") if k.strip()]
    if isinstance(base_urls, str):
        base_urls = [u.strip() for u in base_urls.split(",") if u.strip()] if base_urls else []
    if isinstance(models, str):
        models = [m.strip() for m in models.split(",") if m.strip()]

    for i, api_key in enumerate(api_keys):
        base_url = base_urls[i] if i < len(base_urls) else (base_urls[0] if base_urls else None)
        model = models[i] if i < len(models) else (models[0] if models else "gpt-4o")
        providers.append(LLMProvider(
            api_key=api_key,
            base_url=base_url,
            model=model,
            temperature=temperature,
            max_tokens=max_tokens,
        ))

    fallbacks = llm_config.get("fallbacks", [])
    for fb in fallbacks:
        providers.append(LLMProvider(
            api_key=fb.get("api_key", ""),
            base_url=fb.get("base_url"),
            model=str(fb.get("model", "gpt-4o")),
            temperature=float(fb.get("temperature", temperature)),
            max_tokens=int(fb.get("max_tokens", max_tokens)),
        ))

    return providers


def _call_llm(
    client: Any,
    provider: LLMProvider,
    prompt: str,
) -> str:
    response = client.chat.completions.create(
        model=provider.model,
        messages=[{"role": "user", "content": prompt}],
        temperature=provider.temperature,
        max_tokens=provider.max_tokens,
    )
    return response.choices[0].message.content or ""


def identify_songs(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> list[SongMatch]:
    providers = _parse_providers(config)
    if not providers:
        raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai not installed. pip install openai") from exc

    batch_size = int(config["llm"].get("batch_size", 20))
    all_matches: list[SongMatch] = []

    for batch_start in range(0, len(segments), batch_size):
        batch = segments[batch_start:batch_start + batch_size]
        prompt = _build_prompt(batch, batch_start)

        content = None
        last_error = None

        for provider in providers:
            if not provider.api_key:
                continue

            try:
                client_kwargs: dict[str, Any] = {"api_key": provider.api_key}
                if provider.base_url:
                    client_kwargs["base_url"] = provider.base_url

                client = OpenAI(**client_kwargs)
                content = _call_llm(client, provider, prompt)
                break
            except Exception as exc:
                last_error = exc
                print(f"LLM API error ({provider.base_url or 'openai'}): {exc}")
                continue

        if content is None:
            print(f"All LLM providers failed for batch {batch_start}. Last error: {last_error}")
            continue

        items = _parse_llm_response(content)
        for item in items:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            all_matches.append(SongMatch(
                title=title,
                artist=str(item.get("artist", "")),
                lyrics_snippet="",
                segment_indices=list(item.get("segment_indices", [])),
                confidence=float(item.get("confidence", 0.5)),
            ))

    return all_matches
