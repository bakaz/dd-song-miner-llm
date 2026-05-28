from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .models import SongMatch, TranscriptSegment
from .search_tools import get_tools, execute_tool


@dataclass
class LLMProvider:
    api_key: str
    base_url: str | None = None
    model: str = "gpt-4o"
    temperature: float = 0.3
    max_tokens: int = 4096
    max_completion_tokens: int | None = None


def _build_prompt(segments: list[TranscriptSegment], batch_start: int) -> str:
    lines = []
    for i, seg in enumerate(segments):
        idx = batch_start + i
        lines.append(f"[{idx}] ({seg.start:.1f}s-{seg.end:.1f}s) {seg.text}")

    transcript_text = "\n".join(lines)

    return f"""你是一个面向演唱会、直播和长视频的歌曲识别专家。
下面是一整段视频的 ASR 转写片段，每行格式为 [序号] (开始秒-结束秒) 文本。

任务：从完整上下文中识别所有演唱片段，返回纯 JSON 数组。
每个对象必须包含以下字段：
- title: 歌名。能识别出原曲时填写准确歌名；无法确认时填写"未知歌曲："加最有代表性的一句歌词。
- artist: 原唱或演唱者。无法判断时填空字符串。
- segment_indices: 属于同一首歌的 ASR 段落序号数组，必须只使用输入中出现的序号，按升序排列。
- confidence: 0 到 1 的置信度。

识别原则：
1. 只要是**在唱歌**的段落都应识别出来，即使无法确定歌名。
2. 同一首歌的连续演唱段落必须合并成一个对象。不要把主歌、副歌、桥段拆成多首。
3. 明显的说话、聊天、感谢、报幕、互动、口播不要放进 segment_indices。
4. ASR 可能没有标点、可能错字。根据重复歌词、押韵、节奏化短句判断是否在唱歌。
5. 不要因为短于 2 分钟就丢弃。只要是在唱歌就标出来。

可以使用 search_lyrics 工具搜索歌词确认歌名，最多搜索2次，然后必须返回结果。
宁可返回"未知歌曲"也不要漏掉任何演唱片段。

输出要求：
- 只返回 JSON 数组，不要 Markdown，不要解释，不要代码块。
- 不要输出输入中不存在的 segment index。
- 示例：[{{"title": "歌曲名", "artist": "歌手名", "segment_indices": [12, 13, 14], "confidence": 0.86}}]

完整 ASR 转写片段：
{transcript_text}"""


def _iter_llm_batches(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
) -> list[tuple[int, list[TranscriptSegment]]]:
    batch_size = config["llm"].get("batch_size")
    if batch_size in (None, "", 0, "0"):
        return [(0, segments)]

    batch_size = int(batch_size)
    if batch_size <= 0:
        return [(0, segments)]

    return [
        (batch_start, segments[batch_start:batch_start + batch_size])
        for batch_start in range(0, len(segments), batch_size)
    ]


def _parse_llm_response(text: str) -> list[dict[str, Any]]:
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        text = "\n".join(lines[1:-1] if lines[-1].strip() == "```" else lines[1:])
    try:
        result = json.loads(text)
        if isinstance(result, list):
            return [item for item in result if isinstance(item, dict)]
    except json.JSONDecodeError:
        start = text.find("[")
        end = text.rfind("]")
        if start != -1 and end > start:
            try:
                result = json.loads(text[start:end + 1])
                if isinstance(result, list):
                    return [item for item in result if isinstance(item, dict)]
            except json.JSONDecodeError:
                pass
    return []


def _parse_segment_indices(value: Any) -> list[int]:
    if not isinstance(value, list):
        return []

    indices: list[int] = []
    seen: set[int] = set()
    for item in value:
        if isinstance(item, bool):
            continue
        try:
            idx = int(item)
        except (TypeError, ValueError):
            continue
        if idx not in seen:
            indices.append(idx)
            seen.add(idx)
    return indices


def _parse_confidence(value: Any) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        return 0.5
    return max(0.0, min(1.0, confidence))


def _parse_providers(config: dict[str, Any]) -> list[LLMProvider]:
    llm_config = config["llm"]
    providers: list[LLMProvider] = []

    api_keys = llm_config.get("api_key", "")
    api_key_env = llm_config.get("api_key_env")
    if not api_keys and api_key_env:
        api_keys = os.environ.get(str(api_key_env), "")
    base_urls = llm_config.get("base_url", "")
    models = llm_config.get("model", "gpt-4o")
    temperature = float(llm_config.get("temperature", 0.3))
    max_tokens = int(llm_config.get("max_tokens", 4096))
    max_completion_tokens_value = llm_config.get("max_completion_tokens")
    max_completion_tokens = (
        int(max_completion_tokens_value)
        if max_completion_tokens_value not in (None, "")
        else None
    )

    if api_keys is None:
        api_keys = []
    elif isinstance(api_keys, str):
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
            max_completion_tokens=max_completion_tokens,
        ))

    fallbacks = llm_config.get("fallbacks", [])
    for fb in fallbacks:
        fallback_api_key = fb.get("api_key", "")
        fallback_api_key_env = fb.get("api_key_env")
        if not fallback_api_key and fallback_api_key_env:
            fallback_api_key = os.environ.get(str(fallback_api_key_env), "")
        providers.append(LLMProvider(
            api_key=fallback_api_key,
            base_url=fb.get("base_url"),
            model=str(fb.get("model", "gpt-4o")),
            temperature=float(fb.get("temperature", temperature)),
            max_tokens=int(fb.get("max_tokens", max_tokens)),
            max_completion_tokens=(
                int(fb["max_completion_tokens"])
                if fb.get("max_completion_tokens") not in (None, "")
                else max_completion_tokens
            ),
        ))

    return providers


def _call_llm(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    max_tokens_override: int | None = None,
    tools: list[dict[str, Any]] | None = None,
) -> Any:
    token_args = (
        {"max_completion_tokens": max_tokens_override}
        if max_tokens_override is not None
        else (
            {"max_completion_tokens": provider.max_completion_tokens}
            if provider.max_completion_tokens is not None
            else {"max_tokens": provider.max_tokens}
        )
    )

    kwargs: dict[str, Any] = {
        "model": provider.model,
        "messages": messages,
        "temperature": provider.temperature,
        **token_args,
    }

    if tools:
        kwargs["tools"] = tools

    response = client.chat.completions.create(**kwargs)
    return response


def _llm_response_debug(response: Any) -> dict[str, Any]:
    choice = response.choices[0] if response.choices else None
    message = choice.message if choice is not None else None
    message_data = message.model_dump() if message is not None else {}
    usage = response.usage.model_dump() if getattr(response, "usage", None) else None
    content = message_data.get("content") or ""
    reasoning = message_data.get("reasoning_content") or ""
    return {
        "model": getattr(response, "model", None),
        "finish_reason": getattr(choice, "finish_reason", None) if choice is not None else None,
        "content": content,
        "content_length": len(content),
        "reasoning_content": reasoning,
        "reasoning_content_length": len(reasoning),
        "message_keys": list(message_data.keys()),
        "usage": usage,
        "tool_calls": message_data.get("tool_calls"),
    }


def _run_llm_with_tools(
    client: Any,
    provider: LLMProvider,
    messages: list[dict[str, Any]],
    tools: list[dict[str, Any]],
    batch_debug: dict[str, Any],
    max_tool_rounds: int = 2,
) -> str:
    """调用LLM，处理tool calls，最多max_tool_rounds轮后强制返回JSON。"""
    for tool_round in range(max_tool_rounds + 1):
        is_last = (tool_round == max_tool_rounds)

        # 最后一轮不传tools，强制返回JSON，使用更大的max_tokens
        call_tools = None if is_last else tools
        last_round_tokens = 8192 if is_last else None
        if is_last and tool_round > 0:
            messages = messages + [{
                "role": "user",
                "content": "搜索已完成。现在请根据已有的搜索结果，直接返回歌曲识别的JSON数组。不要再调用任何工具。只返回JSON数组，不要其他文字。",
            }]

        response = _call_llm(client, provider, messages, max_tokens_override=last_round_tokens, tools=call_tools)
        debug = _llm_response_debug(response)
        batch_debug.setdefault("tool_rounds", []).append({
            "round": tool_round + 1,
            "content": debug["content"][:200],
            "reasoning_content": debug["reasoning_content"][:200],
            "finish_reason": debug["finish_reason"],
            "has_tool_calls": bool(debug.get("tool_calls")),
        })

        content = debug["content"]
        tool_calls_data = debug.get("tool_calls")

        # 没有tool calls -> 直接返回content
        if not tool_calls_data:
            # content为空但reasoning有内容时，提取reasoning中的JSON
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            return content

        # 最后一轮强制返回，即使还有tool calls
        if is_last:
            if not content.strip() and debug["reasoning_content"].strip():
                content = debug["reasoning_content"]
            return content

        # 执行tool calls
        choice = response.choices[0] if response.choices else None
        message = choice.message if choice is not None else None
        if not message or not message.tool_calls:
            return content

        messages.append(message.model_dump())
        for tc in message.tool_calls:
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {}
            result = execute_tool(tc.function.name, args)
            batch_debug.setdefault("tool_calls_log", []).append({
                "round": tool_round + 1,
                "function": tc.function.name,
                "arguments": args,
                "result_preview": result[:200],
            })
            messages.append({
                "role": "tool",
                "tool_call_id": tc.id,
                "content": result,
            })

    return ""


def _build_reasoning_followup_prompt(reasoning_content: str, partial_content: str = "") -> str:
    partial_block = (
        f"\n\n上一轮已经生成但可能被截断或格式不完整的内容：\n{partial_content}"
        if partial_content.strip()
        else ""
    )
    return f"""下面是上一轮模型对 ASR 歌曲识别任务的分析内容。它可能是不完整的，但里面已经包含了歌曲边界判断。

不要继续分析，不要解释，不要输出思考过程。请只把分析中已经确定的歌曲整理成 JSON 数组。

每个对象只允许包含以下字段：
- title: 歌名。无法确定时使用"未知歌曲："加代表性歌词关键词。
- artist: 歌手。无法判断时填空字符串。
- segment_indices: 属于该歌曲的 ASR 段落序号数组，按升序排列。
- confidence: 0 到 1 的置信度。

输出必须是纯 JSON 数组，不要 Markdown，不要代码块，不要额外文字。

上一轮分析内容：
{reasoning_content}{partial_block}"""


def _reasoning_followup_settings(config: dict[str, Any]) -> tuple[bool, int, int | None]:
    llm_config = config["llm"]
    enabled = bool(llm_config.get("retry_empty_with_reasoning", True))
    rounds = int(llm_config.get("reasoning_followup_rounds", 2))
    tokens_value = llm_config.get("reasoning_followup_max_tokens", 8192)
    tokens = int(tokens_value) if tokens_value not in (None, "") else None
    return enabled, max(0, rounds), tokens


def _run_reasoning_followups(
    client: Any,
    provider: LLMProvider,
    config: dict[str, Any],
    reasoning_content: str,
    partial_content: str,
    batch_debug: dict[str, Any],
) -> str:
    retry_reasoning, followup_rounds, followup_tokens = _reasoning_followup_settings(config)
    if not retry_reasoning:
        return ""

    content = ""
    material = reasoning_content
    partial = partial_content
    for _ in range(followup_rounds):
        if not material.strip() and not partial.strip():
            break

        followup_prompt = _build_reasoning_followup_prompt(material, partial)
        try:
            followup_response = _call_llm(
                client,
                provider,
                [{"role": "user", "content": followup_prompt}],
                max_tokens_override=followup_tokens,
            )
            followup_debug = _llm_response_debug(followup_response)
            content = followup_debug["content"]
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "content": content[:500],
                "reasoning_content": followup_debug["reasoning_content"][:500],
            })
            batch_debug["raw_response"] = content
        except Exception as exc:
            batch_debug["reasoning_followups"].append({
                "round": len(batch_debug["reasoning_followups"]) + 1,
                "error": str(exc),
            })
            return ""

        if content.strip() and _parse_llm_response(content):
            return content

        material = str(followup_debug.get("reasoning_content") or "")
        partial = content

    return content


def identify_songs(
    segments: list[TranscriptSegment],
    config: dict[str, Any],
    debug_dir: str | Path | None = None,
) -> list[SongMatch]:
    providers = _parse_providers(config)
    if not providers:
        raise RuntimeError("LLM API key not configured. Set llm.api_key in config.")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise RuntimeError("openai not installed. pip install openai") from exc

    all_matches: list[SongMatch] = []
    debug_path = Path(debug_dir) if debug_dir is not None else None
    if debug_path is not None:
        debug_path.mkdir(parents=True, exist_ok=True)

    tools = get_tools()
    use_tools = bool(config["llm"].get("use_tools", True))

    for batch_start, batch in _iter_llm_batches(segments, config):
        prompt = _build_prompt(batch, batch_start)
        batch_debug: dict[str, Any] = {
            "batch_start": batch_start,
            "batch_end": batch_start + len(batch) - 1,
            "segment_count": len(batch),
            "provider": None,
            "raw_response": None,
            "parsed_items": [],
            "tool_calls_log": [],
            "tool_rounds": [],
            "reasoning_followups": [],
            "error": None,
        }

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
                messages: list[dict[str, Any]] = [
                    {"role": "user", "content": prompt}
                ]

                batch_debug["provider"] = {
                    "base_url": provider.base_url or "openai",
                    "model": provider.model,
                }

                if use_tools:
                    content = _run_llm_with_tools(
                        client, provider, messages, tools, batch_debug
                    )
                else:
                    response = _call_llm(client, provider, messages)
                    debug = _llm_response_debug(response)
                    content = debug["content"]
                    if not content.strip() and debug["reasoning_content"].strip():
                        content = debug["reasoning_content"]

                batch_debug["raw_response"] = content
                break
            except Exception as exc:
                last_error = exc
                print(f"LLM API error ({provider.base_url or 'openai'}): {exc}")
                continue

        if content is None:
            batch_debug["error"] = str(last_error)
            if debug_path is not None:
                _write_llm_debug(debug_path, batch_start, batch_debug)
            print(f"All LLM providers failed for batch {batch_start}. Last error: {last_error}")
            continue

        # content为空时尝试reasoning followup
        if not content.strip():
            reasoning_content = ""
            if batch_debug.get("tool_rounds"):
                for tr in batch_debug["tool_rounds"]:
                    if tr.get("reasoning_content"):
                        reasoning_content = tr["reasoning_content"]
                        break
            content = _run_reasoning_followups(
                client, provider, config, reasoning_content, "", batch_debug
            )
            if not content.strip():
                batch_debug["error"] = "LLM returned empty content"
                if debug_path is not None:
                    _write_llm_debug(debug_path, batch_start, batch_debug)
                print(f"LLM returned empty response for batch {batch_start}.")
                continue

        # 尝试解析JSON
        items = _parse_llm_response(content)
        if not items:
            # 从reasoning中提取
            if batch_debug.get("tool_rounds"):
                for tr in batch_debug["tool_rounds"]:
                    rc = tr.get("reasoning_content", "")
                    if rc.strip():
                        items = _parse_llm_response(rc)
                        if items:
                            break
            # reasoning followup兜底
            if not items:
                response_debug = batch_debug.get("response") or {}
                reasoning_content = str(response_debug.get("reasoning_content") or "")
                if reasoning_content.strip():
                    followup_content = _run_reasoning_followups(
                        client, provider, config, reasoning_content, content, batch_debug
                    )
                    if followup_content.strip():
                        content = followup_content
                        items = _parse_llm_response(content)

        batch_debug["parsed_items"] = items
        if debug_path is not None:
            _write_llm_debug(debug_path, batch_start, batch_debug)

        for item in items:
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            all_matches.append(SongMatch(
                title=title,
                artist=str(item.get("artist", "")),
                lyrics_snippet="",
                segment_indices=_parse_segment_indices(item.get("segment_indices", [])),
                confidence=_parse_confidence(item.get("confidence", 0.5)),
            ))

    return all_matches


def _write_llm_debug(debug_dir: Path, batch_start: int, payload: dict[str, Any]) -> None:
    target = debug_dir / f"llm_batch_{batch_start:06d}.json"
    target.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
