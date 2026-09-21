#!/usr/bin/env python3
"""Interactive terminal chat for the server-side OpenAI-compatible vLLM model."""

from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from typing import Any, Dict, List, Mapping


def _request(
    url: str,
    model: str,
    messages: List[Mapping[str, str]],
    max_tokens: int,
    temperature: float,
    thinking: bool,
    stream: bool,
) -> str:
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "stream": stream,
        "chat_template_kwargs": {"enable_thinking": thinking},
    }
    request = urllib.request.Request(
        url,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Accept": "text/event-stream, application/json", "Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=600) as response:
            if not stream:
                raw = json.loads(response.read().decode("utf-8"))
                return str(raw["choices"][0]["message"].get("content") or "")
            chunks: List[str] = []
            for line in response:
                decoded = line.decode("utf-8").strip()
                if not decoded.startswith("data:"):
                    continue
                data = decoded[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    raw = json.loads(data)
                except json.JSONDecodeError:
                    continue
                choices = raw.get("choices") or []
                delta = choices[0].get("delta") if choices and isinstance(choices[0], Mapping) else None
                text = delta.get("content") if isinstance(delta, Mapping) else None
                if text:
                    text = str(text)
                    chunks.append(text)
                    sys.stdout.write(text)
                    sys.stdout.flush()
            if chunks:
                sys.stdout.write("\n")
            return "".join(chunks)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError("LLM HTTP %s: %s" % (exc.code, detail[:2000])) from exc
    except (urllib.error.URLError, TimeoutError, OSError) as exc:
        raise RuntimeError("cannot reach LLM at %s: %s" % (url, exc)) from exc


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--url",
        default=os.environ.get("CONTROLLER_LLM_URL", "http://127.0.0.1:8000/v1/chat/completions"),
    )
    parser.add_argument("--model", default=os.environ.get("CONTROLLER_LLM_MODEL", "qwen3.5-controller"))
    parser.add_argument("--system", default="你是一个有帮助、简洁的中文助手。")
    parser.add_argument("--max-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--thinking", action="store_true", help="保留模型思考输出；默认关闭以降低延迟")
    parser.add_argument("--no-stream", action="store_true", help="等待完整回答后一次性输出")
    args = parser.parse_args()
    if args.max_tokens <= 0 or not 0 <= args.temperature <= 2:
        raise SystemExit("--max-tokens must be positive and --temperature must be in [0, 2]")

    messages: List[Dict[str, str]] = [{"role": "system", "content": args.system}]
    print("Connected to %s (%s). 输入 /clear 清空上下文，/quit 退出。" % (args.model, args.url))
    while True:
        try:
            prompt = input("你> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if not prompt:
            continue
        if prompt in {"/quit", "/exit", "quit", "exit"}:
            return 0
        if prompt == "/clear":
            messages = [{"role": "system", "content": args.system}]
            print("上下文已清空。")
            continue
        if prompt.startswith("/system "):
            args.system = prompt[len("/system ") :].strip() or args.system
            messages[0] = {"role": "system", "content": args.system}
            print("system prompt 已更新。")
            continue
        messages.append({"role": "user", "content": prompt})
        try:
            print("LLM> ", end="", flush=True)
            answer = _request(
                args.url,
                args.model,
                messages,
                args.max_tokens,
                args.temperature,
                args.thinking,
                not args.no_stream,
            )
            if args.no_stream:
                sys.stdout.write(answer + "\n")
                sys.stdout.flush()
            if not answer:
                print("[空回答]")
            messages.append({"role": "assistant", "content": answer})
        except RuntimeError as exc:
            messages.pop()
            print("\n[错误] %s" % exc, file=sys.stderr)


if __name__ == "__main__":
    raise SystemExit(main())
