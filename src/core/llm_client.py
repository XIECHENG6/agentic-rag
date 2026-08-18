"""LLM client with OpenAI-compatible API + 4-layer fallback JSON parser."""

import json
import os
import re
import time

import yaml
from openai import OpenAI


class LLMCallBudgetExceeded(RuntimeError):
    """Raised before an API call when the current request budget is exhausted."""


def load_config(path=None):
    if path is None:
        path = os.path.join(os.path.dirname(__file__), "..", "..", "config", "settings.yaml")
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            return yaml.safe_load(f)
    return {}


class LLMClient:
    """OpenAI-compatible LLM client (works with DeepSeek, OpenAI, etc.)."""

    def __init__(self, api_base=None, api_key=None, model=None):
        config = load_config().get("llm", {})
        timeout = float(os.getenv(
            "LLM_TIMEOUT_SECONDS", config.get("timeout_seconds", 60)
        ))
        max_retries = int(os.getenv(
            "LLM_MAX_RETRIES", config.get("max_retries", 2)
        ))
        self.client = OpenAI(
            base_url=api_base or os.getenv(
                "OPENAI_API_BASE",
                config.get("api_base", "https://api.deepseek.com/v1"),
            ),
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
            timeout=timeout,
            max_retries=max_retries,
        )
        self.model = model or os.getenv(
            "LLM_MODEL", config.get("model", "deepseek-chat")
        )
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2048)
        self._usage = {
            "calls": 0,
            "errors": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "latency_seconds": 0.0,
        }
        self._call_budget = None

    def set_call_budget(self, max_total_calls):
        """Set an absolute cumulative call limit for the current operation."""
        if max_total_calls is None:
            self._call_budget = None
            return
        if max_total_calls < 0:
            raise ValueError("max_total_calls must be non-negative")
        self._call_budget = int(max_total_calls)

    def clear_call_budget(self):
        self._call_budget = None

    @staticmethod
    def _usage_value(usage, name):
        if usage is None:
            return 0
        if isinstance(usage, dict):
            return int(usage.get(name, 0) or 0)
        return int(getattr(usage, name, 0) or 0)

    def snapshot_usage(self):
        return dict(self._usage)

    def usage_delta(self, snapshot):
        current = self._usage
        return {key: current[key] - snapshot.get(key, 0) for key in current}

    def get_usage_stats(self, input_cost_per_1m=0.0, output_cost_per_1m=0.0):
        stats = dict(self._usage)
        stats["estimated_cost_usd"] = (
            stats["prompt_tokens"] * input_cost_per_1m / 1_000_000
            + stats["completion_tokens"] * output_cost_per_1m / 1_000_000
        )
        return stats

    def generate(self, messages, temperature=None, max_tokens=None):
        """Send a list of messages and return the assistant reply."""
        if self._call_budget is not None and self._usage["calls"] >= self._call_budget:
            raise LLMCallBudgetExceeded(
                f"LLM call budget exhausted ({self._call_budget} calls)"
            )
        started = time.perf_counter()
        self._usage["calls"] += 1
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=temperature if temperature is not None else self.temperature,
                max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
            )
            usage = getattr(response, "usage", None)
            self._usage["prompt_tokens"] += self._usage_value(usage, "prompt_tokens")
            self._usage["completion_tokens"] += self._usage_value(usage, "completion_tokens")
            self._usage["total_tokens"] += self._usage_value(usage, "total_tokens")
            if not getattr(response, "choices", None):
                raise RuntimeError("LLM response contained no choices")
            content = response.choices[0].message.content
            if not isinstance(content, str) or not content.strip():
                raise RuntimeError("LLM response contained an empty message")
            return content
        except Exception:
            self._usage["errors"] += 1
            raise
        finally:
            self._usage["latency_seconds"] += time.perf_counter() - started

    def generate_text(self, prompt, system=None, **kwargs):
        """Convenience: single-turn generation with optional system prompt."""
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        return self.generate(messages, **kwargs)

    @staticmethod
    def extract_json(text):
        """4-layer fallback JSON parser for LLM outputs.

        Layer 1: ```json ... ``` fenced block
        Layer 2: ``` ... ``` fenced block
        Layer 3: raw JSON object/array (balanced bracket matching)
        Layer 4: regex extraction for simple flat objects
        """
        # Layer 1: ```json ... ```
        m = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Layer 2: ``` ... ```
        m = re.search(r"```\s*(.*?)\s*```", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(1))
            except json.JSONDecodeError:
                pass

        # Layer 3: balanced bracket matching (string-aware)
        for start_char, end_char in [("{", "}"), ("[", "]")]:
            start = text.find(start_char)
            if start == -1:
                continue
            depth = 0
            in_string = False
            escape_next = False
            for i in range(start, len(text)):
                ch = text[i]
                if escape_next:
                    escape_next = False
                    continue
                if ch == "\\":
                    escape_next = True
                    continue
                if ch == '"':
                    in_string = not in_string
                    continue
                if in_string:
                    continue
                if ch == start_char:
                    depth += 1
                elif ch == end_char:
                    depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start : i + 1])
                    except json.JSONDecodeError:
                        break

        # Layer 4: regex for simple flat objects
        objects = re.findall(r"\{[^{}]+\}", text)
        if objects:
            parsed = []
            for obj in objects:
                try:
                    parsed.append(json.loads(obj))
                except json.JSONDecodeError:
                    continue
            if parsed:
                return parsed

        return []
