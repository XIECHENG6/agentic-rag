"""LLM client with OpenAI-compatible API + 4-layer fallback JSON parser."""

import json
import os
import re

import yaml
from openai import OpenAI


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
        self.client = OpenAI(
            base_url=api_base or os.getenv(
                "OPENAI_API_BASE",
                config.get("api_base", "https://api.deepseek.com/v1"),
            ),
            api_key=api_key or os.getenv("OPENAI_API_KEY", ""),
        )
        self.model = model or os.getenv(
            "LLM_MODEL", config.get("model", "deepseek-chat")
        )
        self.temperature = config.get("temperature", 0.0)
        self.max_tokens = config.get("max_tokens", 2048)

    def generate(self, messages, temperature=None, max_tokens=None):
        """Send a list of messages and return the assistant reply."""
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature if temperature is not None else self.temperature,
            max_tokens=max_tokens if max_tokens is not None else self.max_tokens,
        )
        return response.choices[0].message.content

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
