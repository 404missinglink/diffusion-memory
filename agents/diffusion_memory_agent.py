import json
import os
from pathlib import Path
import sys
from datetime import datetime
from hashlib import blake2b
from typing import Any, Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from openai import OpenAI
from pydantic import AliasChoices, BaseModel, ConfigDict, Field

from lib.diffusion_memory import DiffusionMemory


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

    model_config = ConfigDict(extra="ignore")


class InputCase(BaseModel):
    case_id: str
    input: str = Field(validation_alias=AliasChoices("input", "question", "query"))
    history: list[Message] = Field(default_factory=list)
    choices: dict[str, str] = Field(default_factory=dict)
    haystack_sessions: list[Any] | None = None
    haystack_dates: list[Any] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    model_config = ConfigDict(extra="ignore")


def _parse_date(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y/%m/%d (%a) %H:%M", "%Y/%m/%d %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _flatten_history(case: InputCase) -> list[Message]:
    if case.history:
        return case.history

    sessions = case.haystack_sessions
    dates = case.haystack_dates
    if not sessions and case.metadata:
        sessions = case.metadata.get("haystack_sessions")
        dates = case.metadata.get("haystack_dates")

    if not isinstance(sessions, list):
        return []
    if not isinstance(dates, list):
        dates = []

    ordered_sessions: list[list[Any]]
    if dates and len(dates) == len(sessions):
        parsed = []
        parsed_all = True
        for idx, (date, session) in enumerate(zip(dates, sessions)):
            parsed_date = _parse_date(str(date) if date is not None else None)
            if parsed_date is None:
                parsed_all = False
            parsed.append((parsed_date, idx, session))
        if parsed_all:
            parsed.sort(key=lambda item: (item[0], item[1]))
        ordered_sessions = [session for _, _, session in parsed]
    else:
        ordered_sessions = sessions

    history: list[Message] = []
    for session in ordered_sessions:
        if not isinstance(session, list):
            continue
        for msg in session:
            if not isinstance(msg, dict):
                continue
            role = str(msg.get("role", "user")).strip().lower()
            content = msg.get("content")
            if content is None:
                continue
            if role not in {"system", "user", "assistant"}:
                role = "user"
            history.append(Message(role=role, content=str(content)))
    return history


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        return default


def _message_text(message: Message) -> str:
    return f"{message.role}: {message.content}"


def _stable_seed(value: str) -> int:
    digest = blake2b(value.encode("utf-8"), digest_size=4).digest()
    return int.from_bytes(digest, byteorder="big", signed=False)


def retrieve_messages(
    case: InputCase,
    *,
    max_messages: int,
    max_chars: int,
) -> list[Message]:
    history = _flatten_history(case)
    if not history:
        return []

    texts = [_message_text(message) for message in history]
    memory = DiffusionMemory(texts)
    if memory.n_tokens == 0:
        return history[-max_messages:]

    scored = memory.score_importance()
    params = memory.default_simulation_params()
    params["stimulus_steps"] = min(len(texts), params["stimulus_steps"])
    params["max_total_steps"] = max(params["stimulus_steps"] + 50, min(500, len(texts) * 4 + 50))
    diff = memory.simulate_graph_diffusion_memory(
        scored["importance"],
        stimulus_seed=_stable_seed(case.case_id),
        **params,
    )

    query_token_ids = set(memory._tokenize(case.input))
    query_indices = [memory.token_to_idx[token_id] for token_id in query_token_ids if token_id in memory.token_to_idx]
    final_strengths = diff["scores"]
    importance = scored["importance"]

    ranked: list[tuple[float, int, Message]] = []
    for idx, (message, doc_indices) in enumerate(zip(history, memory.indexed_docs)):
        if doc_indices.size == 0:
            ranked.append((0.0, idx, message))
            continue

        unique_indices = np.unique(doc_indices)
        doc_strength = float(final_strengths[unique_indices].mean())
        doc_importance = float(importance[unique_indices].mean())
        overlap = np.intersect1d(unique_indices, query_indices, assume_unique=False)
        query_match = float((final_strengths[overlap] + importance[overlap]).sum()) if overlap.size else 0.0
        recency = idx / max(1, len(history) - 1)
        score = query_match * 4.0 + doc_strength + doc_importance * 0.5 + recency * 0.05
        ranked.append((score, idx, message))

    selected: list[tuple[int, Message]] = []
    used_chars = 0
    for _, idx, message in sorted(ranked, key=lambda item: item[0], reverse=True):
        message_chars = len(message.content)
        if selected and used_chars + message_chars > max_chars:
            continue
        selected.append((idx, message))
        used_chars += message_chars
        if len(selected) >= max_messages or used_chars >= max_chars:
            break

    selected.sort(key=lambda item: item[0])
    return [message for _, message in selected]


def build_messages(case: InputCase) -> list[dict[str, str]]:
    context_messages = retrieve_messages(
        case,
        max_messages=_env_int("DIFFUSION_CONTEXT_MESSAGES", 24),
        max_chars=_env_int("DIFFUSION_CONTEXT_CHARS", 24000),
    )

    context_text = "\n".join(f"[{message.role}] {message.content}" for message in context_messages)
    user_prompt = "Relevant conversation history:\n"
    user_prompt += context_text if context_text else "(No relevant history retrieved.)"
    user_prompt += f"\n\nQuestion: {case.input}"
    if case.choices:
        choices_text = "\n".join(f"{key}. {value}" for key, value in sorted(case.choices.items()))
        user_prompt += f"\n\nChoices:\n{choices_text}\n\nRespond with only the choice letter (A, B, C, or D)."

    return [
        {
            "role": "system",
            "content": "You are a helpful assistant. Use the retrieved conversation history to answer accurately.",
        },
        {"role": "user", "content": user_prompt},
    ]


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY") or "missing"
    base_url = os.getenv("OPENAI_BASE_URL") or None
    model = os.getenv("OPENAI_MODEL") or "gpt-5-nano"
    client = OpenAI(api_key=api_key, base_url=base_url)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            data = json.loads(line)
            case = InputCase.model_validate(data)
        except Exception:
            continue

        try:
            response = client.chat.completions.create(
                model=model,
                messages=build_messages(case),
            )
            print(json.dumps({"output": response.choices[0].message.content, "error": None}))
        except Exception as exc:
            print(json.dumps({"output": None, "error": str(exc)}))

        try:
            sys.stdout.flush()
        except BrokenPipeError:
            return


if __name__ == "__main__":
    main()
