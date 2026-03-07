from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

from openai import OpenAI
from PIL import Image

from .utils import pil_to_base64

DEFAULT_REASONING_MODEL = "gpt-4.1"
DEFAULT_QA_MODEL = "gpt-4.1-mini"
VISION_SYSTEM_PROMPT = (
    "You are a concise radiology assistant for a research prototype. "
    "Describe only visible chest X-ray findings, acknowledge uncertainty, "
    "and do not provide treatment advice."
)
AGENT_QA_SYSTEM_PROMPT = (
    "You are a concise radiology assistant for a research prototype. "
    "Answer only from the provided context. "
    "If context is missing, say so clearly. "
    "Do not provide treatment plans or clinical advice."
)


def _parse_env_value(raw: str) -> str:
    value = raw.strip()
    if value.startswith(("'", '"')) and value.endswith(("'", '"')) and len(value) >= 2:
        value = value[1:-1]
    return value.strip()


def _load_key_from_env_file(path: Path) -> str:
    if not path.exists():
        return ""

    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[len("export ") :].strip()
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key.strip() != "OPENAI_API_KEY":
            continue
        parsed = _parse_env_value(value)
        if parsed:
            os.environ.setdefault("OPENAI_API_KEY", parsed)
            return parsed
    return ""


def resolve_openai_api_key(explicit_key: str | None = None) -> str:
    if explicit_key:
        return explicit_key

    env_key = os.getenv("OPENAI_API_KEY", "").strip()
    if env_key:
        return env_key

    candidates = [
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
    ]
    for path in candidates:
        key = _load_key_from_env_file(path)
        if key:
            return key
    return ""


def get_openai_client(
    api_key: str | None = None,
    base_url: str | None = None,
) -> OpenAI:
    key = resolve_openai_api_key(api_key)
    if not key:
        raise ValueError(
            "Missing OPENAI_API_KEY. Put it in .env, export it, or pass api_key explicitly."
        )

    kwargs: dict[str, Any] = {"api_key": key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def _extract_output_text(response: Any) -> str:
    text = getattr(response, "output_text", "")
    if isinstance(text, str) and text.strip():
        return text.strip()

    parts: list[str] = []
    for item in getattr(response, "output", []) or []:
        content = getattr(item, "content", None)
        if content is None and isinstance(item, dict):
            content = item.get("content")
        if not content:
            continue
        for chunk in content:
            if isinstance(chunk, dict):
                chunk_type = chunk.get("type")
                chunk_text = chunk.get("text")
            else:
                chunk_type = getattr(chunk, "type", None)
                chunk_text = getattr(chunk, "text", None)
            if chunk_type in {"output_text", "text"} and isinstance(chunk_text, str):
                parts.append(chunk_text)
    return "".join(parts).strip()


def build_reasoning_prompt(p_abnormal: float, tier: str) -> str:
    probability = float(p_abnormal)
    normalized_tier = tier.strip().lower()

    if normalized_tier == "borderline":
        return f"""
A binary chest X-ray classifier assigned this image an abnormal probability of {probability:.2f}, indicating a borderline abnormal signal.

- Carefully describe any subtle or equivocal imaging findings.
- Discuss alternative normal variants that could mimic abnormality.
- Emphasize uncertainty.
- Provide low-confidence differential considerations only if justified.

Avoid strong conclusions and clearly state limitations.
""".strip()

    if normalized_tier == "moderate":
        return f"""
A binary chest X-ray classifier assigned this image an abnormal probability of {probability:.2f}.

- Describe observable imaging features.
- Provide a short differential diagnosis list.
- Indicate level of certainty.
- Avoid definitive medical diagnosis.
- Emphasize that interpretation is preliminary and image-only.
""".strip()

    if normalized_tier == "high":
        return f"""
A binary chest X-ray classifier assigned this image an abnormal probability of {probability:.2f}.

Given the strong abnormal signal:

- Describe the dominant and most likely radiographic patterns visible.
- Explain which imaging features most strongly support abnormality.
- Provide 2-3 likely explanations.
- Indicate level of certainty.
- Avoid definitive diagnosis.

Focus on prominent findings rather than subtle ones.
""".strip()

    raise ValueError(f"Unsupported reasoning tier for image analysis: {tier!r}")


def analyze_xray_image(
    image: Image.Image,
    p_abnormal: float,
    tier: str,
    model: str = DEFAULT_REASONING_MODEL,
) -> str:
    client = get_openai_client()
    prompt = build_reasoning_prompt(p_abnormal, tier)
    encoded_image = pil_to_base64(image, fmt="JPEG", max_size=512)

    response = client.responses.create(
        model=model,
        input=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "input_image",
                        "image_url": f"data:image/jpeg;base64,{encoded_image}",
                    },
                    {"type": "input_text", "text": prompt},
                ],
            }
        ],
        instructions=VISION_SYSTEM_PROMPT,
        temperature=0.2,
        max_output_tokens=400,
    )

    text = _extract_output_text(response)
    if not text:
        raise RuntimeError("Model returned an empty reasoning response.")
    return text


def build_agent_qa_prompt(report_payload: Mapping[str, Any], question: str) -> str:
    if not question.strip():
        raise ValueError("Question must be non-empty.")

    context = {
        "source_filename": report_payload.get("source_filename"),
        "impression": str(report_payload.get("impression", "")).strip(),
        "reasoning": report_payload.get("reasoning"),
        "p_abnormal": report_payload.get("p_abnormal"),
        "threshold": report_payload.get("threshold"),
        "confidence_tier": report_payload.get("confidence_tier"),
        "reasoning_error": report_payload.get("reasoning_error"),
    }
    context_json = json.dumps(context, ensure_ascii=True, indent=2)
    return (
        "Answer the user question using only the context below.\n"
        "Rules:\n"
        "- Be concise.\n"
        "- Do not claim findings that are not in the context.\n"
        "- If the answer is not supported by the context, say that directly.\n"
        "- Include a short uncertainty note when confidence is borderline.\n\n"
        f"Context:\n{context_json}\n\n"
        f"User question:\n{question.strip()}"
    )


def answer_question_about_report(
    report_payload: Mapping[str, Any],
    question: str,
    model: str = DEFAULT_QA_MODEL,
) -> str:
    client = get_openai_client()
    prompt = build_agent_qa_prompt(report_payload, question)
    response = client.responses.create(
        model=model,
        instructions=AGENT_QA_SYSTEM_PROMPT,
        input=prompt,
        temperature=0.1,
        max_output_tokens=260,
    )

    text = _extract_output_text(response)
    if not text:
        raise RuntimeError("Model returned an empty answer.")
    return text
