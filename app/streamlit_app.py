from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Any

import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from bda_chest.evaluation import evaluate_response, load_medgemma_judge
from bda_chest.llm import (
    DEFAULT_LLAMA_MODEL,
    analyze_xray_image,
    analyze_xray_image_llama,
    answer_question_about_report,
    answer_question_about_report_llama,
    load_llama_model,
)
from bda_chest.models import checkpoint_metadata, load_checkpoint
from bda_chest.pipeline import infer_from_pil, load_inference_bundle
from bda_chest.version import APP_VERSION

LATEST_REPORT_STATE_KEY = "latest_inference_payload"
AGENT_CHAT_STATE_KEY = "agent_chat_messages"


def resolve_project_path(path_str: str) -> Path:
    path = Path(path_str).expanduser()
    if not path.is_absolute():
        path = ROOT / path
    return path.resolve()


@st.cache_resource(show_spinner=False)
def load_bundle_cached(checkpoint_path: str):
    resolved = str(resolve_project_path(checkpoint_path))
    return load_inference_bundle(resolved)


@st.cache_data(show_spinner=False)
def load_checkpoint_metadata_cached(path: str, mtime: float) -> dict[str, Any]:
    del mtime
    checkpoint = load_checkpoint(path, map_location="cpu")
    return checkpoint_metadata(checkpoint)


def get_checkpoint_metadata(path: Path) -> dict[str, Any]:
    checkpoint_path = resolve_project_path(path)
    return load_checkpoint_metadata_cached(
        str(checkpoint_path), checkpoint_path.stat().st_mtime
    )


PROVIDER_LLAMA = "Llama (Local)"
PROVIDER_OPENAI = "OpenAI (API)"


@st.cache_resource(show_spinner="Loading Llama radiology model...")
def load_llama_cached():
    return load_llama_model()


# MedGemma judge — adapted from Radiology_Assistant_Evaluation.ipynb (cell 3).
# The notebook uses MedGemma as an LLM judge to score radiology assistant responses
# on a 1-5 correctness scale.
@st.cache_resource(show_spinner="Loading MedGemma evaluation model...")
def load_medgemma_cached():
    return load_medgemma_judge()


def maybe_run_reasoning(
    image,
    payload: dict[str, Any],
    provider: str,
    model_name: str,
) -> dict[str, Any]:
    if provider == PROVIDER_LLAMA:
        try:
            llama_model, llama_tokenizer = load_llama_cached()
            reasoning = analyze_xray_image_llama(
                image=image,
                p_abnormal=float(payload["p_abnormal"]),
                tier=str(payload["confidence_tier"]),
                model=llama_model,
                tokenizer=llama_tokenizer,
            )
            return {"ok": True, "model": DEFAULT_LLAMA_MODEL, "text": reasoning}
        except Exception as exc:
            return {
                "ok": False,
                "model": DEFAULT_LLAMA_MODEL,
                "error": f"{type(exc).__name__}: {exc}",
            }
    chosen_model = model_name.strip() or "gpt-4.1"
    try:
        reasoning = analyze_xray_image(
            image=image,
            p_abnormal=float(payload["p_abnormal"]),
            tier=str(payload["confidence_tier"]),
            model=chosen_model,
        )
        return {"ok": True, "model": chosen_model, "text": reasoning}
    except Exception as exc:
        return {
            "ok": False,
            "model": chosen_model,
            "error": f"{type(exc).__name__}: {exc}",
        }


def maybe_answer_question(
    payload: dict[str, Any],
    question: str,
    provider: str,
    model_name: str,
) -> dict[str, Any]:
    if provider == PROVIDER_LLAMA:
        try:
            llama_model, llama_tokenizer = load_llama_cached()
            answer = answer_question_about_report_llama(
                report_payload=payload,
                question=question,
                model=llama_model,
                tokenizer=llama_tokenizer,
            )
            return {"ok": True, "model": DEFAULT_LLAMA_MODEL, "text": answer}
        except Exception as exc:
            return {
                "ok": False,
                "model": DEFAULT_LLAMA_MODEL,
                "error": f"{type(exc).__name__}: {exc}",
            }
    chosen_model = model_name.strip() or "gpt-4.1-mini"
    try:
        answer = answer_question_about_report(
            report_payload=payload,
            question=question,
            model=chosen_model,
        )
        return {"ok": True, "model": chosen_model, "text": answer}
    except Exception as exc:
        return {
            "ok": False,
            "model": chosen_model,
            "error": f"{type(exc).__name__}: {exc}",
        }


def render_chat_component(
    payload: dict[str, Any], provider: str, model_name: str
) -> None:
    if AGENT_CHAT_STATE_KEY not in st.session_state:
        st.session_state[AGENT_CHAT_STATE_KEY] = []

    messages = st.session_state[AGENT_CHAT_STATE_KEY]

    st.divider()
    st.subheader("Chat with Radiology Assistant")
    st.caption("Ask questions about this specific analysis.")

    clear_col, _ = st.columns([1, 4])
    with clear_col:
        if st.button("Clear Chat", key="clear_chat_button", width="stretch"):
            st.session_state[AGENT_CHAT_STATE_KEY] = []
            st.rerun()

    for message in messages:
        with st.chat_message(str(message.get("role", "assistant"))):
            st.markdown(str(message.get("content", "")))

    question = st.chat_input(
        "Ask about findings, confidence, or rationale...", key="chat_input_box"
    )
    if question:
        messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer_result = maybe_answer_question(
                    payload, question, provider, model_name
                )
            if bool(answer_result.get("ok")):
                answer_text = str(answer_result.get("text", "")).strip()
                st.markdown(answer_text)
                messages.append({"role": "assistant", "content": answer_text})
            else:
                error_text = str(answer_result.get("error", "Unknown error"))
                st.error(error_text)
                messages.append(
                    {"role": "assistant", "content": f"Error: {error_text}"}
                )
        st.session_state[AGENT_CHAT_STATE_KEY] = messages


def render_inference_page(
    checkpoint_path: str,
    threshold: float,
    llm_enabled: bool,
    llm_provider: str,
    llm_model: str,
    llm_qa_model: str,
    eval_enabled: bool = False,
) -> None:
    uploaded = st.file_uploader(
        "Upload chest X-ray image",
        type=["png", "jpg", "jpeg"],
        accept_multiple_files=False,
    )
    if uploaded is None:
        st.info("Upload a chest X-ray image to run inference.")
        # Clear state if no file is uploaded to avoid stale chats
        if LATEST_REPORT_STATE_KEY in st.session_state:
            del st.session_state[LATEST_REPORT_STATE_KEY]
        if AGENT_CHAT_STATE_KEY in st.session_state:
            del st.session_state[AGENT_CHAT_STATE_KEY]
        return

    from PIL import Image

    pil_image = Image.open(uploaded).convert("RGB")
    left_col, right_col = st.columns([1, 1.2])
    with left_col:
        st.image(pil_image, caption=uploaded.name, use_container_width=True)

    if not st.button("Analyze Image", type="primary", width="stretch"):
        # If we already have a result for THIS image, show the chat
        cached_payload = st.session_state.get(LATEST_REPORT_STATE_KEY)
        if cached_payload and cached_payload.get("source_filename") == uploaded.name:
            with right_col:
                st.success("Analysis loaded from session.")
                st.subheader("Impression")
                st.write(cached_payload["impression"])
                if cached_payload.get("reasoning"):
                    st.subheader("LLM Reasoning")
                    st.write(cached_payload["reasoning"])
                render_chat_component(cached_payload, llm_provider, llm_qa_model)
        return

    with right_col:
        try:
            with st.spinner("Loading model and running inference..."):
                started = time.perf_counter()
                bundle = load_bundle_cached(checkpoint_path)
                payload, probability = infer_from_pil(
                    bundle=bundle,
                    image=pil_image,
                    threshold=threshold,
                )
                elapsed_ms = (time.perf_counter() - started) * 1000.0
        except FileNotFoundError as exc:
            st.error(str(exc))
            return
        except Exception as exc:
            st.error("Inference failed.")
            st.exception(exc)
            return

        payload["source_filename"] = uploaded.name
        payload["checkpoint"] = str(bundle.checkpoint_path)
        payload["device"] = str(bundle.device)
        payload["reasoning"] = None
        payload["reasoning_model"] = None
        payload["reasoning_error"] = None

        st.success(f"Inference complete in {elapsed_ms:.1f} ms")
        st.subheader("Impression")
        st.write(payload["impression"])

        if llm_enabled and payload["prediction"] == "PNEUMONIA":
            st.subheader("LLM Reasoning")
            with st.spinner("Generating radiology reasoning..."):
                reasoning_result = maybe_run_reasoning(
                    pil_image, payload, llm_provider, llm_model
                )
            payload["reasoning_model"] = reasoning_result.get("model")
            if bool(reasoning_result.get("ok")):
                payload["reasoning"] = str(reasoning_result.get("text", "")).strip()
                st.write(payload["reasoning"])

                # MedGemma evaluation — adapted from Radiology_Assistant_Evaluation.ipynb
                # cell 5: feedback = judge.evaluate(q, answer, payload['impression'], ...)
                if eval_enabled and payload["reasoning"]:
                    st.subheader("MedGemma Evaluation")
                    with st.spinner("Running MedGemma judge..."):
                        try:
                            judge_model, judge_tokenizer = load_medgemma_cached()
                            eval_result = evaluate_response(
                                model=judge_model,
                                tokenizer=judge_tokenizer,
                                question="What are the radiologic findings?",
                                answer=payload["reasoning"],
                                context=str(payload.get("impression", "")),
                                ground_truth="Verify clinical accuracy.",
                            )
                            score = eval_result.get("correctness_score")
                            justification = eval_result.get("justification")
                            if score is not None:
                                st.metric("Correctness Score", f"{score} / 5")
                            if justification:
                                st.info(justification)
                            elif not score:
                                # JSON parsing failed; show raw judge output
                                st.caption("Raw judge output:")
                                st.text(eval_result.get("raw", ""))
                            payload["eval_score"] = score
                            payload["eval_justification"] = justification
                        except Exception as exc:
                            st.warning(
                                f"Evaluation failed: {type(exc).__name__}: {exc}"
                            )
            else:
                payload["reasoning_error"] = str(
                    reasoning_result.get("error", "Unknown error")
                )
                st.error(payload["reasoning_error"])
        elif llm_enabled:
            st.info("LLM reasoning is not invoked for normal predictions.")

        st.subheader("Run Metadata")
        st.code(
            "\n".join(
                [
                    f"checkpoint={payload['checkpoint']}",
                    f"device={payload['device']}",
                    f"threshold={payload['threshold']}",
                    f"p_abnormal={probability:.6f}",
                ]
            )
        )

        st.session_state[LATEST_REPORT_STATE_KEY] = dict(payload)
        st.session_state[AGENT_CHAT_STATE_KEY] = []  # Reset chat for new analysis

        st.download_button(
            label="Download Report JSON",
            data=json.dumps(payload, indent=2),
            file_name=f"{Path(uploaded.name).stem}_report.json",
            mime="application/json",
            width="stretch",
        )

        render_chat_component(payload, llm_provider, llm_qa_model)


def render_model_info_page(checkpoint_path: str) -> None:
    st.subheader("Model Info")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Patch Size", "16", width="stretch")
    c2.metric("Embed Dim", "192", width="stretch")
    c3.metric("Depth", "12", width="stretch")
    c4.metric("Heads", "3", width="stretch")

    try:
        metadata = get_checkpoint_metadata(resolve_project_path(checkpoint_path))
    except FileNotFoundError as exc:
        st.error(str(exc))
        return
    except Exception as exc:
        st.error("Unable to load checkpoint metadata.")
        st.exception(exc)
        return

    st.caption("Checkpoint Metadata")
    st.code(
        "\n".join(
            [
                f"checkpoint={resolve_project_path(checkpoint_path)}",
                f"epoch={metadata.get('epoch', 'N/A')}",
                f"best_val_loss={metadata.get('best_val_loss', 'N/A')}",
                f"class_to_idx={metadata.get('class_to_idx', 'N/A')}",
            ]
        )
    )


def render_ask_agent_page(llm_provider: str, llm_qa_model: str) -> None:
    st.subheader("Ask Agent")
    st.caption("Research prototype only. Not for clinical use.")

    context_source = st.radio(
        "Context Source",
        ["Latest inference in this session", "Upload report JSON"],
        index=0,
        horizontal=True,
    )

    payload: dict[str, Any] | None = None
    if context_source == "Latest inference in this session":
        cached_payload = st.session_state.get(LATEST_REPORT_STATE_KEY)
        if isinstance(cached_payload, dict):
            payload = dict(cached_payload)
        else:
            st.info("Run one inference first, then ask questions here.")
            return
    else:
        uploaded_report = st.file_uploader(
            "Upload report JSON",
            type=["json"],
            key="ask_agent_report_json",
            accept_multiple_files=False,
        )
        if uploaded_report is None:
            st.info("Upload a report JSON file to use it as Ask Agent context.")
            return
        try:
            parsed = json.load(uploaded_report)
        except Exception as exc:
            st.error(f"Unable to parse JSON: {exc}")
            return
        if not isinstance(parsed, dict):
            st.error("Report JSON must be an object.")
            return
        payload = parsed

    with st.expander("Context Preview", expanded=False):
        st.json(
            {
                "impression": payload.get("impression"),
                "reasoning": payload.get("reasoning"),
                "confidence_tier": payload.get("confidence_tier"),
                "p_abnormal": payload.get("p_abnormal"),
            }
        )

    render_chat_component(payload, llm_provider, llm_qa_model)


def main() -> None:
    st.set_page_config(page_title="BDA - Chest X-ray Analyzer", layout="wide")
    st.title("BDA - Chest X-ray Analyzer")
    st.caption("Research prototype only. Not for clinical use.")

    with st.sidebar:
        st.title("BDA - Chest X-ray Analyzer")
        st.caption("Research prototype only. Not for clinical use.")
        st.caption(APP_VERSION)

        st.subheader("Page")
        page = st.radio(
            "Page",
            ["Inference", "Model Info", "Ask Agent"],
            index=0,
            label_visibility="collapsed",
        )

        st.header("Run Settings")
        checkpoint_path = st.text_input(
            "Checkpoint Path",
            value="eva_x_tiny_binary_best.pt",
        )
        threshold = st.slider("Threshold", min_value=0.0, max_value=1.0, value=0.5)
        llm_enabled = st.checkbox(
            "LLM reasoning",
            value=False,
            disabled=(page != "Inference"),
            help="Llama runs locally with GPU. OpenAI requires OPENAI_API_KEY.",
        )
        # MedGemma evaluation toggle — from Radiology_Assistant_Evaluation.ipynb.
        # Uses google/medgemma-1.5-4b-it to score reasoning on a 1-5 scale.
        eval_enabled = st.checkbox(
            "MedGemma evaluation",
            value=False,
            disabled=(not llm_enabled or page != "Inference"),
            help="Score LLM reasoning with MedGemma judge (requires GPU).",
        )
        llm_provider = st.selectbox(
            "LLM Provider",
            [PROVIDER_LLAMA, PROVIDER_OPENAI],
            index=0,
        )
        if llm_provider == PROVIDER_OPENAI:
            llm_model = st.text_input(
                "LLM Model",
                value="gpt-4.1",
                disabled=(page != "Inference"),
            )
            llm_qa_model = st.text_input(
                "LLM Q&A Model",
                value="gpt-4.1-mini",
                disabled=(page == "Model Info"),
            )
        else:
            st.caption(f"Model: `{DEFAULT_LLAMA_MODEL}`")
            llm_model = DEFAULT_LLAMA_MODEL
            llm_qa_model = DEFAULT_LLAMA_MODEL

        st.divider()
        st.markdown("EECS E6893 Big Data Analytics Midterm Project")

    if page == "Inference":
        render_inference_page(
            checkpoint_path,
            threshold,
            llm_enabled,
            llm_provider,
            llm_model,
            llm_qa_model,
            eval_enabled=eval_enabled,
        )
    elif page == "Model Info":
        render_model_info_page(checkpoint_path)
    else:
        render_ask_agent_page(llm_provider, llm_qa_model)


if __name__ == "__main__":
    main()
