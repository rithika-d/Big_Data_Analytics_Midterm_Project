from __future__ import annotations

import argparse
import json

import torch

from .bda_chest.pipeline import load_inference_bundle, infer_from_pil
from .bda_chest.utils import load_image
from .bda_chest.reporting import classify_confidence_tier
from .bda_chest.llm import build_reasoning_prompt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the chest X-ray diagnosis pipeline."
    )
    parser.add_argument(
        "--image", required=True, help="Path or URL to a chest X-ray image"
    )
    parser.add_argument(
        "--checkpoint", required=True, help="Path to the trained EVA-X checkpoint"
    )
    parser.add_argument("--backend", required=True, choices=("llama", "chexagent"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    # Stage 1: classify
    bundle = load_inference_bundle(args.checkpoint)
    image = load_image(args.image)
    _, p_abnormal = infer_from_pil(bundle, image, threshold=args.threshold)

    output = {
        "source": args.image,
        "p_abnormal": round(float(p_abnormal), 4),
        "y_pred": int(p_abnormal > args.threshold),
        "reasoning": None,
    }

    if p_abnormal <= args.threshold:
        print(json.dumps(output, indent=2))
        return

    # Stage 2: load LLM lazily (only when abnormal)
    tier = classify_confidence_tier(p_abnormal, args.threshold)
    prompt = build_reasoning_prompt(p_abnormal, tier)

    if args.backend == "llama":
        from .bda_chest.llm import load_llama_model, make_llama_generate_fn

        llm_model, tokenizer = load_llama_model()
        generate_fn = make_llama_generate_fn(llm_model, tokenizer)
    else:
        from .bda_chest.llm import load_chexagent, make_chexagent_generate_fn

        device = bundle.device
        llm_model, tokenizer = load_chexagent(device=device)
        generate_fn = make_chexagent_generate_fn(llm_model, tokenizer, device)

    output["reasoning"] = generate_fn(image, prompt)
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
