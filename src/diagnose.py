from __future__ import annotations

import argparse
import json

import torch

from .cxr_pipeline.diagnosis import diagnose_chest_xray
from .cxr_pipeline.model import load_model_for_inference


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
    parser.add_argument(
        "--pretrained-weights", required=True, help="Path to EVA-X pretrained weights"
    )
    parser.add_argument("--backend", required=True, choices=("llama", "chexagent"))
    parser.add_argument("--threshold", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    eva_model = load_model_for_inference(
        args.checkpoint, args.pretrained_weights, device
    )

    if args.backend == "llama":
        from .cxr_pipeline.llama_backend import load_llama_model, make_llama_generate_fn

        llm_model, tokenizer = load_llama_model()
        generate_fn = make_llama_generate_fn(llm_model, tokenizer)
    else:
        from .cxr_pipeline.chexagent import load_chexagent, make_chexagent_generate_fn

        llm_model, tokenizer = load_chexagent(device=device)
        generate_fn = make_chexagent_generate_fn(llm_model, tokenizer, device)

    result = diagnose_chest_xray(
        image_path=args.image,
        eva_model=eva_model,
        device=device,
        llm_generate_fn=generate_fn,
        threshold=args.threshold,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
