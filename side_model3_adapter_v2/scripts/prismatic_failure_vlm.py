"""Run local Prismatic VLM failure classification over request JSONL."""

from __future__ import annotations

import argparse
import json
import os
import sys
from contextlib import redirect_stdout
from pathlib import Path
from typing import Any

import torch
from PIL import Image


DEFAULT_MODEL_PATH = Path(
    "/data/public/VLA-Adapter/pretrained_models/"
    "prism-qwen25-extra-dinosiglip-224px-0_5b"
)
DEFAULT_HF_CACHE = Path("/data/public/VLA-Adapter/hf_cache/transformers")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--hf-cache", type=Path, default=DEFAULT_HF_CACHE)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    return parser.parse_args()


def _load_model(model_path: Path, hf_cache: Path):
    os.environ.setdefault("HUGGINGFACE_HUB_CACHE", str(hf_cache))
    with redirect_stdout(sys.stderr):
        from prismatic import load

        model = load(model_path)
    model.to(torch.device("cuda"), dtype=torch.bfloat16).eval()
    return model


def _extract_json(text: str) -> dict[str, Any]:
    candidate = text.strip()
    if candidate.startswith("```"):
        candidate = candidate.split("\n", 1)[1].rsplit("```", 1)[0].strip()
    if not candidate.startswith("{"):
        candidate = candidate[candidate.find("{") : candidate.rfind("}") + 1]
    payload = json.loads(candidate)
    if not isinstance(payload, dict):
        raise ValueError("VLM response is not a JSON object")
    return payload


def _fallback_response(reason: str) -> dict[str, Any]:
    return {
        "primary_failure": "其他/无法判断",
        "secondary_failure": None,
        "outcome_awareness_failure": False,
        "recovery_failure": False,
        "furthest_stage": "无法判断",
        "confidence": 0.0,
        "short_evidence": f"本地视觉模型未能确认结构化结果：{reason}",
    }


def _build_prompt(request: dict[str, Any]) -> str:
    schema = json.dumps(request["response_schema"], ensure_ascii=False)
    return (
        f"{request['system_prompt']}\n\n"
        f"{request['episode_prompt']}\n\n"
        "只输出一个 JSON object，不要 markdown、解释或额外字段。"
        "所有 taxonomy value 必须逐字使用给定中文 enum；boolean 必须是 true 或 false。\n"
        f"response_schema={schema}"
    )


def _classify(model: Any, request: dict[str, Any], max_new_tokens: int) -> dict[str, Any]:
    image = Image.open(request["image_path"]).convert("RGB")
    prompt_builder = model.get_prompt_builder(system_prompt="You are a precise visual analyst.")
    prompt_builder.add_turn(role="human", message=_build_prompt(request))
    generated = model.generate(
        image,
        prompt_builder.get_prompt(),
        do_sample=False,
        max_new_tokens=max_new_tokens,
        min_length=1,
    )
    return _extract_json(generated)


def main() -> None:
    args = parse_args()
    model = _load_model(args.model_path, args.hf_cache)
    for line_number, line in enumerate(sys.stdin, start=1):
        if not line.strip():
            continue
        request = json.loads(line)
        try:
            response = _classify(model, request, args.max_new_tokens)
        except Exception as exc:
            print(
                f"VLM fallback at request {line_number}: {exc}",
                file=sys.stderr,
                flush=True,
            )
            response = _fallback_response(str(exc))
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
