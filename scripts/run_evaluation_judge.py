"""Run the advisory evaluation-case judge with a local Transformers model.

Heavy execution belongs on Kaggle Control Plane. Imports are intentionally
dynamic so the core project does not require a GPU inference stack.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.judge import (
    build_judge_prompt,
    build_judge_report,
    parse_judge_response,
)
from app.evaluation.loader import load_suite


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="Qwen/Qwen2.5-3B-Instruct")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    return parser.parse_args()


def _generate(
    prompts: list[str], model_name: str, batch_size: int, max_new_tokens: int
) -> list[str]:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("The judge runner requires CUDA; use Kaggle Control Plane.")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float16,
        device_map="auto",
    )
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    outputs: list[str] = []
    for start in range(0, len(prompts), batch_size):
        batch = prompts[start : start + batch_size]
        chats = [[{"role": "user", "content": prompt}] for prompt in batch]
        rendered = tokenizer.apply_chat_template(
            chats, tokenize=False, add_generation_prompt=True
        )
        encoded = tokenizer(rendered, return_tensors="pt", padding=True).to(model.device)
        with torch.inference_mode():
            generated = model.generate(
                **encoded,
                do_sample=False,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,
            )
        prompt_width = encoded["input_ids"].shape[1]
        outputs.extend(
            tokenizer.batch_decode(generated[:, prompt_width:], skip_special_tokens=True)
        )
    return outputs


def main() -> None:
    args = _parse_args()
    suite = load_suite(args.suite)
    prompts = [build_judge_prompt(case) for case in suite.cases]
    responses = _generate(prompts, args.model, args.batch_size, args.max_new_tokens)
    results = [
        parse_judge_response(case, response)
        for case, response in zip(suite.cases, responses, strict=True)
    ]
    report = build_judge_report(suite, args.model, results)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report.model_dump(mode="json"), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(report.model_dump(mode="json", exclude={"results"}), indent=2))


if __name__ == "__main__":
    main()
