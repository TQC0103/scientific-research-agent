"""Run the claim-verifier development benchmark with a CUDA model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.claim_verifier import (
    load_claim_verifier_suite,
    run_claim_verifier_benchmark,
    write_claim_verifier_outputs,
)

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class TransformersGenerator:
    def __init__(self, model_name: str, revision: str | None, *, batch_size: int,
                 max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("The claim-verifier benchmark requires CUDA; use Kaggle.")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, dtype=torch.float16, device_map={"": 0}
        )
        self.model.generation_config.temperature = None
        self.model.generation_config.top_p = None
        self.model.generation_config.top_k = None
        self.batch_size = batch_size
        self.max_new_tokens = max_new_tokens
        self.batch_calls = 0

    def __call__(self, prompts: list[str]) -> list[str]:
        outputs = []
        for start in range(0, len(prompts), self.batch_size):
            batch = prompts[start : start + self.batch_size]
            chats = [[{"role": "user", "content": prompt}] for prompt in batch]
            rendered = self.tokenizer.apply_chat_template(
                chats, tokenize=False, add_generation_prompt=True, enable_thinking=False
            )
            encoded = self.tokenizer(rendered, return_tensors="pt", padding=True).to(
                self.model.device
            )
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            width = encoded["input_ids"].shape[1]
            outputs.extend(
                self.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
            )
            self.batch_calls += 1
        return outputs


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=1200)
    parser.add_argument("--smoke-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _args()
    suite = load_claim_verifier_suite(args.suite)
    generator = TransformersGenerator(
        args.model,
        args.model_revision,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if args.smoke_cases:
        smoke = suite.model_copy(update={"cases": suite.cases[: args.smoke_cases]})
        report = run_claim_verifier_benchmark(
            smoke, generator, model=args.model, model_revision=args.model_revision
        )
        if report.schema_valid_rate != 1.0:
            raise RuntimeError("Claim-verifier smoke returned invalid structured output.")
        print(json.dumps({"smoke_cases": report.case_count,
                          "smoke_schema_valid_rate": report.schema_valid_rate}, indent=2))
        generator.batch_calls = 0
    report = run_claim_verifier_benchmark(
        suite, generator, model=args.model, model_revision=args.model_revision
    )
    write_claim_verifier_outputs(report, args.output_dir)
    print(json.dumps(report.model_dump(mode="json", exclude={"results"}), indent=2))


if __name__ == "__main__":
    main()
