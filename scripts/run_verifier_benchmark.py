"""Run the controlled verifier benchmark with a CUDA Transformers model."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.verifier import (
    load_verifier_benchmark,
    run_verifier_benchmark,
    write_verifier_outputs,
)

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class TransformersGenerator:
    def __init__(
        self,
        model_name: str,
        revision: str | None,
        *,
        batch_size: int,
        max_new_tokens: int,
    ) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("The verifier benchmark requires CUDA; use Kaggle Control Plane.")
        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=torch.float16,
            device_map={"": 0},
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
                chats,
                tokenize=False,
                add_generation_prompt=True,
                enable_thinking=False,
            )
            encoded = self.tokenizer(
                rendered, return_tensors="pt", padding=True
            ).to(self.model.device)
            with self.torch.inference_mode():
                generated = self.model.generate(
                    **encoded,
                    do_sample=False,
                    max_new_tokens=self.max_new_tokens,
                    pad_token_id=self.tokenizer.pad_token_id,
                )
            prompt_width = encoded["input_ids"].shape[1]
            outputs.extend(
                self.tokenizer.batch_decode(
                    generated[:, prompt_width:], skip_special_tokens=True
                )
            )
            self.batch_calls += 1
        return outputs


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--definition", type=Path, required=True)
    parser.add_argument("--source-suite", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--max-new-tokens", type=int, default=400)
    parser.add_argument("--smoke-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    benchmark = load_verifier_benchmark(args.definition, args.source_suite)
    generator = TransformersGenerator(
        args.model,
        args.model_revision,
        batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens,
    )
    if args.smoke_cases:
        smoke = benchmark.model_copy(
            update={"cases": benchmark.cases[: args.smoke_cases]}
        )
        smoke_report = run_verifier_benchmark(
            smoke,
            generator,
            model=args.model,
            model_revision=args.model_revision,
        )
        if smoke_report.parse_failure_count:
            raise RuntimeError("Verifier smoke produced an invalid structured response.")
        print(
            json.dumps(
                {
                    "smoke_cases": smoke_report.case_count,
                    "smoke_model_calls": smoke_report.model_calls,
                    "smoke_initial_accuracy": smoke_report.initial_metrics["accuracy"],
                },
                indent=2,
            )
        )
        generator.batch_calls = 0
    report = run_verifier_benchmark(
        benchmark,
        generator,
        model=args.model,
        model_revision=args.model_revision,
    )
    write_verifier_outputs(report, args.output_dir)
    print(
        json.dumps(
            report.model_dump(mode="json", exclude={"results"}),
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
