"""Run native-label SciFact classification with cited documents supplied."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from app.evaluation.external import load_scifact
from app.evaluation.scifact import run_scifact_benchmark, write_scifact_outputs

DEFAULT_MODEL = "Qwen/Qwen3-4B"
DEFAULT_MODEL_REVISION = "1cfa9a7208912126459214e8b04321603b3df60c"


class TransformersGenerator:
    def __init__(self, model_name: str, revision: str | None, *, batch_size: int,
                 max_new_tokens: int) -> None:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if not torch.cuda.is_available():
            raise RuntimeError("SciFact model evaluation requires CUDA; use Kaggle.")
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

    def _generate(self, prompts: list[str], batch_size: int) -> list[str]:
        outputs = []
        for start in range(0, len(prompts), batch_size):
            batch = prompts[start : start + batch_size]
            try:
                chats = [[{"role": "user", "content": prompt}] for prompt in batch]
                rendered = self.tokenizer.apply_chat_template(
                    chats, tokenize=False, add_generation_prompt=True, enable_thinking=False
                )
                encoded = self.tokenizer(
                    rendered, return_tensors="pt", padding=True, truncation=True,
                    max_length=7168
                ).to(self.model.device)
                with self.torch.inference_mode():
                    generated = self.model.generate(
                        **encoded, do_sample=False, max_new_tokens=self.max_new_tokens,
                        pad_token_id=self.tokenizer.pad_token_id
                    )
                width = encoded["input_ids"].shape[1]
                outputs.extend(
                    self.tokenizer.batch_decode(generated[:, width:], skip_special_tokens=True)
                )
                self.batch_calls += 1
            except RuntimeError as exc:
                if "out of memory" not in str(exc).casefold() or batch_size == 1:
                    raise
                self.torch.cuda.empty_cache()
                outputs.extend(self._generate(batch, max(1, batch_size // 2)))
        return outputs

    def __call__(self, prompts: list[str]) -> list[str]:
        return self._generate(prompts, self.batch_size)


def _args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus", type=Path, required=True)
    parser.add_argument("--claims", type=Path, required=True)
    parser.add_argument("--source-split", default="dev")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--model-revision", default=DEFAULT_MODEL_REVISION)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--smoke-cases", type=int, default=0)
    return parser.parse_args()


def main() -> None:
    args = _args()
    cases = load_scifact(
        args.corpus, args.claims, source_split=args.source_split, limit=args.limit
    )
    generator = TransformersGenerator(
        args.model, args.model_revision, batch_size=args.batch_size,
        max_new_tokens=args.max_new_tokens
    )
    if args.smoke_cases:
        smoke = run_scifact_benchmark(
            cases[: args.smoke_cases], generator, model=args.model,
            model_revision=args.model_revision
        )
        print(json.dumps({
            "smoke_cases": smoke.case_count,
            "smoke_accuracy": smoke.label_metrics["accuracy"],
            "smoke_parse_failures": smoke.parse_failure_count,
            "smoke_failures": [
                {
                    "claim_id": result.claim_id,
                    "error": result.parse_error,
                    "raw_response": result.raw_response[:1000],
                }
                for result in smoke.results
                if not result.schema_valid
            ],
        }, ensure_ascii=False, indent=2))
        if smoke.parse_failure_count == smoke.case_count:
            raise RuntimeError("Every SciFact smoke response failed structured parsing.")
        generator.batch_calls = 0
    report = run_scifact_benchmark(
        cases, generator, model=args.model, model_revision=args.model_revision
    )
    write_scifact_outputs(report, args.output_dir)
    print(json.dumps(report.model_dump(mode="json", exclude={"results"}), indent=2))


if __name__ == "__main__":
    main()
