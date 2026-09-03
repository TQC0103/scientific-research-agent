from contextlib import nullcontext
from types import SimpleNamespace

import pytest

from scripts.run_end_to_end_transformers import TransformersRuntime


class FakeOutOfMemoryError(RuntimeError):
    pass


class FakeCuda:
    def __init__(self) -> None:
        self.empty_cache_calls = 0

    def empty_cache(self) -> None:
        self.empty_cache_calls += 1

    def reset_peak_memory_stats(self, device: int) -> None:
        assert device == 0

    def max_memory_allocated(self, device: int) -> int:
        return 12

    def max_memory_reserved(self, device: int) -> int:
        return 15

    def memory_allocated(self, device: int) -> int:
        return 8

    def memory_reserved(self, device: int) -> int:
        return 9


class FakeInputs(dict):
    def to(self, device: str) -> "FakeInputs":
        assert device == "cuda:0"
        return self


class FakeTokenizer:
    pad_token_id = 0
    eos_token_id = 1

    def apply_chat_template(self, *args, **kwargs) -> str:
        return "prompt"

    def __call__(self, *args, **kwargs) -> FakeInputs:
        return FakeInputs(input_ids=SimpleNamespace(shape=(1, 10)))


def test_transformers_runtime_cleans_cuda_after_oom() -> None:
    cuda = FakeCuda()
    runtime = TransformersRuntime.__new__(TransformersRuntime)
    runtime.torch = SimpleNamespace(
        cuda=cuda,
        OutOfMemoryError=FakeOutOfMemoryError,
        inference_mode=nullcontext,
    )
    runtime.tokenizer = FakeTokenizer()
    runtime.model = SimpleNamespace(
        generate=lambda **kwargs: (_ for _ in ()).throw(FakeOutOfMemoryError("oom"))
    )
    runtime.calls = 0
    runtime.successful_calls = 0
    runtime.oom_calls = 0
    runtime.peak_allocated_bytes = 0
    runtime.peak_reserved_bytes = 0
    runtime.post_call_allocated_max_bytes = 0
    runtime.post_call_reserved_max_bytes = 0

    with pytest.raises(FakeOutOfMemoryError):
        runtime.generate("question", max_new_tokens=10)

    assert runtime.calls == 1
    assert runtime.successful_calls == 0
    assert runtime.oom_calls == 1
    assert cuda.empty_cache_calls == 2
    assert runtime.peak_allocated_bytes == 12
    assert runtime.peak_reserved_bytes == 15
    assert runtime.post_call_allocated_max_bytes == 8
    assert runtime.post_call_reserved_max_bytes == 9
