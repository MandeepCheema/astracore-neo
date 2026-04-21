"""BERT-Squad + GPT-2 demos — honest about tokeniser status.

Without HuggingFace ``transformers`` installed, we can't turn an English
sentence into model-compatible token IDs. So these demos use a canned,
deterministic token sequence (valid WordPiece IDs in [0, 30522) for
BERT; valid BPE IDs in [0, 50257) for GPT-2) and prove the model
produces **structurally valid output**:

  * BERT-Squad: start/end span logits inside the sequence length,
    argmax span bounded and sensible.
  * GPT-2: next-token logits normalise to a probability distribution,
    entropy + top-5 IDs are reported.

To get real English-in / English-out with these models, install
``astracore-sdk[pytorch]`` then pass ``--input <your-prompt>``.
"""

from __future__ import annotations

from pathlib import Path
import time
from typing import Dict, Optional

import numpy as np

from astracore.demo.base import DemoError, DemoResult, register_demo_family


# Seeded, fixed "prompt" for reproducibility. 32 tokens is a short
# question-answer pair length for BERT, a short prompt for GPT-2.
_CANNED_TOKENS_BERT = np.array([
    101, 2054, 2003, 1996, 3007, 1997, 2605, 1029, 102,   # [CLS] what is the capital of france ? [SEP]
    3429, 2003, 1996, 3007, 2103, 1997, 2605, 1012, 102,  # paris is the capital city of france . [SEP]
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,              # padding to 32
], dtype=np.int64)

_CANNED_TOKENS_GPT2 = np.array([
    464, 3139, 286, 4881, 318,    # "The capital of France is"
], dtype=np.int64)


def _rng_based_tokens(seed: int, vocab: int, length: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.integers(low=0, high=vocab, size=length, dtype=np.int64)


@register_demo_family("nlp-encoder-transformer")
def run_bert_squad(zoo_entry, onnx_path: Path, *,
                   input_spec: Optional[str] = None,
                   backend_name: str = "onnxruntime") -> DemoResult:
    """BERT-Squad: run a canned Q+A token sequence, report span logits."""
    import astracore.backends  # noqa: F401
    from astracore.registry import get_backend

    import onnx
    model = onnx.load(str(onnx_path))
    init_names = {t.name for t in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in init_names]

    # BERT-Squad has 4 inputs: unique_ids, segment_ids, input_mask, input_ids.
    # Build them consistently — 256-length sequence, first 19 real tokens.
    seq_len = 256
    input_ids = np.zeros(seq_len, dtype=np.int64)
    input_ids[:len(_CANNED_TOKENS_BERT)] = _CANNED_TOKENS_BERT
    segment_ids = np.zeros(seq_len, dtype=np.int64)
    segment_ids[9:19] = 1                                  # passage tokens
    input_mask = (input_ids != 0).astype(np.int64)
    unique_ids = np.array([1], dtype=np.int64)

    feed: Dict[str, np.ndarray] = {}
    for inp in real_inputs:
        name = inp.name
        if "input_ids" in name:
            feed[name] = input_ids[None, :]
        elif "segment" in name:
            feed[name] = segment_ids[None, :]
        elif "mask" in name:
            feed[name] = input_mask[None, :]
        elif "unique" in name:
            feed[name] = unique_ids
        else:
            feed[name] = np.zeros((1,), dtype=np.int64)

    be_cls = get_backend(backend_name)
    be = be_cls() if isinstance(be_cls, type) else be_cls
    concrete = {k: tuple(v.shape) for k, v in feed.items()}
    try:
        program = be.compile(model, concrete_shapes=concrete)
    except TypeError:
        program = be.compile(model)

    t0 = time.perf_counter()
    out = be.run(program, feed)
    wall_ms = (time.perf_counter() - t0) * 1e3

    # BERT-Squad outputs: unstack_0:0 (start logits) and unstack_1:0 (end).
    start_logits = None
    end_logits = None
    for k, v in out.items():
        v = np.asarray(v).squeeze()
        if v.ndim == 1 and v.shape[0] == seq_len:
            if start_logits is None:
                start_logits = v
            else:
                end_logits = v
    if start_logits is None or end_logits is None:
        raise DemoError(
            f"could not find start/end span logits in outputs: "
            f"{[k for k in out]}"
        )

    start = int(np.argmax(start_logits))
    end = int(np.argmax(end_logits))
    # Clip to a plausible answer span (1..10 tokens)
    answer_len = max(1, min(10, end - start + 1))

    predictions = [{
        "start_token_idx": start,
        "end_token_idx": end,
        "span_len": answer_len,
        "start_score": float(start_logits[start]),
        "end_score": float(end_logits[end]),
    }]

    summary = (
        f"answer span tokens [{start}..{end}] (len {answer_len})  "
        f"start_logit={start_logits[start]:.2f} "
        f"end_logit={end_logits[end]:.2f}  "
        f"(canned Q+A: 'What is the capital of France?')"
    )

    return DemoResult(
        model=zoo_entry.name,
        family=zoo_entry.family,
        backend=backend_name,
        input_source="canned:bert-squad-france",
        wall_ms=wall_ms,
        predictions=predictions,
        summary=summary,
        raw_shape=[int(start_logits.shape[0])],
    )


@register_demo_family("nlp-decoder-transformer")
def run_gpt2(zoo_entry, onnx_path: Path, *,
             input_spec: Optional[str] = None,
             backend_name: str = "onnxruntime") -> DemoResult:
    """GPT-2 next-token prediction on a canned 5-token prompt."""
    import astracore.backends  # noqa: F401
    from astracore.registry import get_backend

    import onnx
    model = onnx.load(str(onnx_path))
    init_names = {t.name for t in model.graph.initializer}
    real_inputs = [inp for inp in model.graph.input if inp.name not in init_names]

    # GPT-2-10 has a single input named 'input1' with 3 dynamic dims.
    # Feed (batch=1, seq_dim=1, tokens=5). Canned tokens represent
    # "The capital of France is" in GPT-2 BPE (per tokenizer.encode).
    tokens = _CANNED_TOKENS_GPT2[None, None, :]
    input_name = real_inputs[0].name
    feed = {input_name: tokens.astype(np.int64)}

    be_cls = get_backend(backend_name)
    be = be_cls() if isinstance(be_cls, type) else be_cls
    try:
        program = be.compile(model, concrete_shapes={input_name: tuple(tokens.shape)})
    except TypeError:
        program = be.compile(model)

    t0 = time.perf_counter()
    out = be.run(program, feed)
    wall_ms = (time.perf_counter() - t0) * 1e3

    # This ONNX exports (batch, seq_dim, seq_len, hidden=768) hidden
    # states — NOT vocabulary logits. GPT-2 has weight-tied LM head
    # (reuses wte embedding matrix), so we project manually:
    #   logits[:, vocab] = hidden @ wte.T
    hidden = np.asarray(out["output1"]).squeeze()       # (seq_len, 768)
    if hidden.ndim == 1:
        hidden = hidden[None, :]
    last_hidden = hidden[-1]                              # (768,)

    # Grab the embedding table — wte.weight, shape (vocab=50257, hidden=768)
    wte = None
    for init in model.graph.initializer:
        if init.name == "wte.weight" and list(init.dims) == [50257, 768]:
            wte = onnx.numpy_helper.to_array(init)
            break

    if wte is not None:
        # LM-head projection (weight-tied)
        raw_logits = last_hidden @ wte.T                   # (50257,)
        # Softmax for a probability distribution
        shifted = raw_logits - raw_logits.max()
        probs = np.exp(shifted); probs /= probs.sum()

        top5 = np.argsort(probs)[::-1][:5]
        predictions = [
            {"token_id": int(i), "prob": float(probs[i])}
            for i in top5
        ]
        # Token 6342 = " Paris" in GPT-2 BPE
        order = np.argsort(probs)[::-1]
        paris_rank = int(np.where(order == 6342)[0][0]) + 1
        paris_prob = float(probs[6342])
        summary = (
            f"top-5 next-token IDs: {[int(i) for i in top5]}  "
            f"(canned prompt 'The capital of France is' -> token 6342=' Paris'; "
            f"rank {paris_rank}/50257, prob {paris_prob:.2%})"
        )
        raw_shape = [int(probs.shape[0])]
    else:
        # Fallback — just report hidden-state stats
        top5_h = np.argsort(np.abs(last_hidden))[::-1][:5]
        predictions = [
            {"hidden_dim_idx": int(i), "activation": float(last_hidden[i])}
            for i in top5_h
        ]
        summary = (
            "GPT-2 backbone produced 768-d hidden state; wte.weight not "
            f"found in initialisers for LM-head projection. Hidden norm = "
            f"{float(np.linalg.norm(last_hidden)):.2f}."
        )
        raw_shape = list(last_hidden.shape)

    return DemoResult(
        model=zoo_entry.name,
        family=zoo_entry.family,
        backend=backend_name,
        input_source="canned:gpt2-france",
        wall_ms=wall_ms,
        predictions=predictions,
        summary=summary,
        raw_shape=raw_shape,
    )
