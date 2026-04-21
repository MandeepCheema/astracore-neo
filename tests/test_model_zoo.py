"""Smoke tests for the model-zoo registry + benchmark matrix.

Network-independent — tests the metadata contract and the fallback
path that reports cleanly when a zoo entry is not on disk.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from astracore import zoo


def test_zoo_has_expected_entries():
    names = {m.name for m in zoo.all_models()}
    # 6 vision + 2 transformers = 8 curated entries.
    for expected in ("squeezenet-1.1", "mobilenetv2-7", "resnet50-v2-7",
                     "efficientnet-lite4-11", "shufflenet-v2-10", "yolov8n",
                     "bert-squad-10", "gpt-2-10"):
        assert expected in names, f"missing zoo entry {expected!r}"


def test_zoo_covers_transformer_architectures():
    """GPT-2 (decoder, LLaMA-family) + BERT-Squad (encoder) both present."""
    families = {m.family for m in zoo.all_models()}
    assert "nlp-encoder-transformer" in families
    assert "nlp-decoder-transformer" in families


def test_zoo_entries_are_frozen_dataclasses():
    """Immutable metadata — consumers can safely cache."""
    m = zoo.get("squeezenet-1.1")
    with pytest.raises(Exception):
        m.display_name = "hack"  # type: ignore[misc]


def test_zoo_unknown_name_raises():
    with pytest.raises(KeyError):
        zoo.get("not-a-real-model")


def test_zoo_local_paths_contains_every_entry():
    paths = zoo.local_paths()
    for m in zoo.all_models():
        assert m.name in paths


def test_zoo_available_matches_disk():
    """The `available` helper returns exactly the entries whose file exists."""
    paths = zoo.local_paths()
    avail = {m.name for m in zoo.available()}
    on_disk = {n for n, p in paths.items() if Path(p).exists()}
    assert avail == on_disk


def test_zoo_manifest_serialisable():
    """Manifest dicts must be JSON-serialisable (used by fetch script)."""
    import json
    data = zoo.as_manifest_dicts()
    # `local_path` is a computed property so not in asdict; input_shape is
    # a tuple which json rejects — convert to list first via json default.
    def _default(o):
        if isinstance(o, tuple):
            return list(o)
        if isinstance(o, Path):
            return str(o)
        raise TypeError(f"not json-serialisable: {type(o).__name__}")
    # Any tuples in the dataclass still need encoding via default hook.
    json.dumps(data, default=_default)


# -- Benchmark matrix end-to-end (skipped if files absent) -----------------

@pytest.mark.parametrize("name", ["squeezenet-1.1", "shufflenet-v2-10"])
def test_benchmark_one_zoo_model(name):
    """Benchmark a zoo model end-to-end via ORT if downloaded."""
    m = zoo.get(name)
    path = zoo.local_paths()[m.name]
    if not path.exists():
        pytest.skip(f"{m.name} not downloaded; run scripts/fetch_model_zoo.py")

    from astracore.benchmark import benchmark_model
    rep = benchmark_model(path, backend="onnxruntime",
                          input_shape=",".join(str(d) for d in m.input_shape),
                          n_iter=1, warmup=0)
    assert rep.wall_ms_per_inference > 0
    assert rep.mac_ops_total > 0, \
        "MAC estimator returned 0 — shape inference path regressed"
    assert rep.model == m.name or path.name == rep.model
