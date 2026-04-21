"""Execute a loaded ``AstracoreConfig`` end-to-end.

``astracore configure --apply X.yaml`` calls into :func:`apply_config`
here. Unlike ``--validate``, which is a schema check, ``--apply``
actually runs the pipeline the YAML describes:

* instantiate the configured dataset connector + replay one scene
  through the perception pipeline;
* benchmark every model declared in ``models:`` whose ONNX file
  exists on disk (precision / sparsity taken from the YAML);
* if ``multistream.enabled``, run the scaling sweep per model;
* surface the declared ``safety_policies`` on the report so the
  customer sees them alongside the numbers.

Output: a single combined JSON + Markdown report under
``reports/apply_<cfg-name>/`` (or a caller-chosen directory).

Design notes
------------
* Orchestration only — no new numeric work. Reuses the existing
  :mod:`astracore.benchmark`, :mod:`astracore.multistream`, and
  :mod:`astracore.dataset.replay` modules so the --apply numbers are
  the same you get from the individual sub-commands.
* Missing models and missing dataset presets degrade gracefully:
  each failure is captured in the report, the run continues so one
  bad row doesn't sink the whole sweep.
* No backend is instantiated for safety policies — we record them
  verbatim. Runtime enforcement is an OEM integration concern (the
  ``examples/`` directory has a proof-of-concept rule).
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional
import json
import re
import time

from astracore.config import AstracoreConfig


# ---------------------------------------------------------------------------
# Report schema
# ---------------------------------------------------------------------------

@dataclass
class ModelRow:
    id: str
    path: str
    family: str
    precision: str
    sparsity: str
    input_sensor: Optional[str] = None
    bench_ok: bool = False
    bench_error: str = ""
    wall_ms_per_inference: float = 0.0
    gmacs: float = 0.0
    delivered_tops: float = 0.0
    active_providers: List[str] = field(default_factory=list)
    multistream_slices: List[Dict[str, Any]] = field(default_factory=list)


@dataclass
class ReplayRow:
    scene_id: str
    scene_name: str
    n_samples: int
    wall_s_total: float
    summary: Dict[str, float]


@dataclass
class ApplyReport:
    config_name: str
    config_description: str
    backend: str
    dataset_connector: str
    dataset_preset: Optional[str]
    sensor_counts: Dict[str, int]
    replay: Optional[ReplayRow]
    models: List[ModelRow]
    safety_policies: List[Dict[str, Any]]
    multistream_enabled: bool
    streams_per_model: int
    wall_s_total: float
    notes: List[str] = field(default_factory=list)

    def as_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _slug(s: str) -> str:
    """Filesystem-safe form of a human name. Empty -> 'config'."""
    s = (s or "config").strip().lower()
    s = re.sub(r"[^a-z0-9_-]+", "_", s)
    return s.strip("_") or "config"


def _shape_csv(shape) -> str:
    return ",".join(str(int(d)) for d in shape)


def _sensor_counts(cfg: AstracoreConfig) -> Dict[str, int]:
    s = cfg.sensors
    counts = {
        "cameras":     len(s.cameras),
        "lidars":      len(s.lidars),
        "radars":      len(s.radars),
        "ultrasonics": len(s.ultrasonics),
        "microphones": len(s.microphones),
        "thermals":    len(s.thermals),
        "events":      len(s.events),
        "depths":      len(s.depths),
        "can":         len(s.can),
        "gnss":        1 if s.gnss else 0,
        "imu":         1 if s.imu else 0,
    }
    return counts


def _build_dataset(cfg: AstracoreConfig):
    """Return (dataset, notes).

    For ``--apply`` we clamp heavy presets ('vlp64', 'robotaxi') that
    would cost multiple GB of RAM per scene — the replay is a
    smoke-test, not a quality measurement. The user gets the full
    dataset via ``astracore replay --preset robotaxi`` explicitly.
    """
    notes: List[str] = []
    ds_cfg = cfg.dataset
    if ds_cfg.connector == "synthetic":
        from astracore.dataset import SyntheticDataset, PRESETS
        preset_name = ds_cfg.preset or "tiny"
        if preset_name not in PRESETS:
            notes.append(
                f"unknown synthetic preset {preset_name!r}; falling back to 'tiny'"
            )
            preset_name = "tiny"
        if preset_name in {"vlp64", "robotaxi"}:
            notes.append(
                f"replay preset downsized from {preset_name!r} to "
                f"'extended-sensors' to keep --apply tractable "
                f"(use `astracore replay --preset {preset_name}` for the full rig)"
            )
            preset_name = "extended-sensors"
        return SyntheticDataset(preset_name=preset_name), notes
    if ds_cfg.connector == "nuscenes":
        if not ds_cfg.dataroot:
            notes.append("nuscenes connector requires dataset.dataroot; skipping replay")
            return None, notes
        try:
            from astracore.dataset import NuScenesDataset
            return NuScenesDataset(dataroot=ds_cfg.dataroot,
                                   version=ds_cfg.version or "v1.0-mini"), notes
        except Exception as exc:
            notes.append(f"nuscenes connector unavailable: {exc!r}")
            return None, notes
    notes.append(f"unknown dataset connector {ds_cfg.connector!r}; skipping replay")
    return None, notes


def _replay_first_scene(dataset, backend_name: str,
                        max_samples: int = 10) -> Optional[ReplayRow]:
    """Replay a clipped prefix of the first scene.

    ``max_samples`` guards against the ``robotaxi`` preset (150 samples
    × 4K × 8 cameras) — we don't need all of it to prove the pipeline
    flows end-to-end. Using the full scene is available via
    ``astracore replay`` when the user actually wants it.
    """
    from astracore.dataset import Scene, replay_scene
    scenes = dataset.list_scenes()
    if not scenes:
        return None
    scene_id = scenes[0]
    scene = dataset.get_scene(scene_id)
    if max_samples and len(scene) > max_samples:
        scene = Scene(
            scene_id=scene.scene_id,
            name=scene.name + f" [clipped first {max_samples}]",
            description=scene.description,
            samples=list(scene.samples[:max_samples]),
        )
    result = replay_scene(scene, backend_name=backend_name)
    return ReplayRow(
        scene_id=result.scene_id,
        scene_name=result.scene_name,
        n_samples=result.n_samples,
        wall_s_total=round(result.wall_s_total, 4),
        summary={k: round(v, 3) for k, v in result.summary().items()},
    )


def _bench_model(m, path: Path, *, backend: str, n_iter: int,
                 backend_options: Optional[Dict[str, Any]] = None) -> ModelRow:
    """Run benchmark_model for one config entry, return a ModelRow."""
    from astracore.benchmark import benchmark_model

    row = ModelRow(
        id=m.id, path=str(path), family=m.family,
        precision=m.precision, sparsity=m.sparsity,
        input_sensor=m.input_sensor,
    )
    try:
        rep = benchmark_model(
            path, backend=backend,
            backend_options=backend_options,
            precision=m.precision if m.precision in {"INT8", "INT4", "INT2"} else "INT8",
            sparsity=m.sparsity if m.sparsity in {"dense", "2:4", "4:1", "8:1"} else "dense",
            n_iter=n_iter, warmup=1,
        )
        row.bench_ok = True
        row.wall_ms_per_inference = round(rep.wall_ms_per_inference, 4)
        row.gmacs = round(rep.mac_ops_total / 1e9, 4)
        row.delivered_tops = round(rep.delivered_tops, 4)
        row.active_providers = list(rep.extra.get("active_providers") or [])
    except Exception as exc:
        row.bench_ok = False
        row.bench_error = repr(exc)
    return row


def _multistream_model(m, path: Path, *, backend: str,
                       streams: List[int], duration_s: float,
                       warmup_s: float,
                       backend_options: Optional[Dict[str, Any]] = None,
                       ) -> List[Dict[str, Any]]:
    from astracore.multistream import run_multistream
    rep = run_multistream(
        path, backend=backend,
        backend_options=backend_options,
        n_streams_list=tuple(streams),
        duration_s=duration_s, warmup_s=warmup_s,
    )
    return [
        {
            "n_streams": s.n_streams,
            "throughput_ips": round(s.throughput_ips, 2),
            "aggregate_tops": round(s.aggregate_tops, 4),
            "p50_ms": round(s.p50_latency_ms, 3),
            "p99_ms": round(s.p99_latency_ms, 3),
            "scale_vs_1x": round(s.scaling_vs_single, 3),
        }
        for s in rep.slices
    ]


# ---------------------------------------------------------------------------
# Main entrypoint
# ---------------------------------------------------------------------------

def apply_config(
    cfg: AstracoreConfig,
    *,
    out_dir: Optional[Path] = None,
    bench_iter: int = 3,
    multistream_duration_s: float = 2.0,
    multistream_warmup_s: float = 0.5,
    multistream_streams: Optional[List[int]] = None,
    skip_replay: bool = False,
    skip_bench: bool = False,
    skip_multistream: bool = False,
    allow_partial: bool = True,
) -> ApplyReport:
    """Run the full pipeline described by ``cfg`` and return an ApplyReport.

    Writes ``report.json`` + ``report.md`` under ``out_dir`` (defaults to
    ``reports/apply_<slug>/``).
    """
    t_start = time.perf_counter()
    notes: List[str] = []

    backend_name = cfg.backend.name or "onnxruntime"
    backend_options = dict(cfg.backend.options or {})

    # ---- Replay -------------------------------------------------------
    replay_row: Optional[ReplayRow] = None
    if not skip_replay:
        dataset, ds_notes = _build_dataset(cfg)
        notes.extend(ds_notes)
        if dataset is not None:
            try:
                replay_row = _replay_first_scene(dataset, backend_name)
            except Exception as exc:
                notes.append(f"replay failed: {exc!r}")
                if not allow_partial:
                    raise

    # ---- Models (bench + optional multistream) ------------------------
    model_rows: List[ModelRow] = []
    streams = multistream_streams or (
        [1, 2, 4, 8] if cfg.multistream.streams_per_model >= 8
        else [1, 2, max(2, cfg.multistream.streams_per_model)]
    )

    for m in cfg.models:
        p = Path(m.path)
        if not p.exists():
            row = ModelRow(
                id=m.id, path=str(p), family=m.family,
                precision=m.precision, sparsity=m.sparsity,
                input_sensor=m.input_sensor,
                bench_ok=False,
                bench_error=f"model file missing: {p}",
            )
            model_rows.append(row)
            continue

        if skip_bench:
            row = ModelRow(
                id=m.id, path=str(p), family=m.family,
                precision=m.precision, sparsity=m.sparsity,
                input_sensor=m.input_sensor,
                bench_ok=False,
                bench_error="bench skipped",
            )
        else:
            row = _bench_model(m, p, backend=backend_name, n_iter=bench_iter,
                               backend_options=backend_options)

        if (cfg.multistream.enabled and not skip_multistream
                and row.bench_ok):
            try:
                row.multistream_slices = _multistream_model(
                    m, p, backend=backend_name,
                    streams=streams,
                    duration_s=multistream_duration_s,
                    warmup_s=multistream_warmup_s,
                    backend_options=backend_options,
                )
            except Exception as exc:
                notes.append(f"multistream failed for {m.id}: {exc!r}")
                if not allow_partial:
                    raise
        model_rows.append(row)

    report = ApplyReport(
        config_name=cfg.name or "(unnamed)",
        config_description=cfg.description or "",
        backend=backend_name,
        dataset_connector=cfg.dataset.connector,
        dataset_preset=cfg.dataset.preset,
        sensor_counts=_sensor_counts(cfg),
        replay=replay_row,
        models=model_rows,
        safety_policies=[
            {"type": p.type, "value": p.value, "description": p.description}
            for p in cfg.safety_policies
        ],
        multistream_enabled=cfg.multistream.enabled,
        streams_per_model=cfg.multistream.streams_per_model,
        wall_s_total=round(time.perf_counter() - t_start, 3),
        notes=notes,
    )

    out_dir = Path(out_dir) if out_dir else Path("reports") / f"apply_{_slug(cfg.name)}"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(
        json.dumps(report.as_dict(), indent=2, default=str)
    )
    (out_dir / "report.md").write_text(render_markdown(report))
    return report


# ---------------------------------------------------------------------------
# Markdown renderer
# ---------------------------------------------------------------------------

def render_markdown(r: ApplyReport) -> str:
    lines: List[str] = []
    lines.append(f"# AstraCore --apply report — {r.config_name}")
    lines.append("")
    if r.config_description:
        lines.append(r.config_description.strip())
        lines.append("")
    # Active EPs — only surface when we actually ran something on ORT.
    eps_seen: List[str] = []
    for m in r.models:
        for ep in m.active_providers:
            if ep not in eps_seen:
                eps_seen.append(ep)
    ep_bit = f"  •  EPs: `{', '.join(eps_seen)}`" if eps_seen else ""
    lines.append(f"Backend: `{r.backend}`  •  Dataset: "
                 f"`{r.dataset_connector}`"
                 + (f" / `{r.dataset_preset}`" if r.dataset_preset else "")
                 + ep_bit
                 + f"  •  Wall: {r.wall_s_total:.1f}s")
    lines.append("")

    # Sensor counts
    lines.append("## Sensors")
    lines.append("")
    lines.append("| Kind | Count |")
    lines.append("|---|---:|")
    for kind, n in r.sensor_counts.items():
        if n:
            lines.append(f"| {kind} | {n} |")
    lines.append("")

    # Replay
    lines.append("## Replay")
    lines.append("")
    if r.replay is None:
        lines.append("_skipped or unavailable_")
    else:
        lines.append(f"Scene `{r.replay.scene_id}` "
                     f"({r.replay.scene_name!r}), {r.replay.n_samples} samples, "
                     f"{r.replay.wall_s_total:.2f}s wall.")
        lines.append("")
        lines.append("| Metric | Mean |")
        lines.append("|---|---:|")
        for k, v in r.replay.summary.items():
            lines.append(f"| {k} | {v:.2f} |")
    lines.append("")

    # Models
    lines.append("## Models")
    lines.append("")
    if not r.models:
        lines.append("_no models declared in config_")
    else:
        lines.append("| id | family | precision | sparsity | "
                     "ms / inf | GMACs | TOPS | notes |")
        lines.append("|---|---|---|---|---:|---:|---:|---|")
        for m in r.models:
            if m.bench_ok:
                lines.append(
                    f"| {m.id} | {m.family} | {m.precision} | {m.sparsity} | "
                    f"{m.wall_ms_per_inference:.2f} | {m.gmacs:.2f} | "
                    f"{m.delivered_tops:.3f} | "
                    f"{'input=' + m.input_sensor if m.input_sensor else ''} |"
                )
            else:
                lines.append(
                    f"| {m.id} | {m.family} | {m.precision} | {m.sparsity} | "
                    f"— | — | — | **FAIL**: `{m.bench_error}` |"
                )
    lines.append("")

    # Multistream
    if r.multistream_enabled:
        lines.append(f"## Multi-stream scaling (streams_per_model={r.streams_per_model})")
        lines.append("")
        for m in r.models:
            if not m.multistream_slices:
                continue
            lines.append(f"### {m.id}")
            lines.append("")
            lines.append("| Streams | IPS | TOPS (agg) | p50 ms | p99 ms | scale |")
            lines.append("|---:|---:|---:|---:|---:|---:|")
            for s in m.multistream_slices:
                lines.append(
                    f"| {s['n_streams']} | {s['throughput_ips']:.1f} | "
                    f"{s['aggregate_tops']:.3f} | {s['p50_ms']:.2f} | "
                    f"{s['p99_ms']:.2f} | {s['scale_vs_1x']:.2f}× |"
                )
            lines.append("")

    # Safety policies
    if r.safety_policies:
        lines.append("## Safety policies (declared)")
        lines.append("")
        lines.append("| Type | Value | Description |")
        lines.append("|---|---|---|")
        for p in r.safety_policies:
            desc = (p.get("description") or "").replace("|", "/")
            lines.append(f"| {p['type']} | `{p['value']}` | {desc} |")
        lines.append("")

    # Notes
    if r.notes:
        lines.append("## Notes")
        lines.append("")
        for n in r.notes:
            lines.append(f"- {n}")
        lines.append("")

    return "\n".join(lines)
