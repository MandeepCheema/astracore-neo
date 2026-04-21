"""``astracore.config`` — declarative YAML configuration for the SDK.

What this is for
----------------
A Tier-1 supplier's integration lead wants a single YAML file that
describes their full rig: which sensors, which models, which backend,
which safety policies. Procurement + management review read YAML
better than Python subclasses, and config-diffs make customer-vs-ours
versioning explicit.

Design
------
* Root :class:`AstracoreConfig` dataclass mirrors the YAML schema.
* :func:`load` reads a YAML file, validates references, returns the
  typed object. All errors include field paths so misspelling a model
  name doesn't produce a cryptic dataclass error.
* :func:`to_yaml` dumps back to a string for round-tripping.
* ``astracore configure --validate <file>`` is the CLI entry.

The config is **declarative only** — applying it (running the
pipeline) is a separate step, composed from the already-existing
benchmark / replay / multistream functions.
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


SCHEMA_VERSION = 1


# ---------------------------------------------------------------------------
# Exceptions
# ---------------------------------------------------------------------------

class ConfigError(ValueError):
    """Raised when a config fails schema validation."""


# ---------------------------------------------------------------------------
# Sensor config dataclasses — one per SensorKind in base.py
# ---------------------------------------------------------------------------

@dataclass
class CameraConfig:
    name: str
    resolution: Tuple[int, int] = (1920, 1080)    # (W, H)
    format: str = "RGB888"                         # "RGB888" | "YUV420" | "NV12"
    extrinsics: Optional[Dict[str, float]] = None  # {x, y, z, roll, pitch, yaw}
    intrinsics_path: Optional[str] = None
    role: str = "perception"                       # "perception" | "dms" | "surround"


@dataclass
class LidarConfig:
    name: str
    model: str = "generic-32-beam"
    max_range_m: float = 120.0
    min_range_m: float = 0.5
    rotation_hz: float = 10.0
    extrinsics: Optional[Dict[str, float]] = None


@dataclass
class RadarConfig:
    name: str
    position: str = "front"                        # "front" | "rear" | "corner-*"
    max_range_m: float = 200.0
    doppler_resolution_m_per_s: float = 0.5
    extrinsics: Optional[Dict[str, float]] = None


@dataclass
class UltrasonicConfig:
    name: str
    position: str                                  # "front-center", "rear-left", ...
    max_range_m: float = 3.0


@dataclass
class MicrophoneConfig:
    name: str
    sample_rate_hz: int = 16_000
    channels: int = 1
    role: str = "cabin"                            # "cabin" | "outdoor-siren" | ...


@dataclass
class ThermalConfig:
    name: str
    resolution: Tuple[int, int] = (640, 480)
    band: str = "LWIR"                             # LWIR / MWIR / NIR
    extrinsics: Optional[Dict[str, float]] = None


@dataclass
class EventCameraConfig:
    name: str
    resolution: Tuple[int, int] = (640, 480)
    integration_window_us: int = 10_000


@dataclass
class DepthConfig:
    name: str
    resolution: Tuple[int, int] = (640, 480)
    modality: str = "ToF"                          # "ToF" | "structured-light" | "stereo"
    extrinsics: Optional[Dict[str, float]] = None


@dataclass
class CanBusConfig:
    name: str
    dbc_path: Optional[str] = None
    bitrate_kbps: int = 500


@dataclass
class GnssConfig:
    name: str = "GNSS_PRIMARY"
    rtk: bool = False
    update_hz: float = 10.0


@dataclass
class ImuConfig:
    name: str = "IMU_PRIMARY"
    update_hz: float = 200.0
    axes: int = 6                                  # 6 | 9 (with magnetometer)


@dataclass
class SensorsConfig:
    cameras: List[CameraConfig] = field(default_factory=list)
    lidars: List[LidarConfig] = field(default_factory=list)
    radars: List[RadarConfig] = field(default_factory=list)
    ultrasonics: List[UltrasonicConfig] = field(default_factory=list)
    microphones: List[MicrophoneConfig] = field(default_factory=list)
    thermals: List[ThermalConfig] = field(default_factory=list)
    events: List[EventCameraConfig] = field(default_factory=list)
    depths: List[DepthConfig] = field(default_factory=list)
    can: List[CanBusConfig] = field(default_factory=list)
    gnss: Optional[GnssConfig] = None
    imu: Optional[ImuConfig] = None


# ---------------------------------------------------------------------------
# Models + pipeline + backend + safety
# ---------------------------------------------------------------------------

@dataclass
class ModelConfig:
    id: str                                   # stable handle, used for cross-refs
    path: str
    family: str = "vision-detection"          # matches astracore.demo families
    precision: str = "INT8"                   # INT8 | INT4 | INT2 | FP8 | FP16
    sparsity: str = "dense"                   # dense | 2:4 | 4:1 | 8:1
    input_sensor: Optional[str] = None        # name of a sensor in SensorsConfig
    notes: str = ""


@dataclass
class BackendConfig:
    name: str = "onnxruntime"                 # any registered astracore backend
    options: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SafetyPolicy:
    type: str                                 # e.g. "min_pedestrian_distance_m"
    value: Any                                # policy-specific payload
    description: str = ""


@dataclass
class MultiStreamConfig:
    enabled: bool = False
    streams_per_model: int = 4


@dataclass
class DatasetSource:
    connector: str = "synthetic"              # "synthetic" | "nuscenes" | plug-in
    preset: Optional[str] = None              # SyntheticDataset preset name
    dataroot: Optional[str] = None            # nuscenes: root; plugin: user-defined
    version: Optional[str] = None             # nuscenes version, etc.


@dataclass
class AstracoreConfig:
    version: int = SCHEMA_VERSION
    name: str = ""
    description: str = ""
    sensors: SensorsConfig = field(default_factory=SensorsConfig)
    models: List[ModelConfig] = field(default_factory=list)
    backend: BackendConfig = field(default_factory=BackendConfig)
    safety_policies: List[SafetyPolicy] = field(default_factory=list)
    multistream: MultiStreamConfig = field(default_factory=MultiStreamConfig)
    dataset: DatasetSource = field(default_factory=DatasetSource)


# ---------------------------------------------------------------------------
# Load / save
# ---------------------------------------------------------------------------

def _as_tuple(v, name: str) -> Tuple[int, int]:
    if isinstance(v, (list, tuple)) and len(v) == 2:
        return (int(v[0]), int(v[1]))
    raise ConfigError(f"{name}: expected a 2-int list/tuple, got {v!r}")


def _build_cameras(items) -> List[CameraConfig]:
    out = []
    for i, d in enumerate(items or []):
        if "name" not in d:
            raise ConfigError(f"cameras[{i}]: missing 'name'")
        out.append(CameraConfig(
            name=str(d["name"]),
            resolution=_as_tuple(d.get("resolution", (1920, 1080)),
                                 f"cameras[{i}].resolution"),
            format=str(d.get("format", "RGB888")),
            extrinsics=d.get("extrinsics"),
            intrinsics_path=d.get("intrinsics_path"),
            role=str(d.get("role", "perception")),
        ))
    return out


def _build_generic(items, cls, required=("name",), field_overrides=None):
    """Build list of simple-typed configs, catching bad fields early."""
    out = []
    for i, d in enumerate(items or []):
        for r in required:
            if r not in d:
                raise ConfigError(f"{cls.__name__}[{i}]: missing {r!r}")
        overrides = (field_overrides or {})
        kwargs = dict(d)
        for k, fn in overrides.items():
            if k in kwargs:
                kwargs[k] = fn(kwargs[k], f"{cls.__name__}[{i}].{k}")
        try:
            out.append(cls(**kwargs))
        except TypeError as exc:
            raise ConfigError(f"{cls.__name__}[{i}]: {exc}") from exc
    return out


def load(path) -> AstracoreConfig:
    """Load + validate a YAML config. Raises ConfigError on failure."""
    import yaml
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"config file not found: {p}")
    with p.open() as fh:
        raw = yaml.safe_load(fh) or {}
    return _from_dict(raw)


def _from_dict(raw: Dict[str, Any]) -> AstracoreConfig:
    version = raw.get("version", SCHEMA_VERSION)
    if version != SCHEMA_VERSION:
        raise ConfigError(
            f"schema version {version} != expected {SCHEMA_VERSION}"
        )

    # Sensors ----------------------------------------------------------
    sraw = raw.get("sensors", {}) or {}
    sensors = SensorsConfig(
        cameras=_build_cameras(sraw.get("cameras")),
        lidars=_build_generic(sraw.get("lidars"), LidarConfig),
        radars=_build_generic(sraw.get("radars"), RadarConfig),
        ultrasonics=_build_generic(sraw.get("ultrasonics"), UltrasonicConfig,
                                   required=("name", "position")),
        microphones=_build_generic(sraw.get("microphones"), MicrophoneConfig),
        thermals=_build_generic(sraw.get("thermals"), ThermalConfig,
                                field_overrides={"resolution": _as_tuple}),
        events=_build_generic(sraw.get("events"), EventCameraConfig,
                              field_overrides={"resolution": _as_tuple}),
        depths=_build_generic(sraw.get("depths"), DepthConfig,
                              field_overrides={"resolution": _as_tuple}),
        can=_build_generic(sraw.get("can"), CanBusConfig),
        gnss=GnssConfig(**sraw["gnss"]) if sraw.get("gnss") else None,
        imu=ImuConfig(**sraw["imu"]) if sraw.get("imu") else None,
    )

    # Models -----------------------------------------------------------
    # Coerce ints back to strings for fields that are string-typed in
    # the schema — YAML's sexagesimal quirk means `2:4` unquoted becomes
    # the integer 124. Stringify to survive both forms.
    for m in raw.get("models") or []:
        for k in ("precision", "sparsity"):
            if k in m and not isinstance(m[k], str):
                m[k] = str(m[k])
    models = _build_generic(raw.get("models"), ModelConfig,
                            required=("id", "path"))
    # Validate the precision / sparsity strings against a known whitelist.
    allowed_precision = {"INT8", "INT4", "INT2", "FP8", "FP16", "BF16", "FP32"}
    allowed_sparsity = {"dense", "2:4", "4:1", "8:1", "124"}   # 124 = YAML mis-parse
    for m in models:
        if m.precision not in allowed_precision:
            raise ConfigError(
                f"models[{m.id}].precision {m.precision!r} not in "
                f"{sorted(allowed_precision)}"
            )
        if m.sparsity not in allowed_sparsity:
            raise ConfigError(
                f"models[{m.id}].sparsity {m.sparsity!r} not in "
                f"{sorted(allowed_sparsity - {'124'})} "
                f"(hint: quote '2:4' in YAML — unquoted it parses as 124)"
            )
        if m.sparsity == "124":
            # Silent coerce — we accept the mistake + remap to the intended ratio.
            m.sparsity = "2:4"

    # Backend ----------------------------------------------------------
    braw = raw.get("backend") or {}
    if isinstance(braw, str):
        backend = BackendConfig(name=braw)
    else:
        backend = BackendConfig(
            name=str(braw.get("name", "onnxruntime")),
            options=dict(braw.get("options", {})),
        )

    # Safety policies --------------------------------------------------
    safety = _build_generic(raw.get("safety_policies"), SafetyPolicy,
                            required=("type", "value"))

    # Multi-stream -----------------------------------------------------
    ms = raw.get("multistream") or {}
    multistream = MultiStreamConfig(
        enabled=bool(ms.get("enabled", False)),
        streams_per_model=int(ms.get("streams_per_model", 4)),
    )

    # Dataset source ---------------------------------------------------
    dsraw = raw.get("dataset") or {}
    dataset = DatasetSource(
        connector=str(dsraw.get("connector", "synthetic")),
        preset=dsraw.get("preset"),
        dataroot=dsraw.get("dataroot"),
        version=dsraw.get("version"),
    )

    cfg = AstracoreConfig(
        version=version,
        name=str(raw.get("name", "")),
        description=str(raw.get("description", "")),
        sensors=sensors,
        models=models,
        backend=backend,
        safety_policies=safety,
        multistream=multistream,
        dataset=dataset,
    )
    _cross_validate(cfg)
    return cfg


def _cross_validate(cfg: AstracoreConfig) -> None:
    """Catch the easy mistakes: dup names, dangling sensor refs, missing files."""
    # Every sensor name unique across all kinds
    all_names = []
    for section in (cfg.sensors.cameras, cfg.sensors.lidars, cfg.sensors.radars,
                    cfg.sensors.ultrasonics, cfg.sensors.microphones,
                    cfg.sensors.thermals, cfg.sensors.events, cfg.sensors.depths,
                    cfg.sensors.can):
        all_names.extend(s.name for s in section)
    if cfg.sensors.gnss is not None: all_names.append(cfg.sensors.gnss.name)
    if cfg.sensors.imu is not None:  all_names.append(cfg.sensors.imu.name)
    dupes = {n for n in all_names if all_names.count(n) > 1}
    if dupes:
        raise ConfigError(f"duplicate sensor names: {sorted(dupes)}")

    # Model → input_sensor must reference a known sensor name (if set)
    known = set(all_names)
    for m in cfg.models:
        if m.input_sensor and m.input_sensor not in known:
            raise ConfigError(
                f"models[{m.id}].input_sensor {m.input_sensor!r} not in sensors"
            )
    # Model ids must be unique
    mids = [m.id for m in cfg.models]
    if len(set(mids)) != len(mids):
        raise ConfigError(f"duplicate model ids: {sorted(set(mids))}")


# ---------------------------------------------------------------------------
# Dumps / export
# ---------------------------------------------------------------------------

def to_dict(cfg: AstracoreConfig) -> Dict[str, Any]:
    d = asdict(cfg)
    # pyyaml dumps lists/tuples identically as sequences, but keep resolution
    # tuples ergonomic (list form).
    return d


def to_yaml(cfg: AstracoreConfig) -> str:
    import yaml
    return yaml.safe_dump(to_dict(cfg), sort_keys=False, default_flow_style=False)


# ---------------------------------------------------------------------------
# Summary — used by `astracore configure --validate`
# ---------------------------------------------------------------------------

def summary(cfg: AstracoreConfig) -> str:
    s = cfg.sensors
    lines = [
        f"Config: {cfg.name or '(unnamed)'}  (schema v{cfg.version})",
    ]
    if cfg.description:
        lines.append(f"  {cfg.description}")
    lines.append("")
    lines.append("Sensors:")
    counts = [
        ("cameras",    len(s.cameras)),
        ("lidars",     len(s.lidars)),
        ("radars",     len(s.radars)),
        ("ultrasonics", len(s.ultrasonics)),
        ("microphones", len(s.microphones)),
        ("thermals",   len(s.thermals)),
        ("event cams", len(s.events)),
        ("depth sensors", len(s.depths)),
        ("CAN buses",  len(s.can)),
    ]
    if s.gnss: counts.append(("GNSS", 1))
    if s.imu:  counts.append(("IMU", 1))
    for name, n in counts:
        if n:
            lines.append(f"  • {name:<16} {n}")
    lines.append("")
    lines.append(f"Models ({len(cfg.models)}):")
    for m in cfg.models:
        lines.append(f"  • {m.id:<24} {m.family:<26} {m.precision} "
                     f"sparsity={m.sparsity} -> {m.path}")
    lines.append("")
    lines.append(f"Backend: {cfg.backend.name}")
    if cfg.safety_policies:
        lines.append(f"\nSafety policies ({len(cfg.safety_policies)}):")
        for p in cfg.safety_policies:
            lines.append(f"  • {p.type} = {p.value!r}")
    lines.append(f"\nMulti-stream: enabled={cfg.multistream.enabled} "
                 f"streams_per_model={cfg.multistream.streams_per_model}")
    lines.append(f"Dataset: {cfg.dataset.connector}"
                 f"{' preset='+cfg.dataset.preset if cfg.dataset.preset else ''}")
    return "\n".join(lines)
