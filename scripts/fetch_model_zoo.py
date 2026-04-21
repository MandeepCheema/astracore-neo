"""Download the AstraCore model zoo.

Usage::

    python scripts/fetch_model_zoo.py                 # fetch everything
    python scripts/fetch_model_zoo.py --only squeezenet-1.1 mobilenetv2-7
    python scripts/fetch_model_zoo.py --list          # show zoo + local status
    python scripts/fetch_model_zoo.py --verify        # re-check checksums
    python scripts/fetch_model_zoo.py --update-manifest  # write computed SHA-256s
                                                      # back into zoo_manifest.json

Models land under ``data/models/zoo/<name>.onnx``. The fetcher is
idempotent — files already on disk are skipped unless ``--force``.

A JSON manifest is written/updated at ``data/models/zoo/manifest.json``
so the download state is reproducible.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

import sys as _sys
_sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from astracore.zoo import ZOO, ZooModel, get, local_paths


ZOO_DIR = Path("data/models/zoo")
MANIFEST_PATH = ZOO_DIR / "manifest.json"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, show_progress: bool = True) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")

    last_pct = [-1]

    def _hook(blocks: int, block_sz: int, total_sz: int) -> None:
        if not show_progress:
            return
        done = blocks * block_sz
        if total_sz > 0:
            pct = min(100, done * 100 // total_sz)
            # Emit ONE line per 10% to avoid scroll / buffer spam.
            if pct // 10 != last_pct[0] // 10:
                last_pct[0] = pct
                sys.stdout.write(
                    f"  {dest.name}: {pct:3d}% "
                    f"({done // (1024*1024)}/{total_sz // (1024*1024)} MB)\n"
                )
                sys.stdout.flush()

    urllib.request.urlretrieve(url, str(tmp), reporthook=_hook)
    tmp.replace(dest)


def _load_manifest() -> dict:
    if MANIFEST_PATH.exists():
        with MANIFEST_PATH.open() as fh:
            return json.load(fh)
    return {}


def _save_manifest(data: dict) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    with MANIFEST_PATH.open("w") as fh:
        json.dump(data, fh, indent=2, sort_keys=True)


def fetch_model(m: ZooModel, *, force: bool = False,
                verify: bool = True) -> dict:
    """Fetch a single model; return a manifest entry."""
    entry = {"name": m.name, "url": m.url}

    if m.url is None:
        # In-repo model (yolov8n). Just report its status.
        path = local_paths()[m.name]
        entry["path"] = str(path)
        entry["in_repo"] = True
        if path.exists():
            entry["sha256"] = _sha256(path)
            entry["size_bytes"] = path.stat().st_size
        else:
            entry["error"] = "expected in-repo but missing"
        return entry

    path = m.local_path
    entry["path"] = str(path)

    need_download = force or not path.exists()
    if need_download:
        print(f"-> downloading {m.name}  ({m.size_bytes or '?'} bytes)")
        t0 = time.perf_counter()
        _download(m.url, path)
        entry["download_s"] = round(time.perf_counter() - t0, 2)
    else:
        print(f"[cached] {m.name}")

    if verify:
        entry["sha256"] = _sha256(path)
        entry["size_bytes"] = path.stat().st_size
        if m.sha256 and entry["sha256"] != m.sha256:
            entry["error"] = (
                f"sha256 mismatch: expected {m.sha256}, got {entry['sha256']}"
            )

    return entry


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Fetch AstraCore ONNX model zoo")
    p.add_argument("--only", nargs="+", default=None,
                   help="only fetch these model names")
    p.add_argument("--force", action="store_true",
                   help="re-download even if file is present")
    p.add_argument("--list", action="store_true",
                   help="list zoo entries + local status and exit")
    p.add_argument("--verify", action="store_true",
                   help="verify checksums of existing files (no downloads)")
    p.add_argument("--no-progress", action="store_true",
                   help="suppress download progress bar")
    args = p.parse_args(argv)

    targets = ZOO if args.only is None else [get(n) for n in args.only]

    if args.list:
        paths = local_paths()
        for m in ZOO:
            status = "[OK]" if paths[m.name].exists() else "[  ]"
            print(f"{status} {m.name:<28} {m.display_name} ({m.family})")
        return 0

    if args.verify:
        manifest = _load_manifest()
        ok = True
        for m in targets:
            path = local_paths()[m.name]
            if not path.exists():
                print(f"missing: {m.name}")
                ok = False; continue
            digest = _sha256(path)
            recorded = manifest.get(m.name, {}).get("sha256")
            match = (recorded is None) or (digest == recorded)
            print(f"{'✓' if match else '✗'} {m.name}  sha256={digest[:16]}…")
            ok = ok and match
        return 0 if ok else 1

    manifest = _load_manifest()
    for m in targets:
        entry = fetch_model(m, force=args.force)
        manifest[m.name] = entry
    _save_manifest(manifest)
    print(f"\nManifest: {MANIFEST_PATH}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
