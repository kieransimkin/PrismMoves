from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import sys
import urllib.error

from . import __version__
from .backend import PixelCueBackend
from .report import build_report, html_report, markdown


def main(argv=None):
    parser = argparse.ArgumentParser(description="Compose PixelCue image analysis into an evidence-labelled visual profile.")
    parser.add_argument("--version", action="version", version=__version__)
    parser.add_argument("images", nargs="+", type=Path)
    parser.add_argument("--out", required=True, type=Path, help="New output directory; existing reports are never overwritten")
    parser.add_argument("--pixelcue-url", help="Call PixelCue's loopback HTTP service instead of its Python API")
    parser.add_argument("--models", nargs="*", choices=["clip", "dinov2", "blip", "ocr"], default=[])
    parser.add_argument("--pixelcue-models", nargs="*", default=[], help="Existing PixelCue VLM tagger profiles; requires --allow-download")
    parser.add_argument("--model-path", action="append", default=[], metavar="NAME=PATH")
    parser.add_argument("--models-dir", type=Path, default=None, help="Project-local models/ folder (default: current project/models)")
    parser.add_argument("--allow-download", action="store_true", help="Explicitly permit selected PixelCue model downloads")
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    parser.add_argument("--max-edge", type=int, default=768)
    parser.add_argument("--background", default="#ffffff")
    parser.add_argument("--role", default="unknown", help="User-supplied role; not inferred approval")
    parser.add_argument("--lineage", default="unknown", help="User-supplied lineage; shared across this invocation")
    args = parser.parse_args(argv)
    if args.out.exists():
        parser.error("Output directory already exists; choose a new directory")
    if len(args.images) > 64:
        parser.error("At most 64 images per bounded invocation")
    try:
        paths = dict(value.split("=", 1) for value in args.model_path)
        if set(paths)-set(args.models):
            raise ValueError("--model-path must name a selected model")
        if not 32 <= args.max_edge <= 2048:
            raise ValueError("--max-edge must be between 32 and 2048")
        backend = PixelCueBackend(args.pixelcue_url)
        identity = backend.identity()
        items = []
        for index, path in enumerate(args.images, 1):
            item = {"id": f"A{index:02}", "name": path.name, "role": args.role, "lineage": args.lineage}
            try:
                result = backend.analyse(path.resolve(), models=args.models, model_paths=paths,
                    keyword_models=args.pixelcue_models, allow_download=args.allow_download,
                    max_edge=args.max_edge, background=args.background, device=args.device,
                    models_dir=str(args.models_dir.resolve()) if args.models_dir else None)
                item.update(status="completed", analysis=result)
            except (OSError, ValueError, RuntimeError) as exc:
                item.update(status="unavailable", reason=f"{type(exc).__name__}: {str(exc)[:1200]}")
            items.append(item)
        report = build_report(items, identity)
        args.out.mkdir(parents=True, exist_ok=False)
        text = markdown(report)
        outputs = {"analysis.json": json.dumps(report, ensure_ascii=False, indent=2, allow_nan=False)+"\n",
            "report.md": text, "report.html": html_report(report, text)}
        for name, content in outputs.items():
            (args.out/name).write_text(content, encoding="utf-8")
        checksums = "".join(hashlib.sha256((args.out/name).read_bytes()).hexdigest()+"  "+name+"\n" for name in outputs)
        (args.out/"SHA256SUMS.txt").write_text(checksums, encoding="ascii")
        unavailable = sum(item["status"] != "completed" for item in items)
        missing_models = sum(m["status"] != "completed" for item in items if item["status"] == "completed" for m in item["analysis"]["models"])
        print(f"{args.out.resolve() / 'report.html'}; {len(items)-unavailable}/{len(items)} images; {missing_models} unavailable model(s)")
        return 2 if unavailable or missing_models else 0
    except (ImportError, OSError, ValueError, RuntimeError) as exc:
        print(f"PrismMoves: {exc}", file=sys.stderr)
        return 1
