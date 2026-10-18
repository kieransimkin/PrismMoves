"""Temporal-first preset analysis, resumable semantic cache and legacy replay.

Usage: python -m prismmoves.preset_pipeline analyse --captures captures.json --out OUT
       python -m prismmoves.preset_pipeline legacy --reports REPORTS --out NEW_OUT
"""
from __future__ import annotations

import argparse
from collections import defaultdict
import gzip
import hashlib
import html
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import uuid

from .preset_description import DESCRIPTION_VERSION, select_semantic_frames, synthesise_description


def digest_file(path):
    with Path(path).open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + "."+uuid.uuid4().hex+".tmp")
    try:
        temp.write_bytes(canonical(value)+b"\n")
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def read_json(path, limit=128*1024**2):
    with Path(path).open("rb") as stream:
        data = stream.read(limit+1)
    if len(data) > limit:
        raise ValueError("JSON receipt exceeds byte limit")
    return json.loads(data)


def capture_records(manifest):
    records = []
    for condition in manifest.get("conditions", []):
        for frame in condition.get("captures", []):
            output, original = frame.get("output", {}), frame.get("input", {})
            # Even a partial condition can contain valuable completed frames.
            if not output.get("path") or not output.get("sha256"):
                continue
            identity = str(condition["id"])+":"+str(frame["frame"])
            records.append(dict(id=identity, condition_id=condition["id"], seconds=frame["output_seconds"],
                                path=output["path"], sha256=output["sha256"],
                                input_path=original.get("path"), input_sha256=original.get("sha256"),
                                imagery=condition.get("imagery"), audio_mode=condition.get("audio_mode"),
                                sample=condition.get("sample")))
    return records


def _versions():
    result = {}
    for name in ("pixelcue", "prismmoves", "numpy", "Pillow", "opencv-python-headless", "torch", "transformers"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = "not installed as a distribution"
    return result


def semantic_identity(models_dir, models, keyword_models, options):
    """Cache invalidates on model manifests, prompt/code changes and preprocessing.

    Unpinned checkpoints are usable but not eligible for cross-run caching.
    No weights are hashed/downloaded or moved by this operation.
    """
    from pixelcue import visual_batch, visual_sequence
    root = Path(models_dir)
    manifests, file_stats, pinned = {}, {}, True
    for name in models:
        if name == "ocr":
            pinned = False  # External executable/trained-data are not identified here.
            continue
        manifest = root/name/"pixelcue-model.json"
        if manifest.is_file():
            manifests[name] = digest_file(manifest)
            metadata = read_json(manifest)
            # Cheap mutation guard alongside the pinned manifest; no giant weight
            # rehash on every preset. Source manifests are verified at installation.
            stats = {}
            for relative in sorted(metadata.get("files_sha256", {})):
                model_file = (manifest.parent/relative).resolve()
                if not model_file.is_relative_to(manifest.parent.resolve()):
                    raise ValueError("Model manifest path escapes its checkpoint directory")
                if not model_file.is_file():
                    pinned = False
                    continue
                st = model_file.stat()
                stats[relative] = [st.st_size, st.st_mtime_ns]
            file_stats[name] = stats
        else:
            pinned = False
    # Existing keyword services have heterogeneous cache manifests. Never assume
    # a profile name identifies immutable model bytes across installations.
    if keyword_models:
        pinned = False
    identity = dict(options, models=list(models), keyword_models=list(keyword_models),
                    model_manifests=manifests, model_file_stats=file_stats, versions=_versions(),
                    batch_code=digest_file(visual_batch.__file__),
                    sequence_code=digest_file(visual_sequence.__file__),
                    prompt_bank_sha256=visual_batch.prompt_digest(visual_batch.VISUALISER_PROMPTS))
    return identity, pinned


def analyse_capture_manifest(captures, output, *, budget=24, models=("clip", "blip"),
                             keyword_models=(), models_dir=None, cache_dir=None,
                             allow_download=False, device="cpu", max_edge=256,
                             semantic_edge=384, max_gap=.25, warmup_seconds=.5,
                             progress=print, batch_analyser=None):
    """Save temporal evidence and the selection plan BEFORE loading any model."""
    from pixelcue.visual_sequence import analyse_sequences
    from pixelcue.visual_batch import analyse_batch
    manifest = read_json(captures) if not isinstance(captures, dict) else captures
    records = capture_records(manifest)
    out = Path(output)
    root = Path(models_dir or Path.cwd()/"models").resolve()
    cache = Path(cache_dir or root/"analysis-cache"/"preset-semantics").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    options = {"max_edge": semantic_edge, "background": "#000000", "device": device,
               "allow_download": allow_download, "prompt_profile": "visualiser"}
    identity, pinned = semantic_identity(root, models, keyword_models, options)
    # Retain identities, never copy machine-local source paths into curated output.
    signature_payload = {"description_version": DESCRIPTION_VERSION,
        "captures": [{k: v for k, v in r.items() if not k.endswith("path")} for r in records],
        "preset": manifest.get("preset", {}).get("id"), "semantic_identity": identity,
        "conditions": [{"id": c.get("id"), "state": c.get("state"), "captures": len(c.get("captures", []))}
                       for c in manifest.get("conditions", [])],
        "budget": budget, "max_edge": max_edge, "max_gap": max_gap, "warmup_seconds": warmup_seconds}
    signature = hashlib.sha256(canonical(signature_payload)).hexdigest()
    request_path = out/"request.json"
    if request_path.exists() and read_json(request_path).get("signature") != signature:
        raise ValueError("Analysis configuration changed; use a new output directory")
    out.mkdir(parents=True, exist_ok=True)
    write_json(request_path, {"signature": signature, "settings": signature_payload})
    temporal_path = out/"temporal.json"
    # Validate retained rasters, even when measurements are cached. Changed or
    # missing inputs invalidate reuse; they are never hidden by a model cache.
    sources_valid = True
    for row in records:
        for path_key, hash_key in (("path", "sha256"), ("input_path", "input_sha256")):
            if row.get(path_key):
                try:
                    sources_valid &= digest_file(row[path_key]) == row[hash_key]
                except OSError:
                    sources_valid = False
    temporal_hash = out/"temporal.sha256"
    temporal_valid = (temporal_path.is_file() and temporal_hash.is_file()
                      and temporal_hash.read_text(encoding="ascii").strip() == digest_file(temporal_path))
    if temporal_valid and sources_valid:
        temporal = read_json(temporal_path)
        progress("Temporal evidence reused; no semantic prerequisite.")
    elif records:
        progress(f"Measuring {len(records)} timed frames before semantic analysis...")
        temporal = analyse_sequences(records, max_edge=max_edge, max_gap=max_gap)
        write_json(temporal_path, temporal)
    else:
        temporal = {"schema": "pixelcue.visual-sequence/1", "frames": [], "conditions": [],
                    "state": "partial", "errors": [{"reason": "No completed renderer captures"}]}
        write_json(temporal_path, temporal)
    coverage = {"declared_conditions": len(manifest.get("conditions", [])),
                "conditions_without_captures": [c.get("id") for c in manifest.get("conditions", [])
                                               if not c.get("captures")],
                "partial_conditions": [c.get("id") for c in manifest.get("conditions", [])
                                       if c.get("state") not in (None, "captured")]}
    temporal["capture_coverage"] = coverage
    for condition in set(coverage["conditions_without_captures"] + coverage["partial_conditions"]):
        entry = {"condition_id": condition, "reason": "Renderer condition is missing or partial"}
        if entry not in temporal.setdefault("errors", []):
            temporal["errors"].append(entry)
    if temporal["errors"]:
        temporal["state"] = "partial"
    write_json(temporal_path, temporal)
    temporal_hash.write_text(digest_file(temporal_path)+"\n", encoding="ascii")
    plan = select_semantic_frames(temporal, budget=budget, warmup_seconds=warmup_seconds)
    write_json(out/"semantic-plan.json", plan)
    provisional = synthesise_description(temporal, (), plan=plan)
    write_json(out/"draft-description.json", provisional)
    progress("Provisional description: " + provisional["summary"])
    selected = plan["selected"]
    progress(f"Semantic plan: {len(selected)} unique frames from {len(temporal['frames'])} captures; "
             f"{plan['excluded_uniform_frames']} uniform frames need no caption.")
    # Unpinned models are intentionally session-local, not indefinitely reused.
    fingerprint = hashlib.sha256(canonical(identity)).hexdigest()
    namespace = fingerprint if pinned else fingerprint+"-unversioned-"+uuid.uuid4().hex
    by_id = {r["id"]: r for r in records}
    items, missing = {}, []
    def key(frame):
        return hashlib.sha256((namespace+":"+frame["sha256"]).encode()).hexdigest()
    expected_models = set(models) | {"pixelcue-tagger:"+name for name in keyword_models}
    for frame in selected:
        path = cache/(key(frame)+".json")
        cached = None
        if path.is_file():
            try:
                cached = read_json(path)
            except (OSError, ValueError):
                pass
        if (cached and cached.get("identity") == identity and cached.get("sha256") == frame["sha256"]
                and cached.get("item", {}).get("status") == "completed"
                and cached.get("item", {}).get("analysis", {}).get("source", {}).get("sha256") == frame["sha256"]
                and {m.get("name") for m in cached["item"].get("analysis", {}).get("models", [])} == expected_models
                and all(m.get("status") == "completed" for m in cached["item"]["analysis"]["models"])):
            items[frame["sha256"]] = cached["item"]
        else:
            missing.append(frame)
    hits = len(items)
    if missing and (models or keyword_models):
        progress(f"Semantic inference: {len(missing)} frames, {hits} exact cache hits; model-major batching.")
        started = time.monotonic()
        result = (batch_analyser or analyse_batch)([by_id[f["id"]]["path"] for f in missing],
                    models=models, keyword_models=keyword_models, models_dir=root, **options)
        semantic_seconds = time.monotonic()-started
        expected = {f["sha256"]: f for f in missing}
        for item in result.get("items", []):
            sha = item.get("analysis", {}).get("source", {}).get("sha256")
            if sha not in expected:
                continue
            actual_models = item.get("analysis", {}).get("models", [])
            if {m.get("name") for m in actual_models} != expected_models:
                continue
            items[sha] = item
            write_json(cache/(key(expected[sha])+".json"), {"identity": identity, "sha256": sha, "item": item})
    else:
        semantic_seconds = 0
    ordered_items = [items[f["sha256"]] for f in selected if f["sha256"] in items]
    description = synthesise_description(temporal, ordered_items, plan=plan,
                                         name=manifest.get("preset", {}).get("name", ""))
    missing_semantics = [f["sha256"] for f in selected if f["sha256"] not in items] if models or keyword_models else []
    if missing_semantics:
        description["status"] = "limited"
        description["semantic_status"] = "partial"
    report = {"schema": "prismmoves.preset-report/1", "signature": signature,
              "preset": {k: manifest.get("preset", {}).get(k) for k in ("id", "name", "engine")},
              "description": description, "selection": plan,
              "analysis_state": "partial" if temporal.get("errors") or missing_semantics or
                  description["semantic_status"] == "partial" else "completed",
              "temporal": temporal, "semantic_items": ordered_items,
              "missing_semantic_sha256": missing_semantics,
              "performance": {"measured_frames": len(temporal["frames"]),
                              "requested_unique_semantics": len(selected), "cache_hits": hits,
                              "new_semantic_frames": len(missing) if models or keyword_models else 0,
                              "semantic_seconds": round(semantic_seconds, 3),
                              "temporal_seconds": temporal.get("seconds"),
                              "persistent_cache_eligible": pinned}}
    write_json(out/"analysis.json", report)
    write_report(out, report)
    return report


def write_report(out, report):
    out = Path(out)
    description = report["description"]
    name = report.get("preset", {}).get("name") or "Visualiser preset"
    text = f"# {name}\n\n{description['summary']}\n\n"
    text += f"Status: {description['status']}; motion: {description['motion_status']}; semantics: {description['semantic_status']}.\n\n"
    text += "## Evidence\n\n"
    for claim in description.get("claims", []):
        text += f"- {claim['axis']}: {claim['text']} ({claim['evidence_class']}; {len(claim['evidence'])} references)\n"
    text += "\n" + "\n".join(description.get("limits", [])) + "\n"
    (out/"report.md").write_text(text, encoding="utf-8")
    html_text = "<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
    html_text += f"<title>{html.escape(name)}</title><style>body{{font:17px/1.6 system-ui;max-width:900px;margin:3rem auto;padding:0 1.5rem;background:#111927;color:#ecf3ff}}a{{color:#8ddbe6}}.lead{{font-size:1.35rem}}pre{{white-space:pre-wrap;overflow-wrap:anywhere}}details{{margin-top:2rem}}</style>"
    html_text += f"<h1>{html.escape(name)}</h1><p class='lead'>{html.escape(description['summary'])}</p>"
    html_text += f"<p>{html.escape(description['status'])} · Motion: {html.escape(description['motion_status'])}</p>"
    html_text += "<details><summary>Measurements, claim evidence and limitations</summary><pre>"+html.escape(text)+"</pre></details>"
    html_text += "<p><a href='analysis.json'>Full machine-readable evidence</a> · <a href='semantic-plan.json'>Frame selection</a></p></html>"
    (out/"report.html").write_text(html_text, encoding="utf-8")


def legacy_evidence(report, receipt=None):
    """Reorganise committed receipts without decoding previews or inventing flow."""
    items = [item for batch in (receipt or {}).get("batches", [])
             for item in batch.get("report", {}).get("items", [])]
    lookup = {item.get("analysis", {}).get("source", {}).get("sha256"): item for item in items}
    examples = (report.get("semantic") or {}).get("examples", [])
    bindings = (receipt or {}).get("frame_bindings", []) or examples
    frames = []
    for i, binding in enumerate(bindings):
        sha = binding.get("sha256", binding.get("image_sha256"))
        item = lookup.get(sha)
        if not item:
            continue
        m = item["analysis"]["measurements"]
        colour, composition = m.get("colour", {}), m.get("composition", {})
        values = colour.get("relative_luminance_percentiles", [0]*5)
        palette = colour.get("palette", [])
        white = 1.0 if values and min(values) > .98 else 0.0
        black = 1.0 if values and max(values) < .002 else 0.0
        contrast = colour.get("luminance_std", 0)
        descriptor = [c.get("mean_luminance", 0) for c in composition.get("grid", [])]
        descriptor += [c.get("mean_saturation", 0) for c in composition.get("grid", [])]
        frames.append({"id": str(binding.get("condition_id", "unknown"))+":"+str(binding.get("frame", i)),
                       "condition_id": binding.get("condition_id", "unknown"),
                       "seconds": binding.get("output_seconds", 0), "sha256": sha,
                       "features": {"mean_linear_luminance": sum(p.get("relative_luminance", 0)*p.get("sample_fraction", 0) for p in palette),
                                    "contrast": contrast, "luminance_std": contrast,
                                    "mean_saturation": colour.get("mean_saturation", 0),
                                    "hue_histogram_12": colour.get("hue_histogram_12", [0]*12),
                                    "near_white_fraction": white, "near_black_fraction": black,
                                    "edge_density": composition.get("edge_density_above_0_08", 0),
                                    "semantic_eligible": contrast > .012 and white < 1 and black < 1,
                                    "descriptor": descriptor}})
    conditions = [{"id": name, "summary": {"motion_status": "unknown", "motion": [],
                   "brightness_period": {"status": "unknown", "reason": "legacy sparse still frames"}}}
                  for name in sorted({r["condition_id"] for r in frames})]
    return {"schema": "pixelcue.visual-sequence/1", "frames": frames, "conditions": conditions,
            "errors": [], "limits": ["Legacy measurements only; dense temporal frames are not available."]}, items


def replay_legacy(reports, output, *, budget=24):
    """Read EVERY per-preset JSON, not a potentially stale index.json."""
    reports, output = Path(reports), Path(output)
    if output.exists():
        raise ValueError("Legacy replay output must be a new directory")
    output.mkdir(parents=True)
    results = []
    for path in sorted(reports.glob("*.json")):
        original = read_json(path)
        if not str(original.get("schema", "")).endswith(".preset"):
            continue
        compressed = path.with_suffix(".pixelcue.json.gz")
        receipt = None
        if compressed.is_file():
            expected = original.get("semantic_receipt_sha256")
            if expected and expected != digest_file(compressed):
                raise ValueError("Legacy semantic evidence hash mismatch: "+path.name)
            with gzip.open(compressed, "rb") as stream:
                data = stream.read(128*1024**2+1)
            if len(data) > 128*1024**2:
                raise ValueError("Decompressed semantic evidence exceeds bound")
            receipt = json.loads(data)
        evidence, items = legacy_evidence(original, receipt)
        plan = select_semantic_frames(evidence, budget=budget)
        selected = {x["sha256"] for x in plan["selected"]}
        sampled = [item for item in items if item.get("analysis", {}).get("source", {}).get("sha256") in selected]
        description = synthesise_description(evidence, sampled, plan=plan)
        record = {"schema": "prismmoves.preset-report/1", "signature": digest_file(path),
                  "preset": {k: original.get(k) for k in ("id", "name", "engine")},
                  "description": description, "selection": plan,
                  "legacy_source_sha256": digest_file(path), "legacy_source_filename": path.name,
                  "note": "Replayed existing interpretations. No new model or temporal inference.",
                  "performance": {"measured_frames": len(evidence["frames"]), "new_semantic_frames": 0,
                                  "simulated_selection_budget": budget}}
        target = output/path.stem
        target.mkdir()
        write_json(target/"analysis.json", record)
        write_json(target/"semantic-plan.json", plan)
        write_report(target, record)
        # Copy already-published evidence previews only; never inspect private paths.
        preview = original.get("preview")
        if preview:
            candidate = (reports.parent/preview).resolve()
            if candidate.is_relative_to(reports.parent.resolve()) and candidate.is_file():
                import shutil
                shutil.copyfile(candidate, target/"preview.webp")
        results.append(record)
    build_gallery(output)
    return results


def build_gallery(output):
    """Rebuild from per-preset receipts on each checkpoint/resume, preventing lag."""
    output = Path(output)
    cards, entries = [], []
    legacy_count = 0
    for path in sorted(output.glob("*/analysis.json")):
        record = read_json(path)
        if record.get("schema") != "prismmoves.preset-report/1":
            continue
        p, d = record["preset"], record["description"]
        legacy_count += bool(record.get("legacy_source_sha256"))
        terms = " ".join([str(p.get("name", "")), str(p.get("engine", "")), d["status"],
                          d["summary"], *d.get("movement", []), *d.get("style", []), *d.get("artefacts", [])]).lower()
        folder = path.parent.name
        preview = path.parent/"preview.webp"
        image = f'<img loading="lazy" src="{html.escape(folder)}/preview.webp" alt="Selected renderer captures">' if preview.is_file() else '<div class="missing">Evidence preview unavailable</div>'
        cards.append(f'<article data-engine="{html.escape(str(p.get("engine", "")), quote=True)}" data-search="{html.escape(terms, quote=True)}">{image}'
                     f'<span class="engine">{html.escape(str(p.get("engine", "unknown")))} · {html.escape(d["status"])}</span>'
                     f'<h2>{html.escape(str(p.get("name", "Visualiser")))}</h2><p>{html.escape(d["summary"])}</p>'
                     f'<a href="{html.escape(folder)}/report.html">Inspect supporting evidence →</a></article>')
        entries.append({"id": p.get("id"), "name": p.get("name"), "engine": p.get("engine"),
                        "description": d["summary"], "status": d["status"],
                        "report": folder+"/analysis.json", "sha256": digest_file(path)})
    write_json(output/"index.json", {"schema": "prismmoves.preset-atlas/1", "entries": entries})
    document = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>Preset behaviour · PrismMoves</title>
<style>body{background:#0b1220;color:#edf4fd;font:16px/1.55 system-ui;margin:0}header,main{max-width:1450px;margin:auto;padding:30px}header{padding-bottom:10px}small,.engine{color:#95b8c8}h1{font-size:clamp(28px,4vw,48px);margin:8px 0}h2{font-size:18px;line-height:1.3}p{line-height:1.6}input,select{font:inherit;border:1px solid #3e546b;background:#142138;color:inherit;padding:12px;border-radius:8px}input{width:min(75%,600px)}main{display:grid;grid-template-columns:repeat(auto-fill,minmax(300px,1fr));gap:22px}article{padding:20px;background:#142033;border:1px solid #294159;border-radius:14px;display:flex;flex-direction:column}article[hidden]{display:none}img{width:100%;max-height:250px;object-fit:contain;background:#080f1b;border-radius:6px}.missing{height:100px;color:#98adbb}a{color:#84d4df;margin-top:auto}.engine{font-size:13px;margin-top:16px}footer{padding:25px;color:#96afc3}label{display:inline-block;margin:8px 8px 0 0}#count{display:block;margin-top:10px}</style>
<header><small>FRAMERUDIMENTS × PRISMMOVES</small><h1>Find the movement. Choose the look.</h1><p>One description per preset. Dense temporal measurements, selective semantic interpretation, inspectable evidence.</p><label><input id="q" aria-label="Search preset descriptions" placeholder="Search rotation, rings, glow, washouts…"></label><label><select id="engine" aria-label="Renderer"><option value="">All renderers</option><option>butterchurn</option><option>projectm</option></select></label><span id="count" aria-live="polite"></span></header><main>''' + "".join(cards) + '''</main><footer>Descriptions cover tested inputs and time windows. Model interpretations may be wrong; no flash-safety or artistic-quality certification.</footer><script>
const q=document.getElementById('q'),engine=document.getElementById('engine'),cards=[...document.querySelectorAll('article')];
function filter(){const words=q.value.toLowerCase().trim().split(/\\s+/).filter(Boolean);let n=0;for(const card of cards){card.hidden=!!engine.value&&card.dataset.engine!==engine.value||!words.every(w=>card.dataset.search.includes(w));if(!card.hidden)n++}document.getElementById('count').textContent=n+' / '+cards.length+' presets'}q.addEventListener('input',filter);engine.addEventListener('change',filter);filter();</script></html>'''
    if entries and legacy_count == len(entries):
        document = document.replace("Find the movement. Choose the look.", "Existing captures. Clearer descriptions.")
        document = document.replace("One description per preset. Dense temporal measurements, selective semantic interpretation, inspectable evidence.",
            "Replayed archived still-image evidence. No new model inference or dense motion data; movement remains unknown where unmeasured.")
    (output/"index.html").write_text(document, encoding="utf-8")
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    analyse = sub.add_parser("analyse")
    analyse.add_argument("--captures", type=Path, required=True)
    analyse.add_argument("--out", type=Path, required=True)
    analyse.add_argument("--models-dir", type=Path)
    analyse.add_argument("--cache-dir", type=Path)
    analyse.add_argument("--budget", type=int, default=24)
    analyse.add_argument("--models", nargs="*", default=["clip", "blip"])
    analyse.add_argument("--keyword-models", nargs="*", default=[])
    analyse.add_argument("--allow-download", action="store_true")
    analyse.add_argument("--device", choices=["cpu", "cuda", "mps"], default="cpu")
    analyse.add_argument("--max-edge", type=int, default=256)
    analyse.add_argument("--semantic-edge", type=int, default=384)
    legacy = sub.add_parser("legacy")
    legacy.add_argument("--reports", type=Path, required=True)
    legacy.add_argument("--out", type=Path, required=True)
    legacy.add_argument("--budget", type=int, default=24)
    index = sub.add_parser("reindex")
    index.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyse":
            report = analyse_capture_manifest(args.captures, args.out, budget=args.budget,
                models=args.models, keyword_models=args.keyword_models, models_dir=args.models_dir,
                cache_dir=args.cache_dir, allow_download=args.allow_download, device=args.device,
                max_edge=args.max_edge, semantic_edge=args.semantic_edge)
            print(report["description"]["summary"])
            return 0 if report["description"]["status"] == "described" else 2
        if args.command == "legacy":
            rows = replay_legacy(args.reports, args.out, budget=args.budget)
            print(f"Replayed {len(rows)} reports without new inference: {args.out/'index.html'}")
            return 0
        build_gallery(args.out)
        return 0
    except (OSError, ValueError, ImportError, RuntimeError, KeyError) as exc:
        print(f"PrismMoves preset analysis: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
