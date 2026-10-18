"""Model-free entry point for no-feed, repeated audio intervention captures."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import html
import json
from pathlib import Path
import shutil
import time

from .audio_response import PROTOCOL, VERSION, build_response, valid_condition
from .preset_pipeline import capture_records, digest_file, read_json, write_json


def validate_manifest(manifest):
    if manifest.get("protocol") != PROTOCOL:
        raise ValueError("Controlled responses require a versioned intervention capture manifest")
    conditions = manifest.get("conditions", [])
    if not isinstance(conditions, list) or not 1 <= len(conditions) <= 512:
        raise ValueError("Expected 1..512 capture conditions")
    ids, cells = set(), set()
    for condition in conditions:
        cid = condition.get("id")
        if not isinstance(cid, str) or not cid or cid in ids:
            raise ValueError("Duplicate or invalid condition id")
        ids.add(cid)
        role, mode, trial = (condition.get(k) for k in ("input_role", "audio_mode", "trial"))
        if role not in {"engine-only", "neutral-a", "neutral-b"}:
            raise ValueError("Artist/reference inputs are not valid isolated evidence")
        if type(trial) is not int or not 0 <= trial < 12:
            raise ValueError("Invalid independent trial index")
        if (role, mode, trial) in cells:
            raise ValueError("Duplicate experimental cell is not an independent replicate")
        cells.add((role, mode, trial))
        if condition.get("input_mix") != (0 if role == "engine-only" else .15):
            raise ValueError("The declared input-isolation role disagrees with its source mix")
        if condition.get("protocol") != PROTOCOL:
            raise ValueError("Mixed protocols cannot be compared")
        if condition.get("state") == "captured":
            if condition.get("fresh_instance") is not True:
                raise ValueError("Fresh renderer state must be recorded for each condition")
            stimulus = condition.get("stimulus", {})
            if stimulus.get("trial") != trial or stimulus.get("id") != mode:
                raise ValueError("Audio stimulus identity disagrees with the experimental cell")
            if len(stimulus.get("sha256", "")) != 64:
                raise ValueError("PCM must have a retained checksum")
            expected_frames = list(range(0, 12*condition["render_fps"]+1,
                                         condition["render_fps"]//condition["capture_fps"]))
            actual = [f["frame"] for f in condition.get("captures", [])]
            if actual != expected_frames:
                raise ValueError("Capture clock or coverage differs from the declared protocol")
            if any(abs(f["output_seconds"]-f["frame"]/condition["render_fps"]) > 1e-8
                   for f in condition["captures"]):
                raise ValueError("Capture timestamps are not tied to the render clock")
    return conditions


def public_condition(condition):
    """Curated evidence contains no file paths or arbitrary engine log strings."""
    keys = ("id", "protocol", "state", "trial", "input_role", "input_mix", "source_pixels_sha256",
            "audio_mode", "render_fps", "capture_fps", "edge", "seed", "seed_control", "fresh_instance",
            "preset_sha256", "processed_frames", "signature", "pcm_delivery")
    result = {key: condition[key] for key in keys if key in condition}
    result["captured_frames"] = len(condition.get("captures", []))
    result["capture_timeline"] = [{"frame": f["frame"], "output_seconds": f["output_seconds"],
        "output_sha256": f.get("output", {}).get("sha256"),
        "input_sha256": f.get("input", {}).get("sha256"),
        **({"pcm_rms": f["pcm_rms"]} if "pcm_rms" in f else {})} for f in condition.get("captures", [])]
    result["stimulus"] = {k: v for k, v in condition.get("stimulus", {}).items() if k != "path"}
    sample_rate = condition.get("stimulus", {}).get("sample_rate")
    samples = condition.get("stimulus", {}).get("samples_per_channel")
    fps = condition.get("render_fps")
    if isinstance(sample_rate, int) and isinstance(samples, int) and fps:
        for item in result["capture_timeline"]:
            end = min(samples, item["frame"] * sample_rate // fps)
            start = max(0, end - 1024)
            item["pcm_window"] = {"start_sample": start, "end_sample_exclusive": end,
                                  "zero_padding_samples": 1024-(end-start),
                                  "sample_rate": sample_rate, "basis": "derived from recorded render clock/history-window contract"}

    result["runtime"] = {k: v for k, v in condition.get("runtime", {}).items()
                         if k in {"engine", "version", "opengl", "renderer", "source_commit", "library_sha256", "seed"}}
    return result


def analyse(captures, output, *, analyser=None):
    from pixelcue.visual_sequence import analyse_sequences
    manifest = read_json(captures) if not isinstance(captures, dict) else captures
    conditions = validate_manifest(manifest)
    records = capture_records(manifest)
    if len(records) > 32768:
        raise ValueError("Split the experiment: at most 32768 frames per preset")
    out = Path(output)
    out.mkdir(parents=True, exist_ok=True)
    print(f"Measuring {len(records)} captured frames before any optional appearance models.", flush=True)
    if records:
        temporal = (analyser or analyse_sequences)(records, max_edge=256, max_gap=.25)
    else:
        temporal = {"frames": [], "conditions": [], "errors": [], "state": "partial"}
    result = build_response(temporal, manifest)
    if manifest.get("expected_conditions") != len(conditions):
        result["state"] = "partial"
    result["preset"] = {k: manifest.get("preset", {}).get(k) for k in ("id", "name", "engine")}
    result["performance"] = {"measured_frames": len(temporal["frames"]), "new_semantic_frames": 0,
                              "requested_conditions": manifest.get("expected_conditions"),
                              "completed_conditions": sum(valid_condition(c) for c in conditions)}
    evidence = {"temporal": temporal, "conditions": [public_condition(c) for c in conditions]}
    payload = gzip.compress(json.dumps(evidence, sort_keys=True, allow_nan=False).encode(), mtime=0)
    result["evidence"] = {"file": "evidence.json.gz", "sha256": hashlib.sha256(payload).hexdigest()}
    result["analysis_code"] = {"audio_response": digest_file(Path(__file__).with_name("audio_response.py")),
                               "entrypoint": digest_file(__file__), "version": VERSION}
    if (out / "response.json").exists():
        history = out / "history" / str(time.time_ns())
        history.mkdir(parents=True)
        for name in ("response.json", "evidence.json.gz", "report.html", "report.md"):
            if (out / name).is_file():
                shutil.copy2(out / name, history / name)
    temp = out / "evidence.json.gz.tmp"
    temp.write_bytes(payload)
    temp.replace(out / "evidence.json.gz")
    write_json(out / "response.json", result)
    write_report(out, result)
    return result


STYLE = """body{font:16px/1.6 system-ui;background:#0e1826;color:#edf4fa;max-width:1150px;margin:35px auto;padding:0 22px}
h1{line-height:1.2}a{color:#92dfea}.lead{font-size:1.3em;border-left:4px solid #60c7c1;padding:16px 22px;background:#16283a}
table{border-collapse:collapse;width:100%;font-size:14px}td,th{border-bottom:1px solid #334b62;text-align:left;padding:10px}th{color:#8ad7d4}
small{color:#adc1d3}input{font:inherit;width:90%;padding:12px;background:#182b40;color:inherit}article{margin:24px 0;padding:20px;background:#182b40}
pre{white-space:pre-wrap;overflow-wrap:anywhere}.note{color:#bdd1de}details{margin:24px 0}.scroll{overflow:auto}"""


def document(title, body):
    return (f"<!doctype html><html lang='en'><meta charset='utf-8'><meta name='viewport' content='width=device-width'>"
            f"<title>{html.escape(title)}</title><style>{STYLE}</style>{body}</html>")


def write_report(out, result):
    name = result.get("preset", {}).get("name") or "Preset"
    lines = [f"# {name}", "", result["summary"], "", "## Audio contrasts against silence", "",
             "Values are repeat-level effect screens, not p-values. 'Not resolved' is not 'no effect'.", ""]
    table = "<div class='scroll'><table><tr><th>Sound</th><th>Measure</th><th>Result</th><th>Effect</th><th>Noise floor</th><th>Trials</th><th>Onset-locked</th></tr>"
    for row in result["contrasts"]:
        number = lambda x: "—" if x is None else f"{x:.4g}"
        values = [row["probe"], row["metric"], row["state"], number(row["effect"]),
                  number(row["noise_floor"]), str(row["trials"]), str(row.get("onset_locked", False))]
        table += "<tr>" + "".join("<td>"+html.escape(v)+"</td>" for v in values) + "</tr>"
        lines.append(" | ".join(values))
    table += "</table></div>"
    lines += ["", "## Evidence scope", "", result["description_scope"], *result["limits"]]
    (out / "report.md").write_text("\n".join(lines)+"\n", encoding="utf-8")
    body = (f"<small>PRISMMOVES · CONTROLLED AUDIO RESPONSE · {html.escape(result['state'])}</small>"
            f"<h1>{html.escape(name)}</h1><p class='lead'>{html.escape(result['summary'])}</p>"
            "<p class='note'>The paragraph uses zero-image-feed conditions only. Reference colours, shapes, captions and titles cannot supply its claims.</p>"
            "<h2>Measured sound effects</h2><p>Each contrast uses repeated fresh renders, a matched silent control and the same render clock."
            " Unresolved and missing evidence remain explicit.</p>"+table)
    body += "<details><summary>Frequency, level and envelope comparisons</summary><pre>"+html.escape(json.dumps(result["probe_comparisons"], indent=2))+"</pre></details>"
    body += "<details><summary>Reference-transfer experiments (excluded from the preset summary)</summary><pre>"+html.escape(json.dumps(result["input_transfers"], indent=2))+"</pre></details>"
    body += "<details><summary>Claims, provenance and limits</summary><pre>"+html.escape(json.dumps(result["claims"], indent=2))+"\n"+html.escape("\n".join(result["limits"]))+"</pre></details>"
    body += "<p><a href='response.json'>Response JSON</a> · <a href='evidence.json.gz'>Compressed measurement evidence</a></p>"
    (out / "report.html").write_text(document(name, body), encoding="utf-8")


def index(out):
    out = Path(out)
    entries, body = [], "<small>FRAMERUDIMENTS × PRISMMOVES</small><h1>What the preset does.<br>What sound changes.</h1><p>Reference-free summaries with repeated audio controls.</p><input id='search' aria-label='Search presets' placeholder='Search motion, audio response or preset name'>"
    for path in sorted(out.glob("*/response.json")):
        data = read_json(path)
        if data.get("schema") != "prismmoves.audio-response/1":
            continue
        row = {"preset": data.get("preset", {}), "summary": data["summary"], "state": data["state"],
               "report": path.parent.name+"/report.html"}
        preview = path.parent / "previews.json"
        if preview.exists():
            sheets = read_json(preview).get("sheets", [])
            chosen = next((s for s in sheets if s.get("kind") == "semantic"), sheets[0] if sheets else None)
            if chosen:
                row["preview"] = path.parent.name + "/" + chosen["file"]
                body += f"<a href='{html.escape(row['report'], quote=True)}'><img loading='lazy' style='width:100%;max-width:700px' src='{html.escape(row['preview'], quote=True)}' alt='Captured output contact sheet'></a>"
        entries.append(row)
        body += f"<article><small>{html.escape(str(row['preset'].get('engine', 'unknown')))} · {html.escape(row['state'])}</small>"
        body += f"<h2>{html.escape(str(row['preset'].get('name', 'Preset')))}</h2><p>{html.escape(row['summary'])}</p><a href='{html.escape(row['report'], quote=True)}'>Inspect audio response and evidence</a></article>"
    body += "<script>document.getElementById('search').addEventListener('input',e=>{const terms=e.target.value.toLowerCase().trim().split(/\\s+/);for(const c of document.querySelectorAll('article'))c.hidden=!terms.every(t=>c.textContent.toLowerCase().includes(t));});</script>"
    (out / "index.html").write_text(document("Controlled preset responses", body), encoding="utf-8")
    write_json(out / "index.json", {"schema": "prismmoves.audio-response-atlas/1", "entries": entries})
    return entries


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    a = sub.add_parser("analyse")
    a.add_argument("--captures", type=Path, required=True)
    a.add_argument("--out", type=Path, required=True)
    i = sub.add_parser("index")
    i.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    try:
        if args.command == "analyse":
            result = analyse(args.captures, args.out)
            print(result["summary"], flush=True)
            return 0 if result["state"] == "completed" else 2
        index(args.out)
        return 0
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(1, f"Controlled response analysis stopped: {exc}\n")


if __name__ == "__main__":
    raise SystemExit(main())
