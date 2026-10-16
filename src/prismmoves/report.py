"""Evidence-labelled presentation of PixelCue receipts, without interpreting scores as truth."""
from __future__ import annotations

import html
import json
from datetime import datetime, timezone

AXES = ["Palette and contrast", "Typography and marks", "Subjects, motifs and setting",
    "Composition, hierarchy and crop", "Medium, texture, material, light and grade",
    "Motion and interaction", "Tone, narrative and representation", "Accessibility and provenance"]


def build_report(items, backend):
    report = {"schema": "prismmoves.visual-profile/1", "version": "0.1.0",
        "created_at": datetime.now(timezone.utc).isoformat(), "backend": backend,
        "items": items, "scope": "individual image profiles; collection rules require independent comparison evidence",
        "unknowns": ["Artist intent, approval, provenance, reuse rights and consent are unknown unless supplied as reviewed context.",
            "A single image cannot establish core invariants, controlled variation, negative rules or outliers.",
            "Model outputs are interpretations; unrequested and unavailable models do not establish absence.",
            "Palette contrast and saliency proxies do not establish actual text legibility, gaze or accessibility compliance.",
            "Static images provide no evidence of animation, phone input, responsive layout or temporal rhythm."],
        "proposed_next_evidence": ["Review semantic and OCR hypotheses against the actual image.",
            "Compare independently authored approved assets, with dates, roles and lineage, before declaring identity rules."]}
    return report


def markdown(report):
    lines = ["# PrismMoves visual-language analysis", "", f"Generated: {report['created_at']}",
        "", "PixelCue owns the image measurements and model analysis. PrismMoves organises the evidence.",
        "", "## Evidence ledger", "", "| ID | Image | Role (supplied) | Lineage (supplied) | SHA-256 | State |",
        "|---|---|---|---|---|---|"]
    def cell(text):
        return html.escape(str(text)).replace("|", "\\|").replace("\n", " ").replace("\r", " ")
    for item in report["items"]:
        p = item.get("analysis", {})
        lines.append(f"| {item['id']} | {cell(item['name'])} | {cell(item['role'])} | {cell(item['lineage'])} | {p.get('source', {}).get('sha256', 'unavailable')} | {item['status']} |")
    for item in report["items"]:
        lines += ["", f"## {item['id']}: {cell(item['name'])}", ""]
        if item["status"] != "completed":
            lines += [f"**Unknown:** analysis unavailable. {cell(item['reason'])}"]
            continue
        p = item["analysis"]
        c, t, g = (p["measurements"][key] for key in ("colour", "texture", "composition"))
        dimensions = p["source"]["dimensions"]
        palette_text = ', '.join(f"{v['hex']} ({v['sample_fraction']:.0%})" for v in c['palette'])
        lines += ["### Observed measurements", "",
            f"- Oriented source: {dimensions[0]} × {dimensions[1]} px. Analysis raster: {p['source']['analysis_dimensions']}.",
            f"- Palette (approximate sampled centres): {palette_text}.",
            f"- Mean HSV saturation: {c['mean_saturation']:.3f}; grayscale entropy: {t['grayscale_entropy_bits']:.3f} bits.",
            f"- Linear luminance percentiles (0/5/50/95/100): {', '.join(f'{x:.3f}' for x in c['relative_luminance_percentiles'])}.",
            f"- Edge density at the stated threshold: {g['edge_density_above_0_08']:.3f}.",
            f"- Mirror errors, horizontal/vertical: {g['horizontal_mirror_mae']:.3f} / {g['vertical_mirror_mae']:.3f} (0 means exact mirror equality).",
            f"- Colour handling: {p['source']['color_handling']}. Alpha composited over {p['source']['background']}.",
            "", "### Model interpretations", ""]
        if not p["models"]:
            lines.append("No semantic or OCR models requested. Subjects, medium, materials and emotional tone remain unknown.")
        for model in p["models"]:
            lines.append(f"- **{cell(model['name'])} — {model['status']}**")
            if model["status"] != "completed":
                lines.append(f"  {cell(model.get('reason', 'No result'))}")
                continue
            result = model["result"]
            if "axes" in result:
                for axis, candidates in result["axes"].items():
                    lines.append(f"  {axis}: {cell(candidates[0]['prompt'])} (relative cosine {candidates[0]['cosine_similarity']:.3f}; review hypothesis).")
            if "caption" in result:
                lines.append("  Caption hypothesis: "+cell(result["caption"]))
            if "tags" in result:
                lines.append("  PixelCue tag hypotheses: "+cell(", ".join(result["tags"])))
            if "words" in result:
                lines.append("  OCR text hypotheses: "+cell(" ".join(w["text"] for w in result["words"])))
            if "embedding" in result:
                lines.append(f"  {len(result['embedding'])}-dimensional embedding retained in JSON; no semantic claim from the vector alone.")
            lines.append("  "+cell(result.get("caveat", "Model interpretation requires visual review.")))
        lines += ["", "### Visual-language coverage", "", "| Axis | Evidence and limits |", "|---|---|"]
        coverage = ["Measured palette, luminance and saturation. Roles and actual text adjacency need visual review.",
            "OCR hypotheses if requested and available. Font, mark ownership and malformed lettering require review.",
            "Caption, CLIP or PixelCue tag hypotheses if requested; no verified object inventory.",
            "Measured edge/grid/mirror/frequency features and a signal novelty proxy. Focal intent and safe zones require review.",
            "Measured LBP, co-occurrence and FFT features. Material, medium and lighting are model hypotheses.",
            "Unknown from a still image.", "Optional prompt-relative tone hypotheses. Intent, audience and personal identity remain unknown.",
            "Dimensions, alpha and colour handling measured. Rights, approval and accessibility conformance remain unknown."]
        lines += [f"| {axis} | {value} |" for axis, value in zip(AXES, coverage)]
        lines += ["", "### Limitations", ""]+["- "+cell(v) for v in p["limits"]+p["source"]["notes"]]
    lines += ["", "## Unknowns and conflicts", ""]+["- "+v for v in report["unknowns"]]
    lines += ["", "## Proposed next evidence", ""]+["- "+v for v in report["proposed_next_evidence"]]
    return "\n".join(lines)+"\n"


def html_report(report, text):
    e = html.escape
    sections = []
    for item in report["items"]:
        if item["status"] != "completed":
            continue
        p = item["analysis"]
        palette = "".join(f'<div class="swatch" style="background:{e(c["hex"])}"><span>{e(c["hex"])} · {c["sample_fraction"]:.0%}</span></div>'
                          for c in p["measurements"]["colour"]["palette"])
        cells = p["measurements"]["composition"]["grid"]
        cols = max(c["column"] for c in cells)+1
        grid = "".join(f'<div style="background:rgb({int(255*c["mean_luminance"])}, {int(255*c["mean_luminance"])}, {int(255*c["mean_luminance"])})" title="Cell {c["row"]},{c["column"]}: linear luminance {c["mean_luminance"]:.3f}"></div>' for c in cells)
        sections.append(f'<section><h2>{e(item["id"])} · {e(item["name"])}</h2><h3>Sampled palette</h3><div class="palette">{palette}</div><h3>Spatial luminance measurements</h3><div class="grid" style="grid-template-columns:repeat({cols},1fr)">{grid}</div></section>')
    return '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1"><title>PrismMoves analysis</title><style>body{font:16px/1.6 system-ui;background:#191426;color:#f2edf7;margin:auto;padding:2rem;max-width:1000px}h1,h2,h3{line-height:1.3}.palette{display:flex;flex-wrap:wrap;gap:8px}.swatch{width:130px;height:100px;display:flex;align-items:end}.swatch span{background:#111;color:white;padding:2px 5px;font-size:12px}.grid{display:grid;max-width:250px;gap:3px}.grid div{aspect-ratio:1;border:1px solid #777}pre{white-space:pre-wrap;overflow-wrap:anywhere;font:14px/1.6 system-ui}section{border-bottom:1px solid #81748f;padding-bottom:2rem}</style><main><h1>PrismMoves</h1><p>Measured by PixelCue · evidence-labelled visual profile</p>'+''.join(sections)+'<h2>Complete evidence report</h2><pre>'+e(text)+'</pre></main></html>'
