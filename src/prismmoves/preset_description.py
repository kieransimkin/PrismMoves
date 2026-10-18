"""Evidence selection and one readable preset description; no image/model engine."""
from __future__ import annotations

from collections import defaultdict
import math
import re
import statistics

DESCRIPTION_VERSION = "2.0.0"
HUES = ("red", "orange", "yellow", "lime", "green", "turquoise", "cyan", "blue",
        "indigo", "violet", "magenta", "pink")


def _median(values, default=0):
    values = [float(x) for x in values if isinstance(x, (float, int)) and not isinstance(x, bool) and math.isfinite(x)]
    return statistics.median(values) if values else default


def _distance(a, b):
    a, b = a.get("features", {}), b.get("features", {})
    da, db = a.get("descriptor", []), b.get("descriptor", [])
    spatial = (sum((x-y)**2 for x, y in zip(da, db))/len(da))**.5 if da and len(da)==len(db) else 0
    ha, hb = a.get("hue_histogram_12", [0]*12), b.get("hue_histogram_12", [0]*12)
    return spatial + .2*sum(abs(x-y) for x, y in zip(ha, hb)) + .2*abs(a.get("edge_density", 0)-b.get("edge_density", 0))


def select_semantic_frames(evidence, *, budget=24, warmup_seconds=.5, novelty_floor=.06):
    """Coverage-first, then temporal/visual novelty; budget counts UNIQUE rasters.

    Exact duplicate frames share inference but retain all experimental bindings.
    Near-duplicates are NOT assigned someone else's caption. Unsupported conditions
    and exhausted budgets stay visible. Motion evidence never depends on this plan.
    """
    if type(budget) is not int or not 0 <= budget <= 64:
        raise ValueError("Semantic budget must be an integer between 0 and 64")
    if not 0 <= warmup_seconds <= 30 or not 0 <= novelty_floor <= 2:
        raise ValueError("Invalid semantic selection settings")
    frames = evidence.get("frames", [])
    groups = defaultdict(list)
    for frame in frames:
        if frame.get("features", {}).get("semantic_eligible", False):
            groups[frame["condition_id"]].append(frame)
    for key in groups:
        groups[key].sort(key=lambda r: (r["seconds"], r["id"]))
    selected, hashes, reasons = [], set(), {}
    def add(frame, reason):
        if frame["sha256"] in hashes:
            return False
        if len(selected) >= budget:
            return False
        selected.append(frame)
        hashes.add(frame["sha256"])
        reasons[frame["sha256"]] = reason
        return True
    # One mature, informative representative per condition, not the latest frame
    # blindly (many recorded examples end at a washed-out or transient frame).
    for condition, rows in sorted(groups.items()):
        mature = [r for r in rows if r["seconds"] >= warmup_seconds] or rows
        mid = (mature[0]["seconds"] + mature[-1]["seconds"]) / 2
        representative = min(mature, key=lambda r: (
            abs(r["seconds"]-mid), -r["features"].get("contrast", 0), r["id"]))
        add(representative, "condition coverage / mature frame")
    # Each condition gets a chance at a second distinct appearance before a busy
    # input consumes the remaining budget. Startup-only outliers rank after mature.
    while len(selected) < budget:
        progress = False
        for condition, rows in sorted(groups.items()):
            candidates = [r for r in rows if r["sha256"] not in hashes]
            if not candidates:
                continue
            local = [r for r in selected if r["condition_id"] == condition]
            comparison = local or selected
            def score(frame):
                novelty = min((_distance(frame, x) for x in comparison), default=1)
                mature_bonus = 1 if frame["seconds"] >= warmup_seconds else .65
                return novelty*mature_bonus
            candidate = max(candidates, key=lambda r: (score(r), r["seconds"], r["id"]))
            if score(candidate) >= novelty_floor:
                progress |= add(candidate, "distinct appearance / temporal phase")
            if len(selected) == budget:
                break
        if not progress:
            break
    # A sampled exact duplicate is still supported by the ORIGINAL frame's hash;
    # it is not a statistical independence claim and is never inflated into votes.
    covered = {r["condition_id"] for r in frames if r["sha256"] in hashes}
    all_conditions = {r["condition_id"] for r in frames}
    distances = [min((_distance(r, x) for x in selected), default=1.0)
                 for r in frames if r.get("features", {}).get("semantic_eligible")]
    return {"schema": "prismmoves.semantic-plan/1", "policy_version": DESCRIPTION_VERSION,
            "budget": budget, "warmup_seconds": warmup_seconds, "novelty_floor": novelty_floor,
            "selected": [{"id": r["id"], "sha256": r["sha256"], "condition_id": r["condition_id"],
                          "seconds": r["seconds"], "reason": reasons[r["sha256"]]} for r in selected],
            "unique_inferences": len(selected), "captured_frames": len(frames),
            "exact_unique_frames": len({r["sha256"] for r in frames}),
            "excluded_uniform_frames": sum(not r.get("features", {}).get("semantic_eligible", False) for r in frames),
            "conditions_covered": sorted(covered),
            "conditions_without_semantic_sample": sorted(all_conditions-covered),
            "max_unrepresented_descriptor_distance": max(distances, default=0),
            "budget_exhausted": len(selected) == budget and any(d > novelty_floor for d in distances),
            "inference_reduction_vs_every_frame": 1-len(selected)/len(frames) if frames else 0,
            "limits": ["Novelty is measured from cheap raster descriptors, not semantic equivalence.",
                       "Unselected near-duplicates retain no inferred individual caption.",
                       "Coverage is experimental coverage, not independent statistical confidence."]}


# Constrain model prose to a visual vocabulary: preserve useful shape analogies,
# never pass through hallucinated rugs, cats, demographic/safety tags or preset names.
MOTIFS = (
    (r"\b(rings?|circles?|circular|orbs?|concentric)\b", "ring-like forms"),
    (r"\b(flower|petals?|rosette|radial)\b", "petal-like radial geometry"),
    (r"\b(particles?|dots|speckled|speckles|sparkles)\b", "fine particles"),
    (r"\b(stripes?|striped|wave-like)\b", "striped patterns"),
    (r"\b(grid|lattice|checkerboard)\b", "grid-like geometry"),
    (r"\b(fractal|branching|filigree)\b", "intricate branching patterns"),
    (r"\b(cloud-like|clouds|smoke-like)\b", "soft cloud-like forms"),
)
LOOKS = (
    (r"\b(neon|luminous neon outlines)\b", "neon-like glow"),
    (r"\b(soft diffuse glow|glowing|glow)\b", "soft glow"),
    (r"\b(smooth digital gradients)\b", "smooth gradients"),
    (r"\b(crisp thin lines|filigree)\b", "fine linework"),
    (r"\b(soft blurred edges)\b", "soft edges"),
    (r"\b(blocky pixelated shapes)\b", "blocky shapes"),
)


def _semantic_votes(frames, items):
    lookup = {item.get("analysis", {}).get("source", {}).get("sha256"): item for item in items
              if item.get("status") == "completed"}
    votes, evidence = defaultdict(set), defaultdict(set)
    per_condition_sha = set()
    for frame in frames:
        sha = frame["sha256"]
        # Within one condition an identical raster is counted only once.
        if (frame["condition_id"], sha) in per_condition_sha:
            continue
        per_condition_sha.add((frame["condition_id"], sha))
        item = lookup.get(sha)
        if not item or not frame.get("features", {}).get("semantic_eligible", False):
            continue
        texts = []
        for model in item["analysis"].get("models", []):
            if model.get("status") != "completed":
                continue
            result = model.get("result", {})
            caption = result.get("caption")
            if isinstance(caption, str):
                texts.append(caption.lower())
            for axis, candidates in result.get("axes", {}).items():
                # Ignore the old generic "outer space / ominous / 3D render" bank.
                if axis not in {"motif", "light", "surface", "structure"} or len(candidates) < 2:
                    continue
                top, runner = candidates[:2]
                margin = top.get("cosine_similarity", 0)-runner.get("cosine_similarity", 0)
                if margin >= .025:
                    texts.append(str(top.get("prompt", "")).lower())
        text = " ".join(texts)
        for pattern, label in MOTIFS + LOOKS:
            if re.search(pattern, text):
                votes[label].add((frame["condition_id"], sha))
                evidence[label].add(frame["id"])
    return votes, evidence


def synthesise_sample_description(evidence, semantic_items=(), *, name="", engine="", plan=None):
    """A short paragraph with per-claim evidence; names never drive conclusions."""
    frames = evidence.get("frames", [])
    claims, clauses = [], []
    def claim(text, axis, refs, evidence_class="measurement"):
        refs = sorted(set(refs))
        claims.append({"text": text, "axis": axis, "evidence_class": evidence_class, "evidence": refs})
        clauses.append(text)
    if not frames:
        return {"schema": "prismmoves.preset-description/1", "version": DESCRIPTION_VERSION,
                "summary": "No usable rendered frames are available; movement, artefacts and style remain unknown.",
                "status": "unavailable", "motion_status": "unknown", "semantic_status": "unavailable",
                "claims": [], "movement": [], "effects": [], "artefacts": [], "style": []}
    relevant = [f for f in frames if f.get("features", {}).get("semantic_eligible")] or frames
    grouped = defaultdict(list)
    for frame in relevant:
        grouped[frame["condition_id"]].append(frame)
    # Equal weight per input condition. A long recording must not dominate style.
    hue = [sum(_median(f["features"].get("hue_histogram_12", [0]*12)[i] for f in rows)
               for rows in grouped.values())/len(grouped) for i in range(12)]
    colours = [HUES[i] for i in sorted(range(12), key=lambda i: -hue[i])[:2] if hue[i] > .08]
    sat = _median(_median(f["features"].get("mean_saturation", 0) for f in rows) for rows in grouped.values())
    light = _median(_median(f["features"].get("mean_linear_luminance", 0) for f in rows) for rows in grouped.values())
    contrast = _median(f["features"].get("contrast", 0) for f in relevant)
    colour_text = " and ".join(colours) if colours else "largely monochrome tones" if sat < .1 else "varied colours"
    tone = "A dark" if light < .12 else "A bright" if light > .55 else "An abstract"
    finish = ", highly saturated" if sat > .65 and colours else ""
    claim(f"{tone}{finish} visualisation built around {colour_text}.", "style", [f["id"] for f in relevant])
    votes, semantic_refs = _semantic_votes(frames, semantic_items)
    # At least two distinct raster interpretations. A one-off caption isn't a motif.
    ranked = sorted((label for label, support in votes.items() if len({sha for _, sha in support}) >= 2),
                    key=lambda label: (-len({c for c, _ in votes[label]}), -len(votes[label]), label))
    motif_labels = {label for _, label in MOTIFS}
    motifs = [x for x in ranked if x in motif_labels][:2]
    looks = [x for x in ranked if x not in motif_labels][:1]
    if motifs or looks:
        wording = " and ".join(motifs) if motifs else "abstract forms"
        if looks:
            wording += " with " + looks[0]
        refs = set().union(*(semantic_refs[x] for x in motifs+looks))
        claim("The selected frames suggest " + wording + ".", "appearance", refs, "model interpretation")
    motion = defaultdict(list)
    motion_conditions = defaultdict(set)
    conditions = evidence.get("conditions", [])
    controls = [c for c in conditions if c.get("summary", {}).get("input_static")]
    motion_scope = controls or conditions
    for condition in motion_scope:
        for item in condition.get("summary", {}).get("motion", []):
            motion[item["label"]].extend(item.get("evidence", []))
            motion_conditions[item["label"]].add(condition["id"])
    labels = sorted(motion, key=lambda x: (-len(motion_conditions[x]), -len(motion[x]), x))
    active = [label for label in labels if "stationary" not in label]
    movement = (active or labels)[:3]
    motion_status = "measured" if labels else "unknown"
    if movement:
        noun = "With static inputs, output motion includes " if controls else "Measured output motion includes "
        if any("stationary" in x for x in movement):
            claim("Textured frames remain visually stationary in the sampled window.", "movement",
                  motion[movement[0]])
        else:
            claim(noun + ", ".join(movement) + ".", "movement", [ref for x in movement for ref in motion[x]])
    else:
        claim("The available frames do not establish a reliable movement pattern.", "movement", [], "unknown")
    pulses = [c for c in evidence.get("conditions", [])
              if c.get("summary", {}).get("brightness_period", {}).get("status") == "measured"]
    if pulses:
        claim("Brightness rises and falls in a recurring pulse.", "movement", [c["id"] for c in pulses])
        movement.append("brightness pulsing")
    if pulses:
        motion_status = "measured"
    artefacts = []
    whites = [f for f in frames if f["features"].get("near_white_fraction", 0) >= .9]
    blacks = [f for f in frames if f["features"].get("near_black_fraction", 0) >= .99 and f["seconds"] >= .5]
    if whites:
        claim("Some sampled frames wash out to almost pure white.", "artefacts", [f["id"] for f in whites])
        artefacts.append("white washouts")
    elif blacks and len(blacks)/len(frames) > .2:
        claim("Some sampled frames are almost completely black.", "artefacts", [f["id"] for f in blacks])
        artefacts.append("near-black intervals")
    else:
        losses = [f for f in frames if f.get("input_comparison", {}).get("input_edge_density", 0) > .04
                  and f["input_comparison"].get("edge_density_difference", 0) < -.04]
        if len(losses)/len(frames) > .5:
            claim("Fine source detail is strongly reduced in many tested frames.", "input_response",
                  [f["id"] for f in losses])
            artefacts.append("reduced source detail")
    # Avoid padding a short honest description with speculative effects.
    requested_missing = any(item.get("status") != "completed" or any(m.get("status") != "completed"
        for m in item.get("analysis", {}).get("models", [])) for item in semantic_items)
    semantic_status = "partial" if requested_missing else "sampled" if semantic_items else "not_run"
    status = "described" if motion_status == "measured" and semantic_status == "sampled" else "limited"
    if evidence.get("errors") or (plan or {}).get("conditions_without_semantic_sample"):
        status = "limited"
    return {"schema": "prismmoves.preset-description/1", "version": DESCRIPTION_VERSION,
            "summary": " ".join(clauses), "status": status, "motion_status": motion_status,
            "semantic_status": semantic_status, "claims": claims, "movement": movement,
            "effects": looks, "artefacts": artefacts,
            "style": list(dict.fromkeys([*colours, *motifs, *looks] +
                (["high contrast"] if contrast > .25 else []))),
            "scope": "Observed behaviour under the supplied input conditions and sampled time window.",
            "motion_scope": "static source controls" if controls else "output motion; moving-source contribution not isolated",
            "limits": ["No claims about beat synchronisation, flash safety, artistic intention or aesthetic quality.",
                       "Unseen phases and inputs may behave differently; model interpretations can be wrong.",
                       "Absence of a detected artefact is not certification that the renderer is artefact-free."]}


def synthesise_description(evidence, semantic_items=(), *, name="", engine="", plan=None):
    """Safe preset-level default: reference-fed appearance is not preset identity.

    Existing measurements are still available through synthesise_sample_description.
    Standalone/audio claims require the controlled_response API and its no-feed,
    repeated-intervention manifest. Captions cannot create missing causal evidence.
    """
    del semantic_items, name, engine, plan
    available = bool(evidence.get("frames"))
    summary = ("These reference-fed captures do not isolate the preset's standalone motion or its response to music. "
               "No image colours, objects or reference patterns are attributed to the preset; no-feed audio controls are needed."
               if available else "No usable rendered frames are available; standalone motion and audio response remain unknown.")
    return {"schema": "prismmoves.preset-description/1", "version": DESCRIPTION_VERSION,
            "summary": summary, "status": "limited" if available else "unavailable",
            "scope_policy": "preset-only", "motion_status": "unknown", "semantic_status": "excluded_from_preset_summary",
            "claims": [], "movement": [], "effects": [], "artefacts": [], "style": [],
            "scope": "Standalone preset effects; reference-conditioned observations are diagnostic only",
            "limits": ["The former capture protocol injected imagery continuously and pooled silence/music in its prose.",
                       "Use controlled no-feed captures to produce a standalone motion and audio-response description."]}
