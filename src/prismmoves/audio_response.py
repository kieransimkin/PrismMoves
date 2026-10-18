"""Repeated-intervention evidence, not soundtrack correlation or object captions.

Consumes PixelCue measurements. The independent observations are render trials,
never individual frames. Thresholds are conservative engineering screens, not
p-values, calibrated confidence, or guarantees of causality on every GPU.
"""
from __future__ import annotations

from collections import Counter
import itertools
import math
import statistics

PROTOCOL = "framerudiments.audio-interventions/1"
VERSION = "1.0.0"
FEATURES = {
    "brightness": ("mean_linear_luminance", .02),
    "saturation": ("mean_saturation", .035),
    "edge-density": ("edge_density", .02),
    "white-coverage": ("near_white_fraction", .04),
}
MOTION = {"rotation-speed": ("rotation_radians_per_second", .06),
          "radial-motion": ("log_scale_per_second", .04),
          "warping": ("deformation_per_second", .015),
          "image-change": ("pixel_change", .01)}
FLOORS = {name: value[1] for name, value in {**FEATURES, **MOTION}.items()}
PROBE_WORDS = {"bass-pulses": "80 Hz pulses", "mid-pulses": "1 kHz pulses",
               "treble-pulses": "6 kHz pulses", "bass-quiet": "quieter 80 Hz pulses",
               "bass-fast": "faster 80 Hz pulses", "bass-sustain": "a sustained 80 Hz tone",
               "broadband-pulses": "broadband noise pulses", "rhythmic-mix": "the rhythmic test music",
               "music": "the supplied musical excerpt"}


def finite(value):
    return isinstance(value, (float, int)) and not isinstance(value, bool) and math.isfinite(value)


def median(values):
    values = [float(v) for v in values if finite(v)]
    return statistics.median(values) if values else None


def quantile(values, fraction=.95):
    values = sorted(v for v in values if finite(v))
    if not values:
        return 0.0
    return values[min(len(values)-1, math.ceil((len(values)-1)*fraction))]


def isolated(condition):
    return (condition.get("protocol") == PROTOCOL and condition.get("input_role") == "engine-only"
            and condition.get("input_mix") == 0 and condition.get("fresh_instance") is True)


def valid_condition(condition):
    return (condition.get("state") == "captured" and condition.get("fresh_instance") is True
            and condition.get("processed_frames") == round(12 * condition.get("render_fps", 0)) + 1
            and len(condition.get("captures", [])) == round(12 * condition.get("capture_fps", 0)) + 1)


def series(temporal, condition_id, metric):
    frames = {f["id"]: f for f in temporal.get("frames", []) if f["condition_id"] == condition_id}
    if metric in FEATURES:
        return {round(f["seconds"], 8): f["features"][FEATURES[metric][0]] for f in frames.values()
                if finite(f["features"].get(FEATURES[metric][0]))}
    condition = next((c for c in temporal.get("conditions", []) if c["id"] == condition_id), {})
    result = {}
    for transition in condition.get("transitions", []):
        # Pixel change is valid even when texture is inadequate for optical flow.
        if metric != "image-change" and transition.get("status") != "measured":
            continue
        value = transition.get(MOTION[metric][0])
        frame = frames.get(transition.get("b"))
        if frame and finite(value):
            result[round(frame["seconds"], 8)] = abs(value) if metric == "rotation-speed" else value
    return result


def paired_effect(a, b, *, events=()):
    """Difference of means in the exposure minus pre-exposure difference.

    No interpolation through missing observations. A 70% coverage requirement
    prevents a few trackable flashes from standing for an entire condition.
    """
    times = sorted(a.keys() & b.keys())
    if len(times) < 30:
        return None
    pre = [t for t in times if .5 <= t < 2.0]
    post = [t for t in times if 2.0 <= t <= 10.5]
    # Use actual observed cadence; missing gaps cannot reduce required coverage.
    gap = median(b-a for a, b in zip(times, times[1:]))
    if gap is None or gap <= 0 or gap > .25:
        return None
    if len(pre) < .7 * 1.5/gap or len(post) < .7 * 8.5/gap:
        return None
    mean = lambda values: statistics.fmean(values)
    baseline = mean(a[t]-b[t] for t in pre)
    effect = mean(a[t]-b[t] for t in post)-baseline
    # Onset-locked contrast is separate from average brightness/flow differences.
    responses, delays = [], []
    for event in events:
        before = [t for t in times if event["on"]-.4 <= t <= event["on"]-.08]
        after = [t for t in times if event["on"]+.04 <= t <= event["on"]+.6]
        if len(before) < 3 or len(after) < 4:
            continue
        local = mean(a[t]-b[t] for t in before)
        response = mean(a[t]-b[t] for t in after)-local
        responses.append(response)
        peak = max(after, key=lambda t: abs(a[t]-b[t]-local))
        delays.append(peak-event["on"])
    return {"effect": effect, "pre_difference": baseline,
            "onset_effects": responses, "peak_delays_seconds": delays,
            "coverage_frames": len(post), "sample_interval_seconds": gap}


def compatible(a, b):
    # Deliberately exclude semantic text, preset names, and source filenames.
    keys = ("input_role", "input_mix", "source_pixels_sha256", "render_fps", "capture_fps", "edge",
            "preset_sha256", "protocol")
    if not all(a.get(k) == b.get(k) and a.get(k) is not None for k in keys):
        return False
    # Renderer/driver changes invalidate cross-condition causal comparisons.
    for key in ("renderer", "opengl", "version", "engine", "library_sha256"):
        if a.get("runtime", {}).get(key) != b.get("runtime", {}).get(key):
            return False
    return True


def paired_protocol(a, b, axis=None):
    if a.get("seed_control") != b.get("seed_control"):
        return False
    if a.get("seed_control") == "paired" and a.get("seed") != b.get("seed"):
        return False
    if axis == "frequency":
        x, y = a.get("stimulus", {}), b.get("stimulus", {})
        if (not finite(x.get("rms")) or not finite(y.get("rms"))
                or not math.isclose(x["rms"], y["rms"], rel_tol=1e-5)
                or not x.get("events") or x["events"] != y.get("events")):
            return False
    return True


def contrast(temporal, treatment, controls, metric, *, label, minimum_trials=3, axis=None):
    pairs, ids = [], []
    for active in treatment:
        base = next((c for c in controls if c["trial"] == active["trial"] and compatible(active, c) and paired_protocol(active, c, axis)), None)
        if base is None:
            continue
        result = paired_effect(series(temporal, active["id"], metric), series(temporal, base["id"], metric),
                               events=active.get("stimulus", {}).get("events", []))
        if result:
            pairs.append(result)
            ids.append([active["id"], base["id"]])
    record = {"probe": label, "metric": metric, "trials": len(pairs), "evidence": ids,
              "state": "insufficient", "effect": None, "noise_floor": None}
    if len(pairs) < minimum_trials:
        return record
    nulls = []
    for a, b in itertools.combinations(controls, 2):
        if not compatible(a, b):
            continue
        sa, sb = series(temporal, a["id"], metric), series(temporal, b["id"], metric)
        values = [abs(sa[t]-sb[t]) for t in sa.keys() & sb.keys() if .5 <= t <= 10.5]
        nulls.extend(values)
    effects = [p["effect"] for p in pairs]
    effect = median(effects)
    spread = median(abs(v-effect) for v in effects)
    # Conservative repeat noise plus effect-size floor; no frames-as-replicates.
    noise = max(FLOORS[metric], quantile(nulls), 2.5 * spread)
    support = sum(v * effect > 0 and abs(v) > noise for v in effects) / len(effects)
    state = "detected" if abs(effect) > 2 * noise and support >= 2/3 else "not_resolved"
    event_values = [v for p in pairs for v in p["onset_effects"]]
    event = median(event_values)
    event_support = sum(v * (event or 0) > 0 and abs(v) > noise for v in event_values)/max(1, len(event_values))
    # Repeated onsets may reveal a short-lived effect lost in a whole-window mean.
    event_trials = sum(len(p["onset_effects"]) >= 3 and
                       abs(median(p["onset_effects"]) or 0) > 2*noise for p in pairs)
    locked = (len(event_values) >= 3*minimum_trials and event is not None and
              abs(event) > 2*noise and event_support >= .75 and event_trials >= minimum_trials)
    if locked and state != "detected":
        state = "detected"
    record.update(state=state, effect=effect, trial_effects=effects, noise_floor=noise,
                  support_fraction=support, onset_locked=locked, onset_effect=event,
                  onset_support_fraction=event_support,
                  peak_delay_seconds=median(d for p in pairs for d in p["peak_delays_seconds"]) if locked else None,
                  sample_interval_seconds=max(p["sample_interval_seconds"] for p in pairs),
                  score=max(abs(effect), abs(event or 0) if locked else 0)/noise,
                  limits="Engineering repeatability screen, not a significance test or a universal frequency response")
    return record


def _motion_phrase(temporal, conditions):
    summaries = {c["id"]: c.get("summary", {}) for c in temporal.get("conditions", [])}
    times = {f["id"]: f["seconds"] for f in temporal.get("frames", [])}
    transitions = {c["id"]: c.get("transitions", []) for c in temporal.get("conditions", [])}
    labels = Counter()
    per_run = []
    for condition in conditions:
        supported = {t["id"] for t in transitions.get(condition["id"], [])
                     if t.get("status") == "measured" and times.get(t.get("a"), -1) >= 2.0}
        items = {m["label"] for m in summaries.get(condition["id"], {}).get("motion", [])
                 if len(set(m.get("evidence", [])) & supported) >= max(4, .3*len(supported))}
        per_run.append(items)
        labels.update(items)
    required = max(2, math.ceil(len(conditions)*2/3))
    chosen = {label for label, n in labels.items() if n >= required}
    phrases = []
    for a, b, both, varied in [
        ("clockwise rotation", "counterclockwise rotation", "rotation in both directions", "rotation whose direction varies between runs"),
        ("outward expansion", "inward contraction", "expansion and contraction", "radial motion that varies between runs")]:
        if a in chosen and b in chosen:
            phrases.append(both if sum(a in r and b in r for r in per_run) >= required else varied)
            chosen -= {a, b}
        elif not ({a, b} & chosen) and labels[a] and labels[b] and labels[a]+labels[b] >= required:
            phrases.append(varied)
    phrases.extend(sorted(x for x in chosen if "stationary" not in x))
    if not phrases and chosen:
        return "visually stationary textured output"
    return ", ".join(phrases[:3]) if phrases else None


def _effect_phrase(result):
    value = result["onset_effect"] if result.get("onset_locked") else result["effect"]
    positive = value > 0
    return {"brightness": "increase brightness" if positive else "reduce brightness",
            "saturation": "increase colour saturation" if positive else "reduce colour saturation",
            "edge-density": "increase fine edge structure" if positive else "reduce fine edge structure",
            "white-coverage": "increase near-white washouts" if positive else "reduce near-white washouts",
            "rotation-speed": "speed up rotation" if positive else "slow rotation",
            "radial-motion": "shift radial motion outward" if positive else "shift radial motion inward",
            "warping": "increase local warping" if positive else "reduce local warping",
            "image-change": "increase frame-to-frame change" if positive else "reduce frame-to-frame change"}[result["metric"]]


def build_response(temporal, manifest):
    declared = manifest.get("conditions", [])
    good = [c for c in declared if valid_condition(c)]
    # A data-integrity failure is not usable evidence for a positive claim.
    bad_ids = {e.get("condition_id") for e in temporal.get("errors", [])}
    good = [c for c in good if c["id"] not in bad_ids]
    alone = [c for c in good if isolated(c)]
    results, transfers = [], []
    for role in sorted({c["input_role"] for c in good}):
        rows = [c for c in good if c["input_role"] == role]
        controls = [c for c in rows if c["audio_mode"] == "silence"]
        for probe in sorted({c["audio_mode"] for c in rows} - {"silence"}):
            active = [c for c in rows if c["audio_mode"] == probe]
            target = results if role == "engine-only" else transfers
            for metric in FLOORS:
                target.append(dict(contrast(temporal, active, controls, metric, label=probe), input_role=role))
    specificity = []
    for active_id, control_id, axis in (
        ("bass-pulses", "mid-pulses", "frequency"),
        ("bass-pulses", "treble-pulses", "frequency"),
        ("mid-pulses", "treble-pulses", "frequency"),
        ("bass-pulses", "bass-quiet", "level"),
        ("bass-fast", "bass-pulses", "pulse-rate"),
        ("bass-sustain", "bass-pulses", "envelope"),
    ):
        active = [c for c in alone if c["audio_mode"] == active_id]
        controls = [c for c in alone if c["audio_mode"] == control_id]
        if active and controls:
            for metric in FLOORS:
                comparison = contrast(temporal, active, controls, metric, label=active_id, axis=axis)
                comparison.update(compared_with=control_id, intervention_axis=axis)
                specificity.append(comparison)
    silent = [c for c in alone if c["audio_mode"] == "silence"]
    clauses, claims = [], []
    silent_motion = _motion_phrase(temporal, silent) if len(silent) >= 3 else None
    def append(text, axis, refs):
        clauses.append(text)
        claims.append({"text": text, "axis": axis, "evidence": refs})
    if len(silent) < 3:
        append("The available captures do not yet establish the preset's standalone behaviour in silence.", "silence", [])
    else:
        motion = silent_motion
        silent_ids = {c["id"] for c in silent}
        frames = [f for f in temporal.get("frames", []) if f["condition_id"] in silent_ids and f["seconds"] >= 2]
        black = median(f["features"].get("near_black_fraction", 0) for f in frames)
        if black is not None and black >= .99 and sum(f["features"].get("near_black_fraction", 0) >= .99 for f in frames) >= .95 * len(frames):
            text = "In silence, the sampled output remains almost entirely black."
        elif motion:
            text = "In silence, it shows " + motion + "."
        else:
            text = "In silence, no reliable motion pattern was resolved in the sampled output."
        append(text, "silence", sorted(silent_ids))
    music_ids = [c for c in alone if c.get("stimulus", {}).get("kind") in {"music", "synthetic-music"}]
    music_probes = {c["audio_mode"] for c in music_ids}
    music_motion = {}
    for probe in sorted(music_probes):
        active = [c for c in music_ids if c["audio_mode"] == probe]
        if len(active) >= 3:
            motion = _motion_phrase(temporal, active)
            music_motion[probe] = {"motion": motion, "evidence": [c["id"] for c in active]}
    # An untrackable black silent frame must not prevent describing measured
    # motion under music. This is an observation, not a numeric speed-change claim.
    for probe in ("music", "rhythmic-mix"):
        observation = music_motion.get(probe, {})
        if observation.get("motion") and observation["motion"] != silent_motion:
            words = "the supplied music" if probe == "music" else "the rhythmic test music"
            append("With " + words + ", it shows " + observation["motion"] + ".", "music-motion", observation["evidence"])
            break
    music_results = [r for r in results if r["probe"] in music_probes and r["state"] == "detected"]
    if music_results:
        specific = [r for r in music_results if r["metric"] != "image-change"] or music_results
        best = sorted(specific, key=lambda r: -r["score"])[0]
        words = "The supplied music" if best["probe"] == "music" else "The rhythmic test music"
        verb, rest = _effect_phrase(best).split(" ", 1)
        append(words + " " + verb + "s " + rest + ".", "music", best["evidence"])
    elif music_ids:
        append("The music comparison did not resolve a repeatable change above run-to-run variation.", "music", [c["id"] for c in music_ids])
    else:
        append("A matched musical-input comparison is not available yet.", "music", [])
    selected = []
    for probe in sorted({r["probe"] for r in results} - music_probes):
        available = [r for r in results if r["probe"] == probe and r["state"] == "detected"]
        if available:
            available = [r for r in available if r["metric"] != "image-change"] or available
            selected.append(max(available, key=lambda r: r["score"]))
    for result in sorted(selected, key=lambda r: -r["score"])[:2]:
        word = PROBE_WORDS.get(result["probe"], "The tested audio")
        append(word[0].upper()+word[1:]+" repeatedly " + _effect_phrase(result) + ".", "audio", result["evidence"])
    if not selected and any(r["state"] != "insufficient" and r["probe"] not in music_probes for r in results):
        append("The isolated frequency probes did not resolve a repeatable response at the tested level.", "audio", [])
    # Source transfer experiments are deliberately not inputs to the paragraph.
    return {"schema": "prismmoves.audio-response/1", "version": VERSION,
            "summary": " ".join(clauses), "claims": claims, "contrasts": results,
            "input_transfers": transfers, "probe_comparisons": specificity,
            "isolated_conditions": len(alone), "music_motion": music_motion,
            "state": "completed" if len(alone) >= 6 and all(valid_condition(c) for c in declared) and not bad_ids else "partial",
            "description_scope": "Zero reference-image contribution only; independent silence and audio trials",
            "limits": ["Not a subjective quality, flash-safety, beat-synchronisation or whole-frequency-band guarantee.",
                       "Synthetic tones are not proof of response to drums, bass instruments, vocals or stems.",
                       "No detected change means not resolved in this experiment, not proof of audio insensitivity.",
                       "Unseeded projectM uses empirical repeat controls; it does not acquire a seed guarantee.",
                       "Optical flow can be unavailable during strong colour changes or weak texture; unavailable is not stationary."]}
