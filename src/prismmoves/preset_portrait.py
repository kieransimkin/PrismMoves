"""One evidence-linked paragraph: appearance, motion and controlled sound response.

No image decoding and no arbitrary model prose. Names, neutral-input observations
and reference captions are NOT inputs to the description.
"""
from __future__ import annotations
from collections import Counter, defaultdict
import math

from .audio_response import isolated, valid_condition, median, PROBE_WORDS, _effect_phrase
from .preset_description import HUES

PORTRAIT_VERSION = '1.0.0'
# Exact prompts avoid turning 'flat colours WITHOUT glow' into a glow claim.
PROMPT_LABELS = {
    'abstract concentric rings': ('shape', 'concentric rings'),
    'abstract radial flower-like geometry': ('shape', 'petal-like radial forms'),
    'a field of small glowing particles': ('shape', 'small luminous particles'),
    'a geometric grid or lattice': ('shape', 'grid-like geometry'),
    'striped or wave-like abstract patterns': ('shape', 'striped or wave-like forms'),
    'branching fractal patterns': ('shape', 'branching patterns'),
    'soft cloud-like abstract forms': ('shape', 'soft cloud-like forms'),
    'luminous neon outlines against darkness': ('style', 'neon-like outlines'),
    'soft diffuse glow': ('style', 'soft glow'),
    'a recognisable photographic image': ('style', 'photographic-looking imagery'),
    'flat even colours without glow': ('style', 'flat colour fields'),
    'sharp high-contrast black and white': ('style', 'high-contrast monochrome styling'),
    'smooth digital gradients': ('style', 'smooth gradients'),
    'grainy speckled texture': ('style', 'speckled texture'),
    'crisp thin lines': ('style', 'fine linework'),
    'soft blurred edges': ('style', 'soft-edged forms'),
    'blocky pixelated shapes': ('shape', 'blocky shapes'),
    'complex fine filigree': ('shape', 'intricate filigree'),
    'radially symmetric abstract geometry': ('structure', 'radial symmetry'),
    'bilaterally mirrored abstract geometry': ('structure', 'mirrored geometry'),
    'a repeating tiled abstract pattern': ('structure', 'repeating tiles'),
    'a layered geometric tunnel': ('shape', 'layered tunnel-like geometry'),
    'thin luminous filaments or ribbons': ('shape', 'luminous filaments or ribbons'),
    'faceted crystalline polygons': ('shape', 'crystalline facets'),
    'fluid marbled colour fields': ('shape', 'marbled colour fields'),
    'wireframe surfaces': ('shape', 'wireframe surfaces'),
    'kaleidoscopic wedges and mirrored facets': ('shape', 'kaleidoscopic facets'),
    'minimalist abstract visual design': ('aesthetic', 'minimalist styling'),
    'dense intricate geometric visual design': ('aesthetic', 'intricate geometric styling'),
    'organic fluid abstract visual design': ('aesthetic', 'an organic, fluid appearance'),
    'retro pixel-art visual design': ('aesthetic', 'a retro pixel-art appearance'),
    'luminous psychedelic abstract visual design': ('aesthetic', 'luminous psychedelic styling'),
    'restrained monochrome abstract visual design': ('aesthetic', 'restrained monochrome styling'),
}


def joined(words):
    words = list(dict.fromkeys(words))
    return words[0] if len(words) == 1 else ', '.join(words[:-1])+' and '+words[-1] if words else ''


def effect_words(rows):
    """Group common verbs without dropping their individual evidence records."""
    verbs = {}
    for row in rows:
        phrase = _effect_phrase(row).replace('fine edge structure', 'visible fine edges').replace('near-white washouts', 'near-white regions')
        verb, _, rest = phrase.partition(' ')
        verbs.setdefault(verb, []).append(rest)
    return joined(verb+' '+joined(objects) for verb, objects in verbs.items())


def isolated_evidence(temporal, manifest):
    bad = {e.get('condition_id') for e in temporal.get('errors', [])}
    conditions = [c for c in manifest['conditions'] if isolated(c) and valid_condition(c) and c['id'] not in bad]
    ids = {c['id'] for c in conditions}
    return conditions, [f for f in temporal['frames'] if f['condition_id'] in ids and f['seconds'] >= .5]


def visual_observations(frames, items):
    allowed = {f['sha256']: f for f in frames if f['features'].get('semantic_eligible')}
    votes, refs = defaultdict(set), defaultdict(set)
    for item in items:
        analysis = item.get('analysis', {})
        sha = analysis.get('source', {}).get('sha256')
        if sha not in allowed or item.get('status') != 'completed':
            continue
        frame = allowed[sha]
        for model in analysis.get('models', []):
            if model.get('name') != 'clip' or model.get('status') != 'completed':
                continue
            for axis, choices in model.get('result', {}).get('axes', {}).items():
                if axis not in {'motif','light','surface','structure','aesthetic'} or len(choices) < 2:
                    continue
                first, second = choices[:2]
                score = first.get('cosine_similarity', 0)
                other = second.get('cosine_similarity', 0)
                if not isinstance(score, (int,float)) or not isinstance(other,(int,float)):
                    continue
                if not math.isfinite(score-other) or score-other < .025:
                    continue
                label = PROMPT_LABELS.get(first.get('prompt'))
                if label:
                    votes[label].add(sha)
                    refs[label].add(frame['id'])
    supported = [k for k in votes if len(votes[k]) >= 2]
    supported.sort(key=lambda k: (-len(votes[k]), k))
    return [{'axis': k[0], 'label': k[1], 'evidence': sorted(refs[k]),
             'distinct_images': len(votes[k]), 'evidence_class': 'model interpretation'} for k in supported]


def motion_observation(temporal, conditions):
    summaries = {c['id']: c for c in temporal.get('conditions', [])}
    times = {f['id']: f['seconds'] for f in temporal.get('frames', [])}
    per_run, references = [], defaultdict(list)
    for condition in conditions:
        row = summaries.get(condition['id'], {})
        transitions = row.get('transitions', []) + row.get('lagged_transitions', [])
        good = {t['id'] for t in transitions if times.get(t.get('a'), -1) >= 2
                and times.get(t.get('b'), 99) <= 10.5
                and (t.get('status') == 'measured' or t.get('local', {}).get('status') == 'measured')}
        labels = set()
        for item in row.get('summary', {}).get('motion', []):
            refs = sorted(set(item.get('evidence', [])) & good)
            if len(refs) >= 4:
                labels.add(item['label']); references[item['label']].extend(refs)
        per_run.append(labels)
    counts = Counter(label for labels in per_run for label in labels)
    needed = max(2, math.ceil(len(conditions)*2/3))
    chosen = {label for label, count in counts.items() if count >= needed}
    words, used = [], []
    for a,b,both,varied in [
        ('clockwise rotation','counterclockwise rotation','rotation in both directions','rotation whose direction varies between trials'),
        ('outward expansion','inward contraction','alternating expansion and contraction','radial movement that varies between trials'),
        ('slow outward expansion','slow inward contraction','slow expansion and contraction','slow radial movement that varies between trials')]:
        if a in chosen and b in chosen:
            words.append(both if sum(a in r and b in r for r in per_run)>=needed else varied)
            chosen -= {a,b}; used += [a,b]
        elif a not in chosen and b not in chosen and counts[a] and counts[b] and counts[a]+counts[b]>=needed:
            words.append(varied);used += [a,b]
    selected = sorted(chosen, key=lambda label: (-counts[label],label))[:3]
    words += ['visually stationary textured output' if 'stationary' in w else w for w in selected]
    used += selected
    return joined(words[:3]), sorted({r for label in used for r in references[label]})


def compose_portrait(temporal, manifest, response, semantic_items=()):
    conditions, frames = isolated_evidence(temporal, manifest)
    claims, sentences = [], []
    def add(text, axis, refs, evidence_class='measurement'):
        if text:
            sentences.append(text)
            claims.append({'text': text, 'axis': axis, 'evidence': refs, 'evidence_class': evidence_class})
    if not frames:
        return {'summary': 'No verified reference-free output is available; style, colours, shapes, motion and audio response remain unresolved.',
                'claims': [], 'appearance': [], 'motion': {}, 'version': PORTRAIT_VERSION}
    by_mode = defaultdict(list)
    for c in conditions:
        by_mode[c['audio_mode']].append(c)
    music = 'music' if 'music' in by_mode else 'rhythmic-mix'
    music_ids = {c['id'] for c in by_mode.get(music, [])}
    # Appearance is scoped, not a pooled palette from all audio conditions.
    appearance_frames = [f for f in frames if f['condition_id'] in music_ids and 2 <= f['seconds'] <= 10.5
                         and f['features'].get('semantic_eligible')]
    scope = 'With the supplied music' if music == 'music' else 'With the rhythmic test music'
    if not appearance_frames:
        appearance_frames = [f for f in frames if f['features'].get('semantic_eligible')]
        scope = 'In the reference-free captures'
    # Uniform output still has a measurable palette, even though it needs no caption.
    if not appearance_frames:
        appearance_frames = [f for f in frames if f['condition_id'] in music_ids and 2 <= f['seconds'] <= 10.5] or frames
    if appearance_frames:
        groups = defaultdict(list)
        for f in appearance_frames: groups[f['condition_id']].append(f)
        def balanced(feature):
            return median(median(f['features'].get(feature) for f in rows) for rows in groups.values()) or 0
        hue = [sum(median(f['features'].get('hue_histogram_12',[0]*12)[i] for f in rows) or 0
                   for rows in groups.values())/len(groups) for i in range(12)]
        colours = [HUES[i] for i in sorted(range(12),key=lambda i:-hue[i])[:3] if hue[i]>.1]
        palette = joined(colours) if colours else 'largely monochrome tones'
        character = 'dark, ' if balanced('mean_linear_luminance') < .10 else 'bright, ' if balanced('mean_linear_luminance')>.55 else ''
        character += 'strongly saturated ' if balanced('mean_saturation')>.65 and colours else ''
        add(f'{scope}, it uses a {character}palette of {palette}.', 'colour-style', [f['id'] for f in appearance_frames])
    observations = visual_observations(frames, semantic_items)
    shapes = [o for o in observations if o['axis']=='shape'][:2]
    styles = [o for o in observations if o['axis']=='style'][:2]
    structures = [o for o in observations if o['axis']=='structure'][:1]
    aesthetics = [o for o in observations if o['axis']=='aesthetic'][:1]
    selected = shapes+styles+structures+aesthetics
    if selected:
        phrase = joined(o['label'] for o in shapes) or 'abstract forms'
        if styles: phrase += ' with '+joined(o['label'] for o in styles)
        if structures: phrase += ' and '+structures[0]['label']
        if aesthetics: phrase += ', with '+aesthetics[0]['label']
        add('Selected isolated frames suggest '+phrase+'.', 'appearance',
            sorted({r for o in selected for r in o['evidence']}), 'model interpretation')
    silent = by_mode.get('silence', [])
    sid = {c['id'] for c in silent}
    sf = [f for f in frames if f['condition_id'] in sid and 2 <= f['seconds'] <= 10.5]
    silence_motion, silence_refs = motion_observation(temporal, silent)
    music_motion, music_refs = motion_observation(temporal, by_mode.get(music, []))
    if sf and sum(f['features'].get('near_black_fraction',0)>.99 for f in sf) >= .95*len(sf):
        add('In silence, it remains almost entirely black.', 'silence', sorted(sid))
    elif silence_motion:
        add('In silence, it shows '+silence_motion+'.', 'silence', silence_refs)
    else:
        add('Its characteristic motion in silence remains unresolved.', 'silence', sorted(sid), 'unknown')
    if music_motion and music_motion != silence_motion:
        add(('With the supplied music' if music=='music' else 'With test music')+', it shows '+music_motion+'.', 'music-motion', music_refs)
    detected = [r for r in response['contrasts'] if r['state']=='detected' and r.get('input_role')=='engine-only']
    def effects(rows, maximum=3):
        order = {'brightness':0,'saturation':1,'white-coverage':2,'edge-density':3,'warping':4,'rotation-speed':5,'radial-motion':6,'image-change':7}
        rows = sorted(rows,key=lambda r:order.get(r['metric'],8))[:maximum]
        return effect_words(rows), [r['evidence'] for r in rows]
    mr = [r for r in detected if r['probe']==music]
    if mr:
        text,refs = effects(mr)
        add(('The supplied music' if music=='music' else 'The rhythmic test music')+' tends to '+text+'.','music',refs)
    elif music in by_mode:
        add('The music comparison did not resolve a repeatable change above run-to-run variation.','music',sorted(music_ids))
    probes = [p for p in ('bass-pulses','mid-pulses','treble-pulses') if p in by_mode]
    pr = [r for r in detected if r['probe'] in probes]
    # Group shared changes instead of arbitrarily selecting the two top tones.
    groups = defaultdict(list)
    for r in pr:
        value = r.get('onset_effect') if r.get('onset_locked') else r['effect']
        groups[(r['metric'],value>0,bool(r.get('onset_locked')))].append(r)
    ranked = sorted(groups.values(),key=lambda rows:(-len({r['probe'] for r in rows}),
                    0 if rows[0]['metric']=='brightness' else 1, -max(r.get('score',0) for r in rows)))
    consumed, response_groups = set(), {}
    for rows in ranked:
        if len(consumed)>=2: break
        metric = rows[0]['metric']
        if metric in consumed: continue
        consumed.add(metric)
        modes = tuple(sorted({r['probe'] for r in rows}, key=probes.index))
        key = (modes, bool(rows[0].get('onset_locked')))
        response_groups.setdefault(key, []).append(rows)
    for (modes, onset_locked), sets in response_groups.items():
        names = joined(PROBE_WORDS[mode] for mode in modes)
        phrase = effect_words([rows[0] for rows in sets])
        if onset_locked:
            text = names[0].upper()+names[1:]+' '+phrase+' in association with pulse onsets.'
        else:
            text = 'Across repeated trials, '+names+' '+phrase+' during exposure; individual onset timing is not established.'
        add(text,'audio-probes',[r['evidence'] for rows in sets for r in rows])
    comparisons = [r for r in response['probe_comparisons'] if r.get('intervention_axis')=='frequency']
    brightness = [r for r in comparisons if r['metric']=='brightness']
    if len(brightness)==3 and all(r['state']=='not_resolved' for r in brightness):
        add('A brightness preference among the three tested frequencies was not resolved.','frequency-specificity',[r['evidence'] for r in brightness])
    elif any(r['state']=='detected' for r in comparisons):
        row = max((r for r in comparisons if r['state']=='detected'),key=lambda r:r.get('score',0))
        add(PROBE_WORDS.get(row['probe'],'One probe').capitalize()+' produces a different '+row['metric'].replace('-',' ')+' response from '+PROBE_WORDS.get(row['compared_with'],'another probe')+'.','frequency-specificity',row['evidence'])
    white = [f for f in frames if f['features'].get('near_white_fraction',0)>.9]
    if white and not any(r['metric']=='white-coverage' for r in detected):
        add('Some isolated frames become almost entirely white.','visual-effects',[f['id'] for f in white])
    return {'version': PORTRAIT_VERSION, 'summary': ' '.join(sentences), 'claims': claims,
            'appearance': selected, 'motion': {'silence': silence_motion or None,'music': music_motion or None},
            'scope': 'Verified zero-reference-feed output only; appearance observations cannot establish audio causation.'}
