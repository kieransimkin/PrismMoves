import copy
import math

import pytest

from prismmoves.audio_response import PROTOCOL, build_response, paired_effect
from prismmoves.controlled_response import validate_manifest, write_report, index
from prismmoves.preset_description import synthesise_description


def fixture(profiles=('silence', 'bass-pulses', 'mid-pulses', 'treble-pulses', 'rhythmic-mix'),
            trials=3, amplitude=.5, randomness=0, seed_control='paired', only_warmup=False):
    frames, conditions, metadata = [], [], []
    for profile in profiles:
        for trial in range(trials):
            cid = f'engine-only__{profile}__{trial}'
            events = [{'on': 2.5+trial*.04, 'off': 2.82+trial*.04},
                      {'on': 4.3+trial*.03, 'off': 4.62+trial*.03},
                      {'on': 6.5-trial*.04, 'off': 6.82-trial*.04},
                      {'on': 8.9+trial*.07, 'off': 9.22+trial*.07}]
            capture = []
            transitions = []
            for i in range(121):
                t = i/10
                pulse = any(e['on'] <= t < e['off']+.15 for e in events)
                if profile == 'rhythmic-mix':
                    pulse = 2.4 <= t < 10.2
                if only_warmup:
                    pulse = t < .4
                signal = amplitude * pulse if profile in ('bass-pulses', 'rhythmic-mix') else 0
                f = {'id': cid+':'+str(i*6), 'condition_id': cid, 'seconds': t, 'sha256': 'a'*64,
                     'features': {'mean_linear_luminance': .2+signal+randomness*math.sin(t*1.6+trial*2),
                                  'mean_saturation': .5, 'edge_density': .2, 'near_white_fraction': 0,
                                  'near_black_fraction': 0, 'hue_histogram_12': [0]*12}}
                frames.append(f)
                capture.append({'frame': i*6, 'output_seconds': t,
                                'output': {'path': 'not-read', 'sha256': 'a'*64}})
                if i:
                    transitions.append({'id': cid+':'+str((i-1)*6)+'->'+cid+':'+str(i*6),
                                        'a': cid+':'+str((i-1)*6), 'b': f['id'], 'status': 'measured',
                                        'rotation_radians_per_second': .2, 'log_scale_per_second': 0,
                                        'deformation_per_second': 0, 'pixel_change': .02})
            conditions.append({'id': cid, 'summary': {'motion': [{'label': 'clockwise rotation',
                                          'evidence': [t['id'] for t in transitions]}]}, 'transitions': transitions})
            metadata.append({'id': cid, 'protocol': PROTOCOL, 'state': 'captured', 'input_role': 'engine-only',
                             'input_mix': 0, 'source_pixels_sha256': 'b'*64, 'fresh_instance': True,
                             'trial': trial, 'render_fps': 60, 'capture_fps': 10, 'edge': 96,
                             'preset_sha256': 'd'*64, 'seed_control': seed_control,
                             'audio_mode': profile, 'processed_frames': 721, 'runtime': {'renderer': 'fixture'},
                             'stimulus': {'id': profile, 'kind': 'synthetic-music' if profile == 'rhythmic-mix' else 'synthetic-probe',
                                          'sha256': 'e'*64, 'events': events, 'trial': trial, 'rms': .12}, 'captures': capture})
    return {'frames': frames, 'conditions': conditions, 'errors': []}, {'protocol': PROTOCOL,
                'preset': {'name': 'Synthetic response fixture', 'engine': 'synthetic'},
                'conditions': metadata, 'expected_conditions': len(metadata)}


def test_separates_silence_music_and_frequency_probes():
    temporal, manifest = fixture()
    validate_manifest(manifest)
    report = build_response(temporal, manifest)
    assert 'In silence' in report['summary'] and 'clockwise rotation' in report['summary']
    assert 'rhythmic test music' in report['summary'] and '80 Hz pulses' in report['summary']
    bass = next(r for r in report['contrasts'] if r['probe']=='bass-pulses' and r['metric']=='brightness')
    treble = next(r for r in report['contrasts'] if r['probe']=='treble-pulses' and r['metric']=='brightness')
    assert bass['state']=='detected' and bass['onset_locked']
    assert treble['state']=='not_resolved'
    assert any(r['intervention_axis']=='frequency' for r in report['probe_comparisons'])


def test_independent_repeat_noise_blocks_false_audio_claim():
    temporal, manifest = fixture(amplitude=.08, randomness=.4, seed_control='not_available')
    report = build_response(temporal, manifest)
    assert all(r['state']!='detected' for r in report['contrasts'])
    assert 'not resolve a repeatable' in report['summary']


def test_frames_are_not_treated_as_independent_trials():
    temporal, manifest = fixture(trials=1)
    report = build_response(temporal, manifest)
    assert all(r['state']=='insufficient' for r in report['contrasts'])
    assert 'do not yet establish' in report['summary']


def test_reference_semantics_palette_and_effects_cannot_change_summary():
    temporal, manifest = fixture()
    original = build_response(temporal, manifest)
    for c in list(manifest['conditions']):
        d = copy.deepcopy(c)
        d['id'] = 'neutral:'+c['id']
        d['input_role'], d['input_mix'] = 'neutral-a', .15
        manifest['conditions'].append(d)
        for f in list(temporal['frames']):
            if f['condition_id']==c['id']:
                extra = copy.deepcopy(f)
                extra['condition_id'], extra['id'] = d['id'], 'neutral:'+f['id']
                extra['features'].update(mean_linear_luminance=1, mean_saturation=1, near_white_fraction=1)
                extra['caption'] = 'cyan concentric circles and a celebrity on an album cover'
                temporal['frames'].append(extra)
    manifest['preset']['name'] = 'HOT RED Rotating Laser Spiral'
    after = build_response(temporal, manifest)
    assert after['summary']==original['summary']
    assert after['claims']==original['claims']
    assert after['contrasts']==original['contrasts']
    assert after['input_transfers']


def test_default_summary_never_inherits_legacy_sample_content():
    temporal, _ = fixture()
    description = synthesise_description(temporal, [{'caption': 'turquoise rings'}], name='Clockwise glitter')
    assert description['scope_policy']=='preset-only'
    for term in ('turquoise', 'rings', 'clockwise', 'glitter'):
        assert term not in description['summary'].lower()
    assert description['style']==[] and description['movement']==[]


@pytest.mark.parametrize('fault', ['duplicate', 'reference', 'mixed-clock', 'mixed-stimulus', 'dirty-history'])
def test_provenance_faults_are_rejected(fault):
    _, manifest = fixture()
    if fault=='duplicate':
        manifest['conditions'].append(copy.deepcopy(manifest['conditions'][0]))
    elif fault=='reference':
        manifest['conditions'][0]['input_mix']=.35
    elif fault=='mixed-clock':
        manifest['conditions'][0]['captures'][2]['output_seconds'] += .1
    elif fault=='mixed-stimulus':
        manifest['conditions'][0]['stimulus']['trial']=2
    else:
        manifest['conditions'][0]['fresh_instance']=False
    with pytest.raises(ValueError):
        validate_manifest(manifest)


def test_missing_or_invalid_capture_is_never_a_positive_result():
    temporal, manifest = fixture()
    temporal['errors'].append({'condition_id': 'engine-only__bass-pulses__0', 'reason':'hash mismatch'})
    report = build_response(temporal, manifest)
    assert report['state']=='partial'
    assert all(r['state']=='insufficient' for r in report['contrasts'] if r['probe']=='bass-pulses')


def test_no_interpolation_through_sparse_or_short_data():
    assert paired_effect({0:0, 4:1}, {0:0, 4:0}) is None
    a = {i/10:0 for i in range(121)}
    b = {i/10:0 for i in range(121) if i<20 or i>90}
    assert paired_effect(a,b) is None


def test_changed_gpu_or_reference_invalidates_comparison():
    temporal, manifest = fixture()
    for c in manifest['conditions']:
        if c['audio_mode']=='bass-pulses':
            c['runtime']['renderer']='other-GPU'
    report = build_response(temporal,manifest)
    assert all(r['state']=='insufficient' for r in report['contrasts'] if r['probe']=='bass-pulses')


def test_warmup_is_not_characteristic_motion():
    temporal, manifest = fixture()
    for c in temporal['conditions']:
        c['summary']['motion'][0]['evidence'] = c['summary']['motion'][0]['evidence'][:3]
    report = build_response(temporal,manifest)
    assert 'clockwise rotation' not in report['summary']


def test_report_and_gallery_escape_names_keep_transfer_separate(tmp_path):
    temporal, manifest = fixture()
    report=build_response(temporal,manifest)
    report['preset']={'name':'<script>bad</script>','engine':'fixture'}
    from prismmoves.preset_pipeline import write_json
    folder=tmp_path/'fixture';folder.mkdir()
    write_json(folder/'response.json',report)
    write_report(folder,report)
    rows=index(tmp_path)
    assert len(rows)==1
    assert '<script>bad</script>' not in (tmp_path/'index.html').read_text()
    assert 'Reference-transfer experiments' in (folder/'report.html').read_text()


def test_frequency_claim_requires_equal_level_and_envelope():
    temporal, manifest = fixture()
    for c in manifest['conditions']:
        if c['audio_mode']=='mid-pulses':
            c['stimulus']['rms']=.06
    report=build_response(temporal,manifest)
    affected=[r for r in report['probe_comparisons'] if 'mid-pulses' in (r['probe'], r['compared_with'])]
    assert affected and all(r['state']=='insufficient' for r in affected)


def test_paired_seed_mismatch_cannot_claim_sound_response():
    temporal,manifest=fixture()
    for c in manifest['conditions']:
        c['seed']=c['trial']+(1 if c['audio_mode']=='bass-pulses' else 0)
    report=build_response(temporal,manifest)
    assert all(r['state']=='insufficient' for r in report['contrasts'] if r['probe']=='bass-pulses')


def test_direction_variation_is_not_claimed_as_within_run_reversal():
    temporal, manifest = fixture()
    for c in temporal['conditions']:
        if c['id'].endswith('__2'):
            c['summary']['motion'][0]['label']='counterclockwise rotation'
    from prismmoves.audio_response import _motion_phrase
    silent=[c for c in manifest['conditions'] if c['audio_mode']=='silence']
    # A repeatable direction in two trials takes precedence; not an invented oscillation.
    assert _motion_phrase(temporal,silent)=='clockwise rotation'
    for c in temporal['conditions']:
        motion=c['summary']['motion'][0]
        c['summary']['motion'].append(dict(motion,label='counterclockwise rotation'))
    assert 'rotation in both directions' in _motion_phrase(temporal,silent)


def test_black_silent_output_does_not_hide_observed_motion_under_music():
    temporal,manifest=fixture()
    for condition in temporal['conditions']:
        if '__silence__' in condition['id']:
            condition['summary']['motion']=[]
            condition['transitions']=[]
    for frame in temporal['frames']:
        if '__silence__' in frame['condition_id']:
            frame['features']['near_black_fraction']=1
            frame['features']['mean_linear_luminance']=0
    result=build_response(temporal,manifest)
    assert 'In silence, the sampled output remains almost entirely black.' in result['summary']
    assert 'With the rhythmic test music, it shows clockwise rotation.' in result['summary']
    assert any(claim['axis']=='music-motion' for claim in result['claims'])
