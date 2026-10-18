"""Controlled measurements first; bounded appearance analysis and contact sheets second.

Reanalyse retained captures without rerendering. Never silently replace a capture,
import reference-fed captions, or claim missing models completed their work.
"""
from __future__ import annotations
import argparse
from collections import defaultdict
from contextlib import contextmanager
import gzip
import hashlib
import html
import json
import os
from pathlib import Path
import shutil
import time
import uuid

from . import controlled_response as controlled
from .preset_pipeline import (capture_records, canonical, digest_file, read_json,
                              semantic_identity, write_json)
from .preset_description import _distance
from .preset_portrait import compose_portrait, isolated_evidence

VERSION = '1.0.0'


def select_frames(temporal, manifest, budget=24):
    if type(budget) is not int or not 0 <= budget <= 64:
        raise ValueError('Semantic budget must be 0..64')
    conditions, frames = isolated_evidence(temporal, manifest)
    modes = {c['id']: c['audio_mode'] for c in conditions}
    eligible = [f for f in frames if f['features'].get('semantic_eligible')]
    groups = defaultdict(list)
    for f in eligible: groups[modes[f['condition_id']]].append(f)
    priority = ['silence','music','rhythmic-mix','bass-pulses','mid-pulses','treble-pulses']
    order = sorted(groups, key=lambda m: (priority.index(m) if m in priority else 20, m))
    selected, used = [], set()
    while len(selected) < budget:
        progress = False
        for mode in order:
            candidates = [f for f in groups[mode] if f['sha256'] not in used]
            if not candidates: continue
            local = [f for f in selected if modes[f['condition_id']]==mode]
            seen_conditions = {f['condition_id'] for f in local}
            def score(f):
                novelty = min((_distance(f,x) for x in local or selected), default=1)
                # Cover sound modes/trials before allowing one spectacular frame
                # to consume the batch; select output phases, never raw inputs.
                return (f['condition_id'] not in seen_conditions,
                        novelty, f['features'].get('contrast',0), -abs(f['seconds']-6))
            best = max(candidates,key=score)
            if local and min(_distance(best,x) for x in local) < .045:
                continue
            selected.append(best); used.add(best['sha256']); progress=True
            if len(selected)==budget: break
        if not progress: break
    return {'schema':'prismmoves.hybrid-selection/1', 'budget':budget,
            'captured_frames':len(temporal['frames']), 'eligible_isolated_frames':len(eligible),
            'selected':[{k:f[k] for k in ('id','sha256','condition_id','seconds')} for f in selected],
            'unique_inferences':len(selected), 'excluded_reference_conditions':len(manifest['conditions'])-len(conditions),
            'modes_without_sample':[m for m in order if m not in {modes[f['condition_id']] for f in selected}],
            'policy':'Exact-hash deduplication; mode/trial coverage and visual novelty; no near-duplicate caption propagation.'}


def sheet_groups(temporal, manifest, plan):
    groups, by_condition = [], defaultdict(list)
    for f in temporal['frames']: by_condition[f['condition_id']].append(f)
    for c in manifest['conditions']:
        rows = sorted(by_condition[c['id']],key=lambda f:f['seconds'])
        if not rows: continue
        picks = [min(rows,key=lambda f:abs(f['seconds']-t)) for t in (.5,2,4,6,8,10.5,12)]
        picks.append(max(rows,key=lambda f:f['features'].get('mean_linear_luminance',0)))
        events=c.get('stimulus',{}).get('events',[])
        if events:
            onset=events[len(events)//2]['on']
            picks += [min(rows,key=lambda f:abs(f['seconds']-t)) for t in (onset-.15,onset+.2,onset+.7)]
        isolated = c['input_role']=='engine-only' and c['input_mix']==0
        groups.append({'title': f"{c['audio_mode']} | trial {c['trial']+1} | " +
                       ('zero reference feed' if isolated else 'TEXTURE CONTROL - EXCLUDED FROM SUMMARY'),
                       'ids':list(dict.fromkeys(f['id'] for f in sorted(picks,key=lambda f:f['seconds']))), 'kind':'condition' if isolated else 'diagnostic'})
    if plan['selected']:
        groups.append({'title':'Exact appearance-analysis selections (only these images reach semantic models)',
                       'ids':[f['id'] for f in plan['selected']], 'kind':'semantic'})
    return groups


@contextmanager
def report_lock(out):
    out.mkdir(parents=True,exist_ok=True)
    lock=out/'.hybrid.lock'
    try:
        handle=lock.open('x',encoding='ascii')
    except FileExistsError as exc:
        raise ValueError(f'Another analysis or an interrupted run owns {lock}; check it before removing the lock.') from exc
    try:
        with handle: handle.write(str(os.getpid()))
        yield
    finally:
        lock.unlink(missing_ok=True)


def snapshot(out):
    entries=[p for p in out.iterdir() if p.name not in {'history','.hybrid.lock'}]
    if not entries:return
    history=out/'history'/str(time.time_ns());history.mkdir(parents=True)
    for p in entries:
        if p.is_symlink():raise ValueError('Report snapshots must not contain symbolic links')
        if p.is_dir():shutil.copytree(p,history/p.name)
        else:shutil.copy2(p,history/p.name)


def semantic_pass(plan, previews, out, root, *, budget, device, allow_download, batch_analyser=None):
    from pixelcue.visual_batch import analyse_batch
    models=('clip','blip') if budget else ()
    identity,pinned=semantic_identity(root,models,(),{'device':device,'semantic_edge':384,'hybrid_version':VERSION})
    prefix=hashlib.sha256(canonical(identity)).hexdigest()
    if not pinned:prefix+='-'+uuid.uuid4().hex
    cache=root/'analysis-cache'/'hybrid-appearance'/prefix;cache.mkdir(parents=True,exist_ok=True)
    files={f['id']:f for f in previews['keyframes']}
    items,missing=[],[]
    for f in plan['selected']:
        if f['id'] not in files:continue
        p=cache/(f['sha256']+'.json')
        try:receipt=read_json(p) if p.exists() else {}
        except (ValueError,OSError):receipt={}
        item=receipt.get('item',{})
        receipts=item.get('analysis',{}).get('models',[])
        if (receipt.get('identity')==identity and receipt.get('sha256')==f['sha256']
            and item.get('analysis',{}).get('source',{}).get('sha256')==f['sha256']
            and {m.get('name') for m in receipts}==set(models)
            and all(m.get('status')=='completed' for m in receipts)):
            items.append(item)
        else:missing.append(f)
    hits=len(items)
    if missing:
        print(f"Appearance analysis: {len(missing)} selected images, {hits} exact cache hits; model-major batch.",flush=True)
        # Content-addressed filenames carry no title, preset name or source hint.
        result=(batch_analyser or analyse_batch)([out/files[f['id']]['file'] for f in missing],
            models=models, models_dir=root, device=device, allow_download=allow_download,
            max_edge=384, prompt_profile='visualiser')
        expected={f['sha256'] for f in missing}
        for item in result.get('items',[]):
            sha=item.get('analysis',{}).get('source',{}).get('sha256')
            if sha not in expected:continue
            items.append(item)
            if all(m.get('status')=='completed' for m in item.get('analysis',{}).get('models',[])):
                write_json(cache/(sha+'.json'),{'identity':identity,'sha256':sha,'item':item})
    successful=sum({m.get('name') for m in i.get('analysis',{}).get('models',[]) if m.get('status')=='completed'}==set(models) for i in items)
    status=('disabled' if not budget else 'not_applicable' if not plan['selected'] else
            'sampled' if successful==len(plan['selected']) else 'partial' if successful else 'unavailable')
    return {'schema':'prismmoves.isolated-appearance/1','status':status,'items':items,
            'attempted_new_frames':len(missing),'cache_hits':hits,'complete_frames':successful,
            'cache_pinned':pinned,'limits':'Captions are retained observations, not motion or audio-causation evidence.'}


def enrich_reports(out, result, previews):
    controlled.write_report(out,result)
    gallery="<h2>Compare the paragraph with the captured output</h2><p>Each label records sound, trial and time. "\
            "Original selected PNGs are retained without resizing; contact sheets are previews.</p>"
    markdown=[]
    for s in previews['sheets']:
        gallery += f"<details open><summary>{html.escape(s['title'])}</summary><a href='{s['file']}'><img style='max-width:100%;height:auto' loading='lazy' src='{s['file']}' alt='{html.escape(s['title'],quote=True)}'></a></details>"
        markdown.append(f"![{s['title']}]({s['file']})")
    statuses=json.dumps(result['completion'],indent=2)
    gallery += "<details><summary>Capture, motion and appearance completion</summary><pre>"+html.escape(statuses)+"</pre></details>"
    gallery += "<p><a href='previews.json'>Preview and original-frame hashes</a> · <a href='appearance.json'>Selected model observations</a> · <a href='semantic-plan.json'>Selection plan</a></p>"
    p=out/'report.html';p.write_text(p.read_text(encoding='utf8').replace('<h2>Measured sound effects</h2>',gallery+'<h2>Measured sound effects</h2>'),encoding='utf8')
    p=out/'report.md';p.write_text(p.read_text(encoding='utf8')+'\n## Contact sheets\n\n'+'\n\n'.join(markdown)+'\n',encoding='utf8')


def analyse(captures, output, *, models_dir=None, budget=24, device='cpu', allow_download=False,
            analyser=None, batch_analyser=None):
    from pixelcue.visual_sequence import analyse_sequences
    from pixelcue.visual_contact_sheet import export_contact_sheets
    manifest=read_json(captures) if not isinstance(captures,dict) else captures
    controlled.validate_manifest(manifest)
    if type(budget) is not int or not 0<=budget<=64:raise ValueError('Semantic budget must be 0..64')
    records=capture_records(manifest);out=Path(output).resolve()
    root=Path(models_dir or Path.cwd()/'models').resolve();root.mkdir(parents=True,exist_ok=True)
    # Integrity first: no semantic claim or preview is built from substituted files.
    for c in manifest['conditions']:
        if c.get('state')!='captured':continue
        for ref in [c.get('source',{}),c.get('stimulus',{})]+[f['output'] for f in c.get('captures',[])]:
            if ref.get('path') and digest_file(ref['path'])!=ref.get('sha256'):
                raise ValueError('Retained capture/source/PCM checksum mismatch; original reports were not changed')
    with report_lock(out):
        snapshot(out)
        print('HYBRID PRESET ANALYSIS: numerical evidence -> contact sheets -> selected appearance -> one paragraph',flush=True)
        numeric=lambda rows,**kw:(analyser or analyse_sequences)(rows,**kw,multi_lag=True)
        result=controlled.analyse(manifest,out,analyser=numeric)
        with gzip.open(out/'evidence.json.gz','rt',encoding='utf8') as stream:evidence=json.load(stream)
        temporal=evidence['temporal']
        plan=select_frames(temporal,manifest,budget)
        write_json(out/'semantic-plan.json',plan)
        previews=export_contact_sheets(records,sheet_groups(temporal,manifest,plan),out)
        write_json(out/'previews.json',previews)
        # Persist a complete measurement-only report even if inference is interrupted.
        provisional=compose_portrait(temporal,manifest,result)
        result.update(summary=provisional['summary'],claims=provisional['claims'],portrait=provisional,
                      hybrid_version=VERSION, completion={'capture':result['state'],'motion':'measured_with_limits','appearance':'pending' if budget else 'disabled'})
        write_json(out/'response.json',result);enrich_reports(out,result,previews)
        appearance=semantic_pass(plan,previews,out,root,budget=budget,device=device,
                                 allow_download=allow_download,batch_analyser=batch_analyser)
        write_json(out/'appearance.json',appearance)
        portrait=compose_portrait(temporal,manifest,result,appearance['items'])
        result.update(summary=portrait['summary'],claims=portrait['claims'],portrait=portrait)
        transitions=[t for c in temporal.get('conditions',[]) for t in c.get('transitions',[])]
        global_count=sum(t.get('status')=='measured' for t in transitions)
        local_count=sum(t.get('local',{}).get('status')=='measured' for t in transitions)
        motion_resolved = bool(portrait['motion'].get('silence')) and bool(portrait['motion'].get('music'))
        result['completion']={'capture':result['state'],'appearance':appearance['status'],
            'motion':{'status':'measured' if motion_resolved else 'limited','global_accepted':global_count,'local_fallback_accepted':local_count,'transitions':len(transitions),
                      'global_coverage':global_count/max(1,len(transitions))},
            'contact_sheets':'partial' if previews['errors'] else 'completed',
            'description':'described' if appearance['status'] in {'sampled','not_applicable'} and not previews['errors'] and motion_resolved else 'limited'}
        result['performance'].update(new_semantic_frames=appearance['attempted_new_frames'],semantic_cache_hits=appearance['cache_hits'])
        result['retention']={'raw_captures':'retained in capture workspace, not reconstructed from this report',
            'selected_originals':len(previews['keyframes']),'contact_sheets':len(previews['sheets']),
            'pcm_rms_frames':sum('pcm_rms' in f for f in temporal['frames']),
            'frame_records':len(temporal['frames']), 'semantic_models':['clip','blip'] if budget else []}
        result['request']={'budget':budget,'device':device,'allow_download':allow_download,'version':VERSION}
        write_json(out/'response.json',result);enrich_reports(out,result,previews)
        print(result['summary'],flush=True)
        if appearance['status'] in {'unavailable','partial'}:
            print('APPEARANCE INCOMPLETE: inspect appearance.json. Numerical data and contact sheets are saved; models were not silently downloaded.',flush=True)
        return result


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command',choices=('analyse','reanalyse','index','status','verify'))
    parser.add_argument('--captures',type=Path);parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--models-dir',type=Path);parser.add_argument('--budget',type=int,default=24)
    parser.add_argument('--device',choices=('cpu','cuda','mps'),default='cpu')
    parser.add_argument('--allow-download',action='store_true');parser.add_argument('--limit',type=int,default=0)
    args=parser.parse_args(argv)
    try:
        if args.command=='index':controlled.index(args.out);return 0
        if args.command in {'status','verify'}:
            entries=list(args.out.glob('*/response.json'))
            if not entries:raise ValueError('No preset reports found in this catalogue')
            for p in entries:
                r=read_json(p);print(p.parent.name,r.get('completion',{'capture':r['state']}))
                if args.command=='verify':
                    if digest_file(p.parent/r['evidence']['file'])!=r['evidence']['sha256']:raise ValueError('Measurement archive checksum mismatch')
                    for f in read_json(p.parent/'previews.json')['keyframes']:
                        if digest_file(p.parent/f['file'])!=f['sha256']:raise ValueError('Original keyframe checksum mismatch')
            return 0
        if args.command=='reanalyse' and (args.out/'_work'/'run.lock').exists():
            raise ValueError('A capture sweep or interrupted sweep owns _work/run.lock; stop/check that process before reanalysis')
        jobs=[(args.captures,args.out)] if args.command=='analyse' else [(p,args.out/p.parent.name) for p in sorted((args.out/'_work').glob('*/captures.json'))]
        if not jobs or any(p is None for p,_ in jobs):raise ValueError('No retained capture manifests found; use the original local catalogue, not only its Git exports')
        for p,out in jobs[:args.limit or None]:
            analyse(p,out,models_dir=args.models_dir,budget=args.budget,device=args.device,allow_download=args.allow_download)
        if args.command=='reanalyse':controlled.index(args.out)
        return 0
    except (ValueError,OSError,KeyError) as exc:
        parser.exit(1,'Hybrid analysis stopped: '+str(exc)+'\n')


if __name__=='__main__':raise SystemExit(main())
