import hashlib
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest
from prismmoves.preset_pipeline import analyse_capture_manifest

def capture(tmp_path):
    rows=[]
    a=np.random.default_rng(14).integers(0,255,(80,80,3),dtype=np.uint8)
    for i in range(18):
        image=Image.fromarray(np.roll(a,i,axis=1))
        path=tmp_path/f'{i}.png';image.save(path)
        rows.append({'frame':i,'output_seconds':i/15,
                     'output':{'path':str(path),'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}})
    return {'preset':{'id':'fixture:1','name':'Synthetic moving texture','engine':'fixture'},
            'conditions':[{'id':'a','captures':rows,'imagery':'grid-rings','audio_mode':'silence','sample':1}]}


def fake_batch(paths,**kwargs):
    return {'items':[{'status':'completed','analysis':{'source':{'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()},
        'models':[{'name':n,'status':'completed','result':{'caption':'a glowing green circle'}} for n in kwargs['models']]}}
        for p in paths]}


def test_pipeline_temporal_precedes_inference_and_caches(tmp_path):
    manifest=capture(tmp_path)
    root=tmp_path/'models';(root/'blip').mkdir(parents=True)
    (root/'blip'/'pixelcue-model.json').write_text('{"revision":"fixture"}')
    calls=[];out=tmp_path/'report'
    def spy(paths,**kwargs):
        assert (out/'temporal.json').is_file() and (out/'semantic-plan.json').is_file()
        calls.append(len(paths));return fake_batch(paths,**kwargs)
    r=analyse_capture_manifest(manifest,out,models=['blip'],models_dir=root,budget=4,batch_analyser=spy)
    assert calls and r['performance']['new_semantic_frames']<=4
    r2=analyse_capture_manifest(manifest,out,models=['blip'],models_dir=root,budget=4,batch_analyser=spy)
    assert len(calls)==1 and r2['performance']['cache_hits']==sum(calls)
    assert r2['description']['motion_status']=='measured'
    assert (out/'report.html').is_file()
    assert str(tmp_path) not in (out/'request.json').read_text()
    with pytest.raises(ValueError):
        analyse_capture_manifest(manifest,out,models=['blip'],models_dir=root,budget=5,batch_analyser=spy)


def test_incomplete_cache_does_not_count_as_success(tmp_path):
    manifest=capture(tmp_path);root=tmp_path/'models';(root/'blip').mkdir(parents=True)
    (root/'blip'/'pixelcue-model.json').write_text('{"revision":"fixture"}')
    out=tmp_path/'report'
    analyse_capture_manifest(manifest,out,models=['blip'],models_dir=root,budget=2,batch_analyser=fake_batch)
    for path in (root/'analysis-cache'/'preset-semantics').glob('*.json'):
        j=json.loads(path.read_text());j['item']['analysis']['models']=[];path.write_text(json.dumps(j))
    r=analyse_capture_manifest(manifest,out,models=['blip'],models_dir=root,budget=2,batch_analyser=fake_batch)
    assert r['performance']['cache_hits']==0


def test_changed_captures_not_hidden_by_resume(tmp_path):
    m=capture(tmp_path);out=tmp_path/'report'
    analyse_capture_manifest(m,out,models=[],models_dir=tmp_path/'models',budget=0)
    Path(m['conditions'][0]['captures'][3]['output']['path']).write_bytes(b'changed')
    r=analyse_capture_manifest(m,out,models=[],models_dir=tmp_path/'models',budget=0)
    assert r['temporal']['state']=='partial' and r['temporal']['errors']


def test_empty_capture_unknown_not_crash(tmp_path):
    r=analyse_capture_manifest({'preset':{'id':'failed'},'conditions':[]},tmp_path/'report',models=[],models_dir=tmp_path/'models')
    assert r['description']['status']=='unavailable'
