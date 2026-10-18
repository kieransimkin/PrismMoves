import hashlib

import pytest
from prismmoves.preset_description import select_semantic_frames,synthesise_description
from prismmoves.preset_pipeline import build_gallery, write_json


def frame(i,condition='a',value=.2,eligible=True):
    return dict(id=f'{condition}:{i}', condition_id=condition, seconds=i/10,
        sha256=hashlib.sha256(f'{condition}:{i}'.encode()).hexdigest(),
        features=dict(semantic_eligible=eligible,descriptor=[value]*192,hue_histogram_12=[0,0,0,0,1,0,0,0,0,0,0,0],
             edge_density=.1, contrast=.2,mean_linear_luminance=.08,mean_saturation=.9))


def semantic(f,caption='a luminous green ring',extra=None):
    models=[{'name':'blip','status':'completed','result':{'caption':caption}}]
    if extra: models.append(extra)
    return {'status':'completed','analysis':{'source':{'sha256':f['sha256']},'models':models}}


def test_selection_budget_and_condition_coverage():
    frames=[frame(i,c,i/30) for c in 'abcd' for i in range(30)]
    plan=select_semantic_frames({'frames':frames},budget=12)
    assert plan['unique_inferences']==12
    assert plan['conditions_covered']==list('abcd')
    assert len({f['sha256'] for f in plan['selected']})==12


def test_does_not_fill_budget_with_near_duplicates():
    frames=[frame(i,value=.2+i*.00001) for i in range(60)]
    plan=select_semantic_frames({'frames':frames})
    assert plan['unique_inferences']==1


def test_exact_duplicates_share_inference_keep_bindings():
    frames=[frame(i,c) for c in 'abc' for i in range(4)]
    for f in frames: f['sha256']='a'*64
    plan=select_semantic_frames({'frames':frames})
    assert plan['unique_inferences']==1 and len(plan['conditions_covered'])==3


def test_all_white_skips_semantics_retains_artifact():
    frames=[frame(i,eligible=False) for i in range(20)]
    for f in frames:f['features']['near_white_fraction']=1
    plan=select_semantic_frames({'frames':frames})
    d=synthesise_description({'frames':frames},[semantic(f,'a white striped rug') for f in frames],plan=plan)
    assert plan['unique_inferences']==0
    assert 'white washouts' in d['artefacts'] and 'rug' not in d['summary']
    assert 'striped' not in d['summary']


def test_caption_does_not_establish_motion_or_inherit_nsfw():
    frames=[frame(5),frame(6)]
    d=synthesise_description({'frames':frames},[semantic(f,'a rotating green ring NSFW') for f in frames])
    assert d['motion_status']=='unknown'
    assert 'rotating' not in d['summary'] and 'NSFW' not in d['summary']
    assert 'ring-like' in d['summary']


def test_one_caption_not_repeated_evidence():
    frames=[frame(5),frame(6)]
    d=synthesise_description({'frames':frames},[semantic(frames[0],'a green flower')])
    assert 'petal' not in d['summary']


def test_mixed_stationary_and_moving_not_reported_stationary():
    frames=[frame(5),frame(6)]
    conditions=[{'id':c,'summary':{'motion':[{'label':label,'evidence':['a:5->a:6']}]}}
                for c,label in [('a','clockwise rotation'),('b','visually stationary in the sampled window')]]
    d=synthesise_description({'frames':frames,'conditions':conditions})
    assert 'clockwise rotation' in d['summary'] and 'remain visually stationary' not in d['summary']


@pytest.mark.parametrize('budget',[-1,65,True,2.5])
def test_invalid_budget(budget):
    with pytest.raises(ValueError): select_semantic_frames({'frames':[]},budget=budget)


def test_atlas_scans_unindexed_reports_and_escapes_html(tmp_path):
    d=synthesise_description({'frames':[frame(5)]})
    for key in ['first','newly-committed']:
        write_json(tmp_path/key/'analysis.json',{'schema':'prismmoves.preset-report/1',
            'preset':{'id':key,'name':'<script>alert(1)</script>','engine':'projectm'},'description':d})
    rows=build_gallery(tmp_path)
    assert len(rows)==2 and '&lt;script&gt;' in (tmp_path/'index.html').read_text()
    assert '<script>alert(1)' not in (tmp_path/'index.html').read_text()


def test_motion_prefers_static_input_controls_over_moving_cover():
    conditions=[{'id':c,'summary':{'input_static':static,'motion':[{'label':label,'evidence':[c+':1->'+c+':2']}]}}
         for c,static,label in [('cover',False,'upward drift'),('grid',True,'clockwise rotation')]]
    d=synthesise_description({'frames':[frame(5),frame(6)],'conditions':conditions})
    assert 'clockwise' in d['summary'] and 'upward' not in d['summary']
    assert d['motion_scope']=='static source controls'
