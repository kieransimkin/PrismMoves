"""Model-free end-to-end checks with actual PNGs, PixelCue and OpenCV."""
import gzip
import hashlib
import json
from pathlib import Path

from PIL import Image, ImageDraw
import pytest

from prismmoves.hybrid_response import analyse
from prismmoves.audio_response import PROTOCOL


def fixture(root):
    conditions=[]
    # Reuse identical hashes across trials while retaining each temporal binding.
    for mode in ('silence','bass-pulses','rhythmic-mix'):
        paths=[]
        for i in range(61):
            t=i/5
            active=mode!='silence' and 2<=t<=10.5
            p=root/f'{mode}-{i}.png'
            im=Image.new('RGB',(64,64),(3,3,8));d=ImageDraw.Draw(im)
            v=230 if active else 100
            for radius in (10,18,26):
                drift=i%6
                d.ellipse((32-radius+drift,32-radius,32+radius+drift,32+radius),outline=(v//3,v,v),width=2)
            im.save(p)
            paths.append({'frame':i*12,'output_seconds':t,'pcm_rms':.1 if active else 0,
                          'output':{'path':str(p),'sha256':hashlib.sha256(p.read_bytes()).hexdigest()}})
        for trial in range(3):
            c={'id':f'engine-only__{mode}__{trial}','protocol':PROTOCOL,'state':'captured',
                'trial':trial,'input_role':'engine-only','input_mix':0,'fresh_instance':True,
                'audio_mode':mode,'render_fps':60,'capture_fps':5,'processed_frames':721,
                'source_pixels_sha256':'b'*64,'preset_sha256':'c'*64,'edge':64,
                'seed_control':'paired','seed':4000+trial,'captures':paths,
                'stimulus':{'id':mode,'trial':trial,'sha256':'d'*64,'rms':.1,
                            'kind':'synthetic-music' if mode=='rhythmic-mix' else 'synthetic-probe',
                            'events':[{'on':x,'off':x+.3} for x in (2.5,4.5,6.5,8.5)],
                            'sample_rate':44100,'samples_per_channel':529200}}
            conditions.append(c)
    return {'protocol':PROTOCOL,'conditions':conditions,'expected_conditions':9,
            'preset':{'name':'Synthetic concentric-ring test fixture','id':'synthetic:hybrid-test','engine':'synthetic'}}


def fake_models(paths,**kwargs):
    # No network/model invocation: an explicit adapter seam tests report binding.
    return {'items':[{'status':'completed','analysis':{'source':{'sha256':hashlib.sha256(Path(p).read_bytes()).hexdigest()},'models':[
        {'name':'clip','status':'completed','result':{'axes':{'motif':[
            {'prompt':'abstract concentric rings','cosine_similarity':.8},
            {'prompt':'a geometric grid or lattice','cosine_similarity':.1}]}}},
        {'name':'blip','status':'completed','result':{'caption':'Synthetic fixture rings (stubbed interpretation)'}}]}} for p in paths]}


def test_end_to_end_restores_sheets_and_preserves_timeline(tmp_path):
    m=fixture(tmp_path)
    calls=[]
    def batch(paths,**kw):
        assert (tmp_path/'report'/'evidence.json.gz').exists()
        assert (tmp_path/'report'/'semantic-plan.json').exists()
        assert (tmp_path/'report'/'previews.json').exists()
        assert all(Path(p).stem==hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths)
        calls.append(len(paths));return fake_models(paths,**kw)
    r=analyse(m,tmp_path/'report',models_dir=tmp_path/'models',budget=8,batch_analyser=batch)
    assert r['performance']['measured_frames']==549
    assert sum(calls)<=8
    assert r['completion']['appearance']=='sampled'
    assert r['retention']['pcm_rms_frames']==549
    assert 'concentric rings' in r['summary'] and '\n' not in r['summary']
    previews=json.loads((tmp_path/'report'/'previews.json').read_text())
    assert len(previews['sheets'])>=10 and not previews['errors']
    assert 'contact-sheets' in (tmp_path/'report'/'report.html').read_text()
    with gzip.open(tmp_path/'report'/'evidence.json.gz','rt') as f:e=json.load(f)
    assert len(e['temporal']['frames'])==549
    assert len(e['temporal']['conditions'][0]['transitions'])==60
    assert e['conditions'][0]['capture_timeline'][0]['pcm_window']['zero_padding_samples']==1024


def test_corrupt_raster_does_not_destroy_previous_report(tmp_path):
    m=fixture(tmp_path);out=tmp_path/'report';out.mkdir();prior=b'{"sentinel":true}'
    (out/'response.json').write_bytes(prior)
    Path(m['conditions'][0]['captures'][0]['output']['path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='checksum'):analyse(m,out,budget=0)
    assert (out/'response.json').read_bytes()==prior


def test_no_semantics_still_has_previews_and_is_explicit(tmp_path):
    m=fixture(tmp_path)
    def forbidden(*args,**kwargs):raise AssertionError('Must not load any model')
    r=analyse(m,tmp_path/'out',models_dir=tmp_path/'models',budget=0,batch_analyser=forbidden)
    assert r['performance']['new_semantic_frames']==0
    assert r['completion']['appearance']=='disabled'
    assert (tmp_path/'out'/'previews.json').exists()


def test_unavailable_models_keep_measurements_and_provisional_report(tmp_path):
    m=fixture(tmp_path)
    def unavailable(paths,**kwargs):return {'items':[]}
    r=analyse(m,tmp_path/'out',models_dir=tmp_path/'models',budget=4,batch_analyser=unavailable)
    assert r['completion']['appearance']=='unavailable'
    assert r['completion']['description']=='limited'
    assert r['performance']['measured_frames']==549
    assert (tmp_path/'out'/'report.html').exists()
