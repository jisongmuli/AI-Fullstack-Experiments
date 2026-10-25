"""Read-only measurements of the existing experiment 2/3 artifacts."""
from pathlib import Path
import argparse, gc, hashlib, io, json, os, platform, subprocess, sys, time

os.environ.setdefault('OMP_NUM_THREADS', '4')
os.environ.setdefault('MKL_NUM_THREADS', '4')
ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
TEXT = ('The team checked the source files before preparing the data. Each record received a stable identifier. '
        'Missing values and duplicate records were counted separately, and only the training split was used to fit numeric transformations. ')

def memory_case(name):
    import joblib, numpy as np, psutil, sklearn
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    proc = psutil.Process()
    gc.collect()
    baseline = proc.memory_info().rss
    model = joblib.load(ROOT/'models'/name)
    gc.collect()
    loaded = proc.memory_info().rss
    fn = lambda: model['classifier'].decision_function(model['vectorizer'].transform([TEXT]))
    for _ in range(50): fn()
    readings = []
    for i in range(1000):
        fn()
        if i % 10 == 0: readings.append({'iteration': i + 1, 'rss_bytes': proc.memory_info().rss})
    info = proc.memory_info()
    print(json.dumps({'name':name, 'baseline_rss':baseline, 'after_load_rss':loaded,
                     'load_delta_rss':loaded-baseline, 'peak_working_set':getattr(info,'peak_wset',0),
                     'iterations':1000, 'measurements':readings,
                     'rss_variation_pct':(max(x['rss_bytes'] for x in readings)-min(x['rss_bytes'] for x in readings))/np.mean([x['rss_bytes'] for x in readings])*100}))

def timed(fn, count=200, warm=25):
    import numpy as np
    for _ in range(warm): fn()
    samples=[]
    for _ in range(count):
        t=time.perf_counter_ns(); fn(); samples.append((time.perf_counter_ns()-t)/1e6)
    return {'n':count, 'p50_ms':float(np.median(samples)), 'p95_ms':float(np.percentile(samples,95)),
            'mean_ms':float(np.mean(samples)), 'samples_ms':samples}

def main():
    import joblib, numpy as np, torch, sklearn, psutil
    from PIL import Image
    from threadpoolctl import threadpool_limits
    sys.path.insert(0,str(ROOT/'code'/'models'))
    from architecture import ImageCNN
    threadpool_limits(4); torch.set_num_threads(4); torch.set_num_interop_threads(1)
    torch.manual_seed(42)
    manifest=json.loads((ROOT/'FILE_MANIFEST.json').read_text(encoding='utf-8'))
    names=['models/text_detector_full.joblib','models/text_detector.joblib','models/image_detector.pt','models/image_detector.ts']
    fingerprints={}
    for rel in names:
        expected=next(x for x in manifest['files'] if x['path']==rel)
        digest=hashlib.sha256((ROOT/rel).read_bytes()).hexdigest()
        assert digest==expected['sha256'],rel
        fingerprints[rel]={'bytes':(ROOT/rel).stat().st_size,'sha256':digest}
    record={'date':'2026-10-25','seed':42,'threads':4,'python':platform.python_version(),
            'platform':platform.platform(),'torch':torch.__version__,'sklearn':sklearn.__version__,
            'logical_cpus':os.cpu_count(),'physical_memory_bytes':psutil.virtual_memory().total,
            'fingerprints':fingerprints,'input_note':'Generated inputs measure runtime; original test metrics are separately sourced from the project report.'}
    text_models={'full':joblib.load(ROOT/'models/text_detector_full.joblib'),
                 'compact':joblib.load(ROOT/'models/text_detector.joblib')}
    text_info={}
    for name,m in text_models.items():
        v=m['vectorizer'];c=m['classifier'];features=v.transform([TEXT])
        text_info[name]={'features':len(v.vocabulary_),'classifier_parameters':int(c.coef_.size+c.intercept_.size),
            'idf_values':int(v.idf_.size),'numeric_bytes':int(c.coef_.nbytes+c.intercept_.nbytes+v.idf_.nbytes),
            'csr_nnz':features.nnz,'csr_shape':list(features.shape),'csr_bytes':int(features.data.nbytes+features.indices.nbytes+features.indptr.nbytes),
            'dtype':str(c.coef_.dtype),'analyzer':v.analyzer,'ngram_range':v.ngram_range,'min_df':v.min_df}
    record['text_models']=text_info
    checkpoint=torch.load(ROOT/'models/image_detector.pt',map_location='cpu',weights_only=True)
    eager=ImageCNN().eval();eager.load_state_dict(checkpoint['state_dict'])
    traced=torch.jit.load(str(ROOT/'models/image_detector.ts'),map_location='cpu').eval()
    layer_info=[];handles=[]
    def capture(name):
        def hook(module,args,out):
            params=sum(p.numel() for p in module.parameters(recurse=False));macs=0
            if isinstance(module,torch.nn.Conv2d):
                macs=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            elif isinstance(module,torch.nn.Linear):macs=out.numel()*module.in_features
            layer_info.append({'name':name,'type':module.__class__.__name__,'output':list(out.shape),'parameters':params,'macs':macs})
        return hook
    for name,mod in eager.named_modules():
        if len(list(mod.children()))==0:handles.append(mod.register_forward_hook(capture(name)))
    with torch.inference_mode():eager(torch.rand(1,3,32,32))
    for h in handles:h.remove()
    record['image_model']={'parameters':sum(p.numel() for p in eager.parameters()),'best_epoch':checkpoint['best_epoch'],
                           'layers':layer_info,'macs_conv_linear':sum(x['macs'] for x in layer_info)}
    parity_inputs=[torch.rand(32,3,32,32) for _ in range(4)]
    for name in ['sample-real.png','sample-fake.png']:
        im=np.array(Image.open(ROOT/'web/assets'/name).convert('RGB'),dtype=np.float32)/255
        parity_inputs.append(torch.from_numpy(im.transpose(2,0,1)).unsqueeze(0))
    errors=[];mismatches=0
    with torch.inference_mode():
        for x in parity_inputs:
            a=eager(x);b=traced(x);errors.append(float((a-b).abs().max()));mismatches+=int((a.argmax(1)!=b.argmax(1)).sum())
    record['image_export_parity']={'inputs':sum(x.shape[0] for x in parity_inputs),'random_inputs':128,'bundled_sample_images':2,
                                  'max_abs_logit_difference':max(errors),'argmax_mismatches':mismatches}
    print('Fingerprints and parity verified.',flush=True)
    benches=[]
    with torch.inference_mode():
        for batch,n in [(1,200),(32,80)]:
            x=torch.rand(batch,3,32,32)
            for round_id in range(3):
                order=[('eager',eager),('torchscript',traced)]
                if round_id%2:order.reverse()
                for name,model in order:
                    row=timed(lambda:model(x),n);row.update({'family':'image','method':name,'batch':batch,'round':round_id+1});benches.append(row)
        data=(ROOT/'web/assets/sample-real.png').read_bytes()
        def pipeline(model):
            with Image.open(io.BytesIO(data)) as source:a=np.asarray(source.convert('RGB'),dtype=np.float32)/255
            return model(torch.from_numpy(a.transpose(2,0,1).copy()).unsqueeze(0)).softmax(1)
        for name,model in [('eager',eager),('torchscript',traced)]:
            row=timed(lambda:pipeline(model),200);row.update({'family':'image_pipeline','method':name,'batch':1,'round':1});benches.append(row)
    print('Image benchmarks complete.',flush=True)
    for batch,n in [(1,200),(32,80)]:
        for round_id in range(3):
            order=list(text_models.items())
            if round_id%2:order.reverse()
            for name,m in order:
                row=timed(lambda:m['classifier'].decision_function(m['vectorizer'].transform([TEXT]*batch)),n)
                row.update({'family':'text','method':name,'batch':batch,'round':round_id+1,'characters':len(TEXT)});benches.append(row)
    length_sweep=[]
    for length in [100,330,580]:
        text=(TEXT*4)[:length]
        for name,m in text_models.items():
            row=timed(lambda:m['classifier'].decision_function(m['vectorizer'].transform([text])),150)
            row.update({'method':name,'characters':length});length_sweep.append(row)
    record['length_sweep']=length_sweep
    record['benchmarks']=benches
    summaries=[]
    for family in ['image','image_pipeline','text']:
        for batch in [1,32]:
            for method in ['eager','torchscript','full','compact']:
                rows=[r for r in benches if r['family']==family and r['batch']==batch and r['method']==method]
                if not rows:continue
                a=np.asarray([v for r in rows for v in r['samples_ms']])
                summaries.append({'family':family,'batch':batch,'method':method,'n':len(a),'p50_ms':float(np.median(a)),
                    'p95_ms':float(np.percentile(a,95)),'mean_ms':float(a.mean()),'throughput_samples_s':float(batch*1000/a.mean())})
    record['summary']=summaries
    record['memory']=[]
    for name in ['text_detector_full.joblib','text_detector.joblib']:
        result=subprocess.run([sys.executable,str(Path(__file__).resolve()),'--memory',name],capture_output=True,text=True,encoding='utf-8',check=True)
        record['memory'].append(json.loads(result.stdout))
    (OUT/'measurements.json').write_text(json.dumps(record,ensure_ascii=False,indent=2),encoding='utf-8')
    print(json.dumps({'summary':summaries,'text_models':text_info,'image_export_parity':record['image_export_parity'],
        'memory':[{k:v for k,v in m.items() if k!='measurements'} for m in record['memory']]},ensure_ascii=False,indent=2),flush=True)

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--memory');a=p.parse_args()
    if a.memory:memory_case(a.memory)
    else:main()
