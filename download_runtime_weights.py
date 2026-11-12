"""Restore the pinned public runtime and historical baseline weights."""
from pathlib import Path
import argparse,hashlib,json,urllib.request
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--mirror',default='https://huggingface.co');a=p.parse_args()
manifest=json.loads((root/'RUNTIME_MODELS.json').read_text(encoding='utf-8'))
for row in manifest['downloads']:
    target=root/row['path'];target.parent.mkdir(parents=True,exist_ok=True)
    if target.is_file():
        with target.open('rb') as existing:
            h=hashlib.file_digest(existing,'sha256').hexdigest()
        if h==row['sha256']:print('Already verified:',row['path']);continue
        raise SystemExit('Existing file failed hash: '+row['path']+'; preserve it and resolve manually.')
    url=a.mirror.rstrip('/')+'/'+row['repo']+'/resolve/'+row['revision']+'/'+row.get('filename','model.safetensors')
    tmp=target.with_suffix('.download');h=hashlib.sha256();n=0
    print('Downloading',row['path'])
    try:
        with urllib.request.urlopen(url,timeout=120) as response,tmp.open('wb') as f:
            while chunk:=response.read(1024*1024):
                n+=len(chunk)
                if n>row['bytes']:raise ValueError('Unexpected download size')
                h.update(chunk);f.write(chunk)
        if n!=row['bytes'] or h.hexdigest()!=row['sha256']:raise ValueError('Download checksum failed')
        tmp.replace(target)
    except Exception:
        if tmp.exists():tmp.unlink()
        raise
print('Runtime weights ready. Run verify_package.py.')
