"""Stream and check shipped file fingerprints without loading pickle models."""
from pathlib import Path
import hashlib,json,sys
root=Path(__file__).resolve().parent
manifest=json.loads((root/'FILE_MANIFEST.json').read_text(encoding='utf-8'))
failed=[];missing=[]
for row in manifest['files']:
    path=root/row['path']
    if not path.is_file():missing.append(row['path']);continue
    h=hashlib.sha256()
    with path.open('rb') as f:
        for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
    if path.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:failed.append(row['path'])
print(json.dumps({'checked':len(manifest['files']),'missing':missing,'modified':failed},ensure_ascii=False,indent=2))
sys.exit(1 if missing or failed else 0)
