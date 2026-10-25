from pathlib import Path
from zipfile import ZipFile
import json
P=Path(__file__).resolve().parent
final=P.parent/'实验3报告_算法优化实验.docx'
preserve=['word/header1.xml','word/webSettings.xml','word/footnotes.xml','word/endnotes.xml']
with ZipFile(P/'template-reference.docx') as ref:
    blobs={n:ref.read(n) for n in preserve}
with ZipFile(final) as src:
    content=[(info,blobs.get(info.filename,src.read(info.filename))) for info in src.infolist()]
temp=final.with_suffix('.preserve.docx')
with ZipFile(temp,'w') as out:
    for info,data in content:out.writestr(info,data)
temp.replace(final)
with ZipFile(P/'template-reference.docx') as a,ZipFile(final) as b:
    for n in preserve:assert a.read(n)==b.read(n)
    removed=[n for n in a.namelist() if n not in b.namelist()]
    assert not removed,removed
    evidence={'restored_preserve_only_parts':preserve,'removed_parts':removed,'unchanged_parts':[n for n in a.namelist() if a.read(n)==b.read(n)],
              'intentional_changes':[n for n in a.namelist() if a.read(n)!=b.read(n)]}
(P/'package-fidelity.json').write_text(json.dumps(evidence,ensure_ascii=False,indent=2),encoding='utf-8')
print('Preserved reference header and unchanged auxiliary parts.')
