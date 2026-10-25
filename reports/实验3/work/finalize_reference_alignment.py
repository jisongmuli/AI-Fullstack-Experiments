from pathlib import Path
from zipfile import ZipFile
from lxml import etree
P=Path(__file__).resolve().parent
f=P.parent/'实验3报告_算法优化实验.docx'
ns={'w':'http://schemas.openxmlformats.org/wordprocessingml/2006/main'}
with ZipFile(f) as z:
    items=[(i,z.read(i.filename)) for i in z.infolist()]
document=etree.fromstring(dict((i.filename,b) for i,b in items)['word/document.xml'])
count=0
for p in document.xpath('//w:p[w:pPr/w:pStyle[@w:val="Exp3Reference"]]',namespaces=ns):
    pp=p.find('w:pPr',ns);jc=pp.find('w:jc',ns)
    if jc is None:jc=etree.SubElement(pp,'{'+ns['w']+'}jc')
    jc.set('{'+ns['w']+'}val','left');count+=1
new=etree.tostring(document,encoding='UTF-8',xml_declaration=True,standalone=True)
t=f.with_suffix('.layout.docx')
with ZipFile(t,'w') as z:
    for i,b in items:z.writestr(i,new if i.filename=='word/document.xml' else b)
t.replace(f)
assert count==10,count
print('Aligned 10 references without changing other package parts.')
