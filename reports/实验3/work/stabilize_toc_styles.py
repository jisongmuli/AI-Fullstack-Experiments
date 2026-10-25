from pathlib import Path
from zipfile import ZipFile
from lxml import etree
P=Path(__file__).resolve().parent
f=P.parent/'实验3报告_算法优化实验.docx'
W='http://schemas.openxmlformats.org/wordprocessingml/2006/main';N={'w':W}
def child(parent,tag,**attrs):
 e=parent.find('{'+W+'}'+tag)
 if e is None:e=etree.SubElement(parent,'{'+W+'}'+tag)
 for k,v in attrs.items():e.set('{'+W+'}'+k,str(v))
 return e
with ZipFile(f) as z:items=[(i,z.read(i.filename)) for i in z.infolist()]
data={i.filename:b for i,b in items};root=etree.fromstring(data['word/styles.xml'])
for identifier in ['TOC1','TOC2']:
 s=root.xpath(f'./w:style[@w:styleId="{identifier}"]',namespaces=N)[0]
 rp=child(s,'rPr');child(rp,'rFonts',ascii='Times New Roman',hAnsi='Times New Roman',eastAsia='宋体')
 child(rp,'sz',val=21);child(rp,'szCs',val=21);child(rp,'color',val='000000')
 pp=child(s,'pPr');child(pp,'spacing',line=300,lineRule='exact',before=0,after=0)
 child(pp,'snapToGrid',val=0);child(pp,'keepLines',val=1)
for identifier in ['TOC11','TOC21']:
 for s in root.xpath(f'./w:style[@w:styleId="{identifier}"]',namespaces=N):root.remove(s)
new=etree.tostring(root,xml_declaration=True,encoding='UTF-8',standalone=True)
t=f.with_suffix('.toc.docx')
with ZipFile(t,'w') as z:
 for i,b in items:z.writestr(i,new if i.filename=='word/styles.xml' else b)
t.replace(f)
print('Native TOC styles stabilized for later field refresh.')
