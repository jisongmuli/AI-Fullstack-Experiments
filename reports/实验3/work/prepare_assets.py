from pathlib import Path
import hashlib,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.font_manager import FontProperties
from docx import Document
from zipfile import ZipFile
from PIL import Image, ImageOps, ImageDraw

P=Path(__file__).resolve().parent
M=json.loads((P/'measurements.json').read_text(encoding='utf-8'))
FIG=P/'figures';FIG.mkdir(exist_ok=True)
font=FontProperties(fname=r'C:/Windows/Fonts/simsun.ttc')
plt.rcParams.update({'font.family':font.get_name(),'font.size':10.5,'axes.unicode_minus':False,
                     'axes.spines.top':False,'axes.spines.right':False,'figure.dpi':180})
C=['#36566F','#7796A8','#B7C8D1','#D9E2E7']
def save(fig,name):
    fig.tight_layout(pad=1.0);fig.savefig(FIG/name,dpi=240,bbox_inches='tight');plt.close(fig)
def label(ax,bars,fmt='{:.2f}'):
    for b in bars: ax.text(b.get_x()+b.get_width()/2,b.get_height(),fmt.format(b.get_height()),ha='center',va='bottom',fontsize=10.5)

fig,ax=plt.subplots(figsize=(5.65,2.35));x=np.arange(2);w=.34
a=ax.bar(x-w/2,[479938/1024,571026/1024],w,label='优化前',color=C[0]);b=ax.bar(x+w/2,[167326/1024,597526/1024],w,label='优化后',color=C[1]);label(ax,a,'{:.1f}');label(ax,b,'{:.1f}')
ax.set_xticks(x,['文本特征压缩','图像推理导出']);ax.set_ylabel('文件大小 KiB');ax.set_ylim(0,720);ax.legend(frameon=False,ncol=2);save(fig,'storage.png')

layers=[x for x in M['image_model']['layers'] if x['type']=='Conv2d']
fig,ax=plt.subplots(figsize=(5.65,2.5));bars=ax.barh([f'卷积 {i+1}' for i in range(5)],[x['macs']/1e6 for x in layers],color=C[0]);ax.invert_yaxis();ax.set_xlabel('单张图像计算量 百万 MAC');ax.set_xlim(0,11.5)
for b in bars:ax.text(b.get_width()+.1,b.get_y()+b.get_height()/2,f'{b.get_width():.3f}',va='center')
save(fig,'conv_macs.png')

def summary(family,method,batch):return next(x for x in M['summary'] if x['family']==family and x['method']==method and x['batch']==batch)
fig,axes=plt.subplots(1,2,figsize=(5.65,2.65))
for ax,fam,methods,names in [(axes[0],'image',['eager','torchscript'],['原生','导出']), (axes[1],'text',['full','compact'],['完整','压缩'])]:
    x=np.arange(2);a=ax.bar(x-.18,[summary(fam,k,1)['p50_ms'] for k in methods],.36,color=C[0],label='P50');b=ax.bar(x+.18,[summary(fam,k,1)['p95_ms'] for k in methods],.36,color=C[1],label='P95');label(ax,a,'{:.3f}');label(ax,b,'{:.3f}')
    ax.set_xticks(x,names);ax.set_ylabel('延迟 ms');ax.set_title('图像模型' if fam=='image' else '文本全流程',fontsize=11);ax.set_ylim(0,.85);ax.legend(frameon=False,fontsize=9)
save(fig,'latency.png')

fig,axes=plt.subplots(1,2,figsize=(5.65,2.6))
for ax,fam,methods,names in [(axes[0],'image',['eager','torchscript'],['原生','导出']), (axes[1],'text',['full','compact'],['完整','压缩'])]:
    x=np.arange(2);a=ax.bar(x-.18,[summary(fam,k,1)['throughput_samples_s'] for k in methods],.36,color=C[0],label='batch 1');b=ax.bar(x+.18,[summary(fam,k,32)['throughput_samples_s'] for k in methods],.36,color=C[1],label='batch 32');label(ax,a,'{:.0f}');label(ax,b,'{:.0f}')
    ax.set_xticks(x,names);ax.set_ylabel('吞吐量 样本每秒');ax.set_ylim(0,12500);ax.legend(frameon=False,fontsize=10.5,loc='upper center',bbox_to_anchor=(.5,1.17),ncol=2)
save(fig,'throughput.png')

fig,ax=plt.subplots(figsize=(5.65,2.6))
for row,color,name in zip(M['memory'],C,['完整词表','压缩词表']):
    ax.plot([x['iteration'] for x in row['measurements']],[x['rss_bytes']/1024**2 for x in row['measurements']],label=name,color=color,lw=1.8)
ax.set_xlabel('连续推理次数');ax.set_ylabel('进程工作集 MiB');ax.set_xlim(0,1000);ax.set_ylim(110,140);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);save(fig,'memory.png')

fig,ax=plt.subplots(figsize=(5.65,2.6))
for method,color,name in [('full',C[0],'完整词表'),('compact',C[1],'压缩词表')]:
    rows=[x for x in M['length_sweep'] if x['method']==method]
    ax.plot([x['characters'] for x in rows],[x['p50_ms'] for x in rows],marker='o',label=name,color=color,lw=1.5)
ax.set_xlabel('输入英文文本字符数');ax.set_ylabel('P50 延迟 ms');ax.set_xticks([100,330,580]);ax.set_ylim(.14,.44);ax.legend(frameon=False);ax.grid(axis='y',alpha=.2);save(fig,'length.png')

cm=np.array([[352,48],[34,366]]);rates=cm/cm.sum(axis=1,keepdims=True)*100
fig,ax=plt.subplots(figsize=(4.4,3.0));im=ax.imshow(rates,cmap='Blues',vmin=0,vmax=100)
for i in range(2):
 for j in range(2):ax.text(j,i,f'{cm[i,j]} 张\n{rates[i,j]:.2f}%',ha='center',va='center',color='white' if rates[i,j]>70 else '#111111',fontsize=12)
ax.set_xticks([0,1],['真实图像','AI 图像']);ax.set_yticks([0,1],['真实图像','AI 图像']);ax.set_xlabel('预测类别');ax.set_ylabel('实际类别');fig.colorbar(im,ax=ax,label='该实际类别占比 %');save(fig,'confusion.png')

fig,ax=plt.subplots(figsize=(5.65,2.7));ax.set_xlim(0,10);ax.set_ylim(0,4);ax.axis('off')
nodes=[(.1,2.45,2.6,'实验一固定划分\n训练 4000 图像\n训练 4855 原始文本'),(3.7,2.45,2.6,'实验二已训练模型\n五层卷积网络\n字符 TF-IDF 分类器'),(7.2,2.45,2.6,'实验三优化\nTorchScript 导出\n词表 30000 → 10000'),(7.2,.25,2.6,'部署验证\n延迟与吞吐量\n内存与输出一致性'),(3.7,.25,2.6,'质量核查\n验证集宏 F1\n原测试集误差分布'),(.1,.25,2.6,'结果归档\n模型指纹与测量日志\n图表和实验报告')]
for x,y,w,text in nodes:
    ax.add_patch(plt.Rectangle((x,y),w,1.1,fill=False,ec='#667680',lw=.8));ax.text(x+w/2,y+.55,text,ha='center',va='center',fontsize=10.5)
for a,b in [((2.7,3),(3.7,3)),((6.3,3),(7.2,3)),((8.5,2.45),(8.5,1.35)),((7.2,.8),(6.3,.8)),((3.7,.8),(2.7,.8))]:ax.annotate('',xy=b,xytext=a,arrowprops={'arrowstyle':'->','color':'#333333','lw':1})
save(fig,'workflow.png')

ref=P/'template-reference.docx';doc=Document(ref)
with ZipFile(ref) as z:
    inventory=[{'path':n,'bytes':len(z.read(n)),'sha256':hashlib.sha256(z.read(n)).hexdigest()} for n in z.namelist()]
(P/'template-inventory.json').write_text(json.dumps(inventory,indent=2),encoding='utf-8')
contract=f'''# 实验三报告模板执行约定

参考文件 {ref}，SHA-256 {hashlib.sha256(ref.read_bytes()).hexdigest()}，原模板两页，一节。
来源旧版 DOC 已由本机 Word 转为 DOCX 保留参照，原 DOC 未改动。

## 页面及样式

A4 纵向，宽 11906、高 16838 twip。上、下边距 1440 twip，左、右 1800 twip。
页眉距边缘 851 twip，页脚 992 twip。保留一节及原页眉。
封面沿用原学校名、课程名、题目、组号、组员三列表、时间的排列与26磅校名和课程名。
六名组员使原五行成员区增加一行，允许删减题目和组号之间一个空行以保证单页封面。
课程名使用 Title 样式，黑色无下划线；题目保持算法优化实验并增加描述性副题。
正文中文宋体、英文 Times New Roman，小四12磅，行距固定20磅。正文首行缩进24磅，段后3磅。
一级节标题沿用原模板14磅加粗风格，黑色；改用语义 Heading 1 以支持目录。
二级标题12磅加粗，三级标题12磅加粗。标题无装饰线或边框。
图题、表题及图表文字五号10.5磅，图表限制在14.65厘米正文宽度内。
表格沿用封面三列表格风格，正文表格按数据类型拓展2至5列；重复表头，行高自动，禁止断行切分。

## 内容及槽位

word/document.xml 的封面前14段及第一张表为封面槽位，填写第四组六名成员和实验三职责。
模板正文所有黄色指导语、空白占位与原目录式要求为可替换槽位。
正文按实验目的、实验环境、实验内容与步骤、关键技术与难点、结果与分析、总结与展望、参考文献七章展开。
允许克隆正文、标题、图题、表格模式以容纳不少于20页报告。
封面后增加目录页；页脚增加连续 PAGE 字段，长报告可读性需要该扩展。
原模板中的编号格式通过显式标题文本统一，移除旧自动编号以免双重编号。

## 包部件

基线见 template-inventory.json。document.xml、styles.xml、settings.xml、numbering.xml、document.xml.rels、内容类型与核心/应用属性属于可编辑部件。
新建媒体、页脚和相关关系为本报告新增部件。原 theme、fontTable、webSettings、header 及其关系等其余不受影响部件保留。
Word 最终更新目录和分页字段，允许其正常保存更新字段缓存及应用属性；原参照文件保持指纹一致。

## 验证

原参照已渲染为 template-page-1.png 与 template-page-2.png，封面和正文两种模式均已检查。
最终用 Word 分页和 PDF 导出，接入原 render_docx.py 的 PNG 渲染流程逐页检查。
检查页数至少20，封面单页，标题与首段同页，图表清晰，无孤立表头、裁切和空白页；目录页码与实际章起始页相符。
''' 
(P/'artifact.md').write_text(contract,encoding='utf-8')
print('Figures:',len(list(FIG.glob('*.png'))),'template contract saved')
