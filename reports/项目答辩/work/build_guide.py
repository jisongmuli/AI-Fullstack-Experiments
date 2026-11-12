"""Build a novice-oriented project handbook from reviewed sources."""
from pathlib import Path
import json
import math
import re
import sys

from PIL import Image, ImageDraw, ImageFont
from docx import Document
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_CELL_VERTICAL_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from docx.opc.constants import RELATIONSHIP_TYPE as RT

WORK = Path(__file__).resolve().parent
OUT = WORK.parent / "内容可信工作台项目详解与答辩手册_李阳.docx"
FIG = WORK / "figures"
FIG.mkdir(exist_ok=True)
FONT = "C:/Windows/Fonts/msyh.ttc"
COLORS = {"ink": "#142d3b", "teal": "#176e65", "light": "#eaf4f1", "gray": "#edf0f2", "line": "#a8b6bc"}


def canvas(name, size):
    im = Image.new("RGB", size, "white")
    return im, ImageDraw.Draw(im)


def text(draw, point, value, size=34, fill=COLORS["ink"], anchor="mm"):
    draw.multiline_text(point, value, font=ImageFont.truetype(FONT, size), fill=fill,
                        anchor=anchor, align="center", spacing=12)


def box(draw, rect, value, size=34, color=None):
    draw.rounded_rectangle(rect, radius=16, fill=color or COLORS["light"], outline=COLORS["line"], width=2)
    text(draw, ((rect[0]+rect[2])/2, (rect[1]+rect[3])/2), value, size)


def arrow(draw, start, end):
    draw.line([start, end], fill=COLORS["teal"], width=5)
    angle = math.atan2(end[1]-start[1], end[0]-start[0])
    wings = [(end[0]-20*math.cos(angle+delta), end[1]-20*math.sin(angle+delta)) for delta in (-.48, .48)]
    draw.polygon([end, *wings], fill=COLORS["teal"])


def save(im, name):
    im.save(FIG / f"{name}.png")


def diagrams():
    im,d=canvas("architecture",(1800,940))
    for y,label in [(60,"浏览器交互层"),(265,"API 与服务层"),(510,"模型执行层"),(750,"项目文件层")]:
        text(d,(50,y+55),label,36,anchor="lm")
    box(d,(360,30,1735,185),"图像  文本  文档  编辑  课程实验  大模型助手",40)
    arrow(d,(1040,190),(1040,235))
    box(d,(360,240,1735,420),"FastAPI 与 Uvicorn\n输入校验  任务调度  报告缓存  工具编排",38)
    arrow(d,(1040,425),(1040,470))
    for x,t in [(360,"课程 CNN\n中文 TF-IDF"),(830,"DEAR 图像检测\nMPU 文本检测"),(1300,"Qwen3.5 + LoRA\n工具选择与汇总")]:
        box(d,(x,480,x+435,655),t,34)
    for x in (575,1045,1515):arrow(d,(x,660),(x,710))
    box(d,(360,720,1735,895),"data 数据  models 权重与配置  results 实验记录\ncode 与 web 源码  reports 与 docs 文档",35,COLORS["gray"])
    save(im,"architecture")
    im,d=canvas("splits",(1800,690))
    box(d,(440,20,1360,150),"来源核验  清洗去重  按原始组划分",40)
    for x in (300,900,1500):arrow(d,(900,160),(x,230))
    for x,t in [(30,"训练集\n学习模型参数"),(630,"验证集\n选择配置与校准"),(1230,"测试集\n统计最终表现")]:
        box(d,(x,240,x+540,430),t,40)
    arrow(d,(300,440),(620,535));arrow(d,(900,440),(820,535));arrow(d,(1500,440),(1200,535))
    box(d,(350,550,1450,670),"保存权重  配置  来源指纹与逐项结果",38,COLORS["gray"])
    save(im,"splits")
    im,d=canvas("cnn",(1800,760))
    items=[(40,"RGB 输入\n3 × 64 × 64"),(630,"卷积块一\n16 × 32 × 32"),(1220,"卷积块二\n32 × 16 × 16")]
    for x,t in items:box(d,(x,50,x+540,240),t,39)
    arrow(d,(585,145),(620,145));arrow(d,(1175,145),(1210,145))
    arrow(d,(1490,250),(1490,375))
    for x,t in [(1220,"卷积块三\n64 × 8 × 8"),(630,"全局平均池化\n64 个通道值"),(40,"线性分类头\n2 个类别 logits")]:box(d,(x,390,x+540,580),t,39)
    arrow(d,(1210,485),(1175,485));arrow(d,(620,485),(585,485))
    text(d,(900,685),"剪枝后通道数 16 / 32 / 64 变为 12 / 24 / 48",38)
    save(im,"cnn")
    im,d=canvas("tiles",(1800,760))
    d.rectangle((110,40,1330,690),outline=COLORS["ink"],width=3)
    for x in (110,620,1130):
        for y in (40,315,590):d.rectangle((x,y,x+200,y+100),fill=COLORS["light"],outline=COLORS["teal"],width=4)
    text(d,(720,210),"原图上的分布式采样\n采样之间保留未进入模型的区域",34)
    text(d,(1575,220),"最多 9 块\n原生尺度\n逐块计算",34)
    arrow(d,(1345,430),(1430,430));box(d,(1440,360,1770,570),"按面积加权\n汇总 logits\n再校准",34)
    text(d,(720,730),"4K 输入采样逻辑示意  非真实检测输出",30)
    save(im,"tiles")
    im,d=canvas("windows",(1800,710))
    for y,left,right,label in [(60,100,1200,"窗口一"),(255,290,1390,"窗口二"),(450,480,1580,"窗口三")]:
        box(d,(left,y,right,y+135),label+"  最多 512 token",38)
        if y>60:
            d.rectangle((left,y,left+160,y+135),fill="#c7e1dc",outline=COLORS["teal"],width=2)
            text(d,(left+80,y+67),"重叠\n64",28)
    text(d,(920,660),"先覆盖全文  再按新增 token 数加权汇总",38)
    save(im,"windows")
    im,d=canvas("agentflow",(1800,910))
    for x,t in [(40,"操作要求\n输入可用情况"),(630,"Qwen + LoRA\n选择计划字母"),(1220,"服务端校验\n允许工具与依赖")]:box(d,(x,35,x+540,215),t,36)
    arrow(d,(585,125),(620,125));arrow(d,(1175,125),(1210,125));arrow(d,(1490,225),(1490,345))
    for x,t in [(1220,"真实 API 调用\n图像  文本  文档等"),(630,"紧凑工具记录\n路径  状态  结果"),(40,"基础模型汇总\n答复与工具详情展示")]:box(d,(x,365,x+540,575),t,35)
    arrow(d,(1210,465),(1175,465));arrow(d,(620,465),(585,465))
    text(d,(900,770),"路由适配器负责工具选择  汇总阶段关闭路由适配器\n所有工具调用都使用固定 API 地址",36)
    save(im,"agentflow")


def element(tag, **attrs):
    node=OxmlElement(tag)
    for key,value in attrs.items():node.set(qn(key),str(value))
    return node


def font(run,size=12,bold=False,mono=False,color="000000"):
    run.font.name="Consolas" if mono else "Arial"
    run.font.size=Pt(size);run.font.bold=bold;run.font.color.rgb=RGBColor.from_string(color)
    prop=run._element.get_or_add_rPr();rf=prop.find(qn("w:rFonts"))
    if rf is None:rf=element("w:rFonts");prop.insert(0,rf)
    for name,value in {"w:ascii":run.font.name,"w:hAnsi":run.font.name,"w:eastAsia":"宋体"}.items():rf.set(qn(name),value)
    for name in list(rf.attrib):
        if name.endswith("Theme"):del rf.attrib[name]


def style_font(style,size,bold=False):
    style.font.name="Arial";style.font.size=Pt(size);style.font.bold=bold;style.font.italic=False;style.font.color.rgb=RGBColor(0,0,0)
    prop=style.element.get_or_add_rPr();rf=prop.find(qn("w:rFonts"))
    if rf is None:rf=element("w:rFonts");prop.insert(0,rf)
    rf.set(qn("w:eastAsia"),"宋体");rf.set(qn("w:ascii"),"Arial");rf.set(qn("w:hAnsi"),"Arial")
    for name in list(rf.attrib):
        if name.endswith("Theme"):del rf.attrib[name]
    pp=style.element.find(qn("w:pPr"))
    if pp is not None:
        for border in list(pp.findall(qn("w:pBdr"))):pp.remove(border)


def inline(p,value,size=12,bold=False,color="000000"):
    pattern=r"(\*\*.*?\*\*|`[^`]+`|\[[^\]]+\]\(https?://[^)]+\))"
    for token in re.split(pattern,value):
        if not token:continue
        link=re.fullmatch(r"\[([^\]]+)\]\((https?://[^)]+)\)",token)
        if link:
            h=element("w:hyperlink",**{"r:id":p.part.relate_to(link[2],RT.HYPERLINK,is_external=True)})
            rr=element("w:r");pr=element("w:rPr");pr.append(element("w:color",**{"w:val":"176E65"}));pr.append(element("w:sz",**{"w:val":int(size*2)}));rr.append(pr);tt=element("w:t");tt.text=link[1];rr.append(tt);h.append(rr);p._p.append(h)
        else:
            strong=token.startswith("**") and token.endswith("**")
            mono=token.startswith("`") and token.endswith("`")
            content=token[2:-2] if strong else token[1:-1] if mono else token
            run=p.add_run(content);font(run,size if not mono else min(size,10.5),bold or strong,mono,color)


def bookmark(p,key,index):
    p._p.insert(0,element("w:bookmarkStart",**{"w:id":index,"w:name":"bm_"+key}))
    p._p.append(element("w:bookmarkEnd",**{"w:id":index}))


def field(p,code,fallback):
    run=p.add_run();font(run,10.5)
    run._r.append(element("w:fldChar",**{"w:fldCharType":"begin"}))
    instr=element("w:instrText",**{"xml:space":"preserve"});instr.text=" "+code+" ";run._r.append(instr)
    run._r.append(element("w:fldChar",**{"w:fldCharType":"separate"}))
    t=element("w:t");t.text=str(fallback);run._r.append(t)
    run._r.append(element("w:fldChar",**{"w:fldCharType":"end"}))


def math_run(value):
    r=element("m:r");r.append(element("w:rPr"));r[-1].append(element("w:rFonts",**{"w:ascii":"Cambria Math","w:hAnsi":"Cambria Math"}))
    t=element("m:t");t.text=value;r.append(t);return r


def frac(top,bottom):
    f=element("m:f");num=element("m:num");num.append(math_run(top));den=element("m:den");den.append(math_run(bottom));f.extend([num,den]);return f


def equation(doc,name):
    p=doc.add_paragraph();p.paragraph_format.space_after=Pt(7);p.paragraph_format.line_spacing=1.05
    op=element("m:oMathPara");math=element("m:oMath");op.append(math)
    if name=="sigmoid":
        math.append(math_run("p = "));f=frac("1","");den=f.find(qn("m:den"));den.clear();den.append(math_run("1 + "))
        sup=element("m:sSup");base=element("m:e");base.append(math_run("e"));power=element("m:sup");power.append(math_run("−(az+b)"));sup.extend([base,power]);den.append(sup);math.append(f)
    elif name=="lora":
        sub=element("m:sSub");base=element("m:e");base.append(math_run("W"));label=element("m:sub");label.append(math_run("new"));sub.extend([base,label]);math.append(sub)
        math.append(math_run(" = W + "));math.append(frac("α","r"));math.append(math_run(" BA"))
    else:
        values={"accuracy":("Accuracy","TP + TN","TP + TN + FP + FN"),"precision":("Precision","TP","TP + FP"),"recall":("Recall","TP","TP + FN"),"f1":("F1","2 × Precision × Recall","Precision + Recall")}
        label,a,b=values[name];math.append(math_run(label+" = "));math.append(frac(a,b))
    p._p.append(op)


def widths_for(rows):
    head=rows[0];n=len(head)
    if n==5:return [2.5,.75,1.18,1.18,1.19]
    if n==4:return [2.05,1.02,1.02,2.71] if "观察" in head else [2.9,1.3,1.3,1.3]
    if n==3:
        if head[0]=="接口":return [3.15,.6,3.05]
        if head[0]=="日期":return [.9,3.8,2.1]
        if head[0]=="时间":return [.95,2.7,3.15]
        if head[0] in {"分工方向","检查项"}:return [1.1,2.6,3.1]
        if head[0]=="入口":return [1.05,1.45,4.3]
        if "训练" in head[1] or "基线" in head[1] or "预测" in head[1]:return [3.1,1.85,1.85]
        return [1.65,2.15,3.0]
    if n==2:return [1.45,5.35] if head[0] in {"索引","目录或文件","现象","术语"} else [2.35,4.45]
    return [6.8/n]*n


def table(doc,rows):
    widths=widths_for(rows);t=doc.add_table(rows=0,cols=len(rows[0]));t.alignment=WD_TABLE_ALIGNMENT.CENTER;t.autofit=False
    for col,w in zip(t.columns,widths):col.width=Inches(w)
    props=t._tbl.tblPr;bd=element("w:tblBorders")
    for side in ("top","left","bottom","right","insideH","insideV"):bd.append(element("w:"+side,**{"w:val":"single","w:sz":"4","w:color":"D9D9D9"}))
    props.append(bd);props.append(element("w:tblLayout",**{"w:type":"fixed"}))
    for i,rowdata in enumerate(rows):
        row=t.add_row();rp=row._tr.get_or_add_trPr();rp.append(element("w:cantSplit"))
        if i==0:rp.append(element("w:tblHeader"))
        for j,value in enumerate(rowdata):
            cell=row.cells[j];cell.width=Inches(widths[j]);cell.vertical_alignment=WD_CELL_VERTICAL_ALIGNMENT.CENTER
            cp=cell._tc.get_or_add_tcPr();cp.append(element("w:shd",**{"w:fill":"193E50" if i==0 else "F3F6F7" if i%2==0 else "FFFFFF"}))
            mar=element("w:tcMar")
            for side,v in (("top",70),("bottom",70),("left",95),("right",95)):mar.append(element("w:"+side,**{"w:w":v,"w:type":"dxa"}))
            cp.append(mar);p=cell.paragraphs[0];p.paragraph_format.line_spacing=Pt(16);p.paragraph_format.space_after=Pt(0);p.paragraph_format.space_before=Pt(0)
            p.paragraph_format.widow_control=True
            numeric=j>0 and len(value)<20 and bool(re.match(r"^[0-9.≤≥×% /A-Za-z]+$",value))
            p.alignment=WD_ALIGN_PARAGRAPH.CENTER if numeric else WD_ALIGN_PARAGRAPH.LEFT
            inline(p,value,10.5,i==0,"FFFFFF" if i==0 else "000000")
    gap=doc.add_paragraph();gap.paragraph_format.space_after=Pt(2);gap.paragraph_format.line_spacing=Pt(2);font(gap.add_run(""),2)


def contents(doc,planned):
    items=[("项目全貌与技术基础","overview","ml"),("数据与预处理","data","preprocessing"),("课程基线与优化","tfidf","optimization"),("增强检测与评估","dear","results"),("文档与报告管理","documents","jobs"),("编辑 大模型与Agent","editing","agentflow"),("稳定性 接口与目录","reliability","code"),("启动 演示与项目讲稿","run","projectspeech"),("全项目常见追问","qa1","qa2"),("李阳个人贡献与讲稿","contribution","liyangqa"),("一周准备与快速复习","week","cheat"),("术语 官方资料与本地证据","glossary","evidence")]
    for label,start,end in items:
        p=doc.add_paragraph();p.paragraph_format.space_after=Pt(4);p.paragraph_format.line_spacing=Pt(19)
        inline(p,label+"  ",10.5)
        field(p,"PAGEREF bm_"+start+" \\h",planned[start]);inline(p," 至 ",10.5);field(p,"PAGEREF bm_end_"+end+" \\h",planned[end]);inline(p," 页",10.5)


def build():
    diagrams();doc=Document();sec=doc.sections[0]
    sec.page_width=Inches(8.5);sec.page_height=Inches(11)
    sec.top_margin=Inches(.75);sec.bottom_margin=Inches(.72);sec.left_margin=Inches(.85);sec.right_margin=Inches(.85)
    sec.footer_distance=Inches(.32)
    for name,size,bold in [("Normal",12,False),("Title",25,True),("Subtitle",14,False),("Heading 1",16,True),("Heading 2",12.5,True),("Caption",10,False)]:style_font(doc.styles[name],size,bold)
    normal=doc.styles["Normal"].paragraph_format;normal.line_spacing=Pt(22);normal.space_after=Pt(6);normal.widow_control=True
    for name,bef,aft in [("Heading 1",16,8),("Heading 2",10,5)]:
        pf=doc.styles[name].paragraph_format;pf.space_before=Pt(bef);pf.space_after=Pt(aft);pf.keep_with_next=True;pf.keep_together=True
    doc.styles["Title"].paragraph_format.space_after=Pt(14)
    doc.styles["Caption"].paragraph_format.space_after=Pt(9)
    doc.core_properties.title="内容可信工作台项目详解与答辩手册"
    doc.core_properties.subject="零基础学习与李阳个人贡献说明"
    doc.core_properties.author="项目学习资料";doc.core_properties.keywords="AIGC Python FastAPI 机器学习 答辩 李阳"
    raw="\n".join((WORK/f"guide_part{i}.md").read_text(encoding="utf-8") for i in (1,2,3))
    split=re.split(r"^:::page ([a-z0-9]+)\n",raw,flags=re.M)
    sections=[(split[i],split[i+1]) for i in range(1,len(split),2)]
    planned={key:i+1 for i,(key,_) in enumerate(sections)}
    for idx,(key,source) in enumerate(sections):
        lines=source.strip().splitlines();title=lines.pop(0)[2:]
        p=doc.add_paragraph(style="Title" if key=="cover" else "Heading 1")
        if key!="cover":p.paragraph_format.page_break_before=True
        inline(p,title,25 if key=="cover" else 16,True);bookmark(p,key,idx+1)
        if key=="cover":
            p.paragraph_format.space_before=Pt(30)
            sub=lines.pop(0);p=doc.add_paragraph(style="Subtitle");inline(p,sub,14)
        i=0
        while i<len(lines):
            value=lines[i].strip()
            if not value:i+=1;continue
            if value=="{{CONTENTS}}":contents(doc,planned);i+=1;continue
            if value.startswith("{{EQ:"):
                equation(doc,value[5:-2]);i+=1;continue
            image=re.fullmatch(r"!\[(.+)\]\(([^)]+)\)",value)
            if image:
                p=doc.add_paragraph();p.alignment=WD_ALIGN_PARAGRAPH.CENTER;p.paragraph_format.keep_with_next=True
                p.paragraph_format.line_spacing=1.0
                p.add_run().add_picture(str(FIG/f"{image[2]}.png"),width=Inches(6.65))
                pic=p._p.xpath('.//wp:docPr')
                if pic:pic[0].set("descr",image[1])
                cap=doc.add_paragraph(style="Caption");cap.alignment=WD_ALIGN_PARAGRAPH.CENTER;inline(cap,image[1],10);i+=1;continue
            if value.startswith("```"):
                i+=1
                while i<len(lines) and not lines[i].strip().startswith("```"):
                    p=doc.add_paragraph();p.paragraph_format.left_indent=Inches(.12);p.paragraph_format.line_spacing=Pt(16)
                    p.paragraph_format.space_after=Pt(0);p.paragraph_format.keep_with_next=i+1<len(lines) and not lines[i+1].strip().startswith("```")
                    r=p.add_run(lines[i]);font(r,10.5,mono=True);i+=1
                i+=1;p=doc.add_paragraph();p.paragraph_format.line_spacing=Pt(4);p.paragraph_format.space_after=Pt(3);continue
            if value.startswith("|"):
                rows=[]
                while i<len(lines) and lines[i].strip().startswith("|"):
                    row=[cell.strip() for cell in lines[i].strip().strip("|").split("|")]
                    if not all(re.fullmatch(r"[:\- ]+",cell) for cell in row):rows.append(row)
                    i+=1
                table(doc,rows);continue
            if value.startswith("## "):
                p=doc.add_paragraph(style="Heading 2");inline(p,value[3:],12.5,True)
            else:
                p=doc.add_paragraph();inline(p,value,10.5 if key=="references" else 12)
                if key=="references":
                    p.paragraph_format.line_spacing=Pt(17);p.paragraph_format.space_after=Pt(4)
            i+=1
        bookmark(doc.paragraphs[-1],"end_"+key,idx+1001)
    footer=sec.footer.paragraphs[0];footer.alignment=WD_ALIGN_PARAGRAPH.CENTER
    inline(footer,"第 ",9.5);field(footer,"PAGE",1);inline(footer," 页 共 ",9.5);field(footer,"NUMPAGES",1);inline(footer," 页",9.5)
    doc.settings.element.append(element("w:updateFields",**{"w:val":"true"}))
    doc.save(OUT)
    info={"file":str(OUT),"topics":len(sections),"tables":len(doc.tables),"figures":len(doc.inline_shapes),"editable_equations":len(doc.element.xpath('.//m:oMath')),
          "content_characters":sum(len(p.text) for p in doc.paragraphs)+sum(len(c.text) for t in doc.tables for row in t.rows for c in row.cells),"code_reference":"57d2f8f","member":"李阳"}
    (WORK/"build-info.json").write_text(json.dumps(info,ensure_ascii=False,indent=2),encoding="utf-8")
    print(json.dumps(info,ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");build()
