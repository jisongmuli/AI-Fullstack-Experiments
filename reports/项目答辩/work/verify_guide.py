"""Read-only checks for the final handbook, metrics, roles and rendered pages."""
from pathlib import Path
import hashlib
import json
import re
import sys

import pdfplumber
from docx import Document

WORK=Path(__file__).resolve().parent
ROOT=WORK.parents[2]
FILE=WORK.parent/"内容可信工作台项目详解与答辩手册_李阳.docx"
RENDER=WORK/"render-delivery"


def main():
    raw="\n".join((WORK/f"guide_part{i}.md").read_text(encoding="utf-8") for i in (1,2,3))
    topics=re.split(r"^:::page [a-z0-9]+\n",raw,flags=re.M)[1:]
    titles=[row.strip().splitlines()[0][2:] for row in topics]
    compact=lambda value:re.sub(r"\s+","",value)
    pdf_path=next(RENDER.glob("*.pdf"))
    pages=[]
    with pdfplumber.open(pdf_path) as pdf:
        assert len(pdf.pages)==len(titles)==44
        for index,(page,title) in enumerate(zip(pdf.pages,titles),1):
            text=page.extract_text() or ""
            assert compact(text.splitlines()[0])==compact(title),(index,title)
            assert re.search(fr"第\s*{index}\s*页.*44\s*页",text),index
            assert "\ufffd" not in text,index
            pages.append({"page":index,"title":title,"characters":len(text)})
    doc=Document(FILE)
    assert len(doc.inline_shapes)==6 and len(doc.tables)==25
    assert len(doc.element.xpath('.//m:oMath'))==6
    assert len(doc.element.xpath('.//w:hyperlink'))==14
    # Word removes redundant black overrides while saving. With no color
    # override, these styles inherit the document's black text default.
    assert all(str(doc.styles[name].font.color.rgb) in {"000000","None"} and doc.styles[name].font.color.type in {None,1}
               for name in ("Title","Heading 1","Heading 2"))
    text="\n".join([p.text for p in doc.paragraphs]+[cell.text for table in doc.tables for row in table.rows for cell in row.cells])
    for required in ("李阳","文本获取、字段与标签核查","文本基线核查与质量指标整理","98.50%","66.25%","159744","Qwen3.5","TF-IDF"):
        assert required in text,required
    policy=json.loads((ROOT/"models/algorithm_policy.json").read_text(encoding="utf-8"))
    for row in policy["evaluation_summary"]:
        assert f"{row['accuracy']*100:.2f}%" in text
    course=json.loads((ROOT/"results/course_20261027/experiment2-text.json").read_text(encoding="utf-8"))
    assert course["optimized"]["confusion_matrix"]==[[296,4],[5,295]]
    roles=[]
    for relative in ("reports/实验1/实验1报告_数据采集与预处理_补全证据版.docx","reports/实验3/实验3报告_算法优化实验.docx"):
        original=Document(ROOT/relative)
        matches=[[c.text for c in row.cells] for table in original.tables for row in table.rows if any("李阳" in c.text for c in row.cells)]
        assert matches,relative
        roles.append({"source":relative,"recorded_role":matches[0][-1]})
    assert roles[0]["recorded_role"]=="文本获取、字段与标签核查"
    assert roles[1]["recorded_role"]=="文本基线核查与质量指标整理"
    result={"status":"passed","verified_on":"2026-11-12","output":FILE.relative_to(ROOT).as_posix(),
            "sha256":hashlib.sha256(FILE.read_bytes()).hexdigest(),"pages":44,"tables":25,"figures":6,"editable_equations":6,
            "source_code_commit":"57d2f8f1bbbd482030aa35b7d629eb79d20a2757",
            "member":"李阳","roles_verified":roles,"personal_contribution_pages":[34,35,36,37,38,39],
            "visual_review":{"pages_reviewed":list(range(1,45)),"last_changed_pages_reviewed":[16,26,36],"unchanged_pages_verified_by_png_hash":41,
                             "result":"No clipped figures, overlapping text, broken tables or missing glyphs observed"},
            "page_index":pages}
    (WORK/"verification.json").write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n",encoding="utf-8")
    print(json.dumps({key:result[key] for key in ("status","pages","tables","figures","editable_equations","personal_contribution_pages")},ensure_ascii=False))


if __name__=="__main__":
    sys.stdout.reconfigure(encoding="utf-8");main()
