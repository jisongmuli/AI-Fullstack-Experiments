"""Use the packaged DOCX rasterizer with native Word PDF conversion on Windows."""
from pathlib import Path
import importlib.util, os, shutil, subprocess, sys
SKILL=Path(r'C:/Users/86158/.codex/plugins/cache/openai-primary-runtime/documents/26.1024.11018/skills/documents')
POPPLER=Path(r'C:/Users/86158/.cache/codex-runtimes/codex-primary-runtime/dependencies/native/poppler/Library/bin')
os.environ['PATH']=str(POPPLER)+os.pathsep+os.environ['PATH']
spec=importlib.util.spec_from_file_location('packaged_render_docx',SKILL/'render_docx.py')
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)
def word_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose=False):
    out=Path(convert_tmp_dir)/(stem+'.pdf')
    ps=Path(user_profile)/'word_export.ps1'
    ps.write_text('''param([string]$InputFile,[string]$OutputFile)
$ErrorActionPreference='Stop'
$app=New-Object -ComObject Word.Application
$app.Visible=$false
$app.DisplayAlerts=0
$app.AutomationSecurity=3
$doc=$null
try {
 $doc=$app.Documents.Open($InputFile,$false,$true)
 $doc.Repaginate()
 $doc.ExportAsFixedFormat($OutputFile,17)
} finally {
 if ($null -ne $doc) { $doc.Close(0) }
 $app.Quit()
}
''',encoding='utf-8-sig')
    completed=subprocess.run(['powershell.exe','-NoProfile','-ExecutionPolicy','Bypass','-File',str(ps),str(Path(doc_path).resolve()),str(out)],capture_output=True,text=True)
    if completed.returncode or not out.exists():return '',completed.stdout+completed.stderr
    return str(out),'Rendered through native Microsoft Word PDF export.'
renderer.convert_to_pdf=word_pdf
if __name__=='__main__':renderer.main()
