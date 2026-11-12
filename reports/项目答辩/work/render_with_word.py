"""Use the packaged rasterizer with native Word export on Windows."""
from pathlib import Path
import importlib.util
import argparse
import os
import subprocess
import sys

parser=argparse.ArgumentParser(add_help=False)
parser.add_argument("--skill-directory",required=True)
options,remaining=parser.parse_known_args()
SKILL=Path(options.skill_directory)
sys.argv=[sys.argv[0],*remaining]
DEPENDENCIES=Path(sys.executable).parent.parent
os.environ["PATH"]=str(DEPENDENCIES/"native/poppler/Library/bin")+os.pathsep+os.environ["PATH"]
spec=importlib.util.spec_from_file_location("packaged_render_docx",SKILL/"render_docx.py")
renderer=importlib.util.module_from_spec(spec);spec.loader.exec_module(renderer)


def word_pdf(doc_path,user_profile,convert_tmp_dir,stem,verbose=False):
    out=Path(convert_tmp_dir)/(stem+".pdf")
    script=Path(user_profile)/"word_export.ps1"
    script.write_text('''param([string]$InputFile,[string]$OutputFile)
$ErrorActionPreference='Stop'
$app=New-Object -ComObject Word.Application
$app.Visible=$false
$app.DisplayAlerts=0
$app.AutomationSecurity=3
$doc=$null
try {
 $doc=$app.Documents.Open($InputFile,$false,$false)
 $doc.Repaginate()
 $doc.Fields.Update() | Out-Null
 $doc.Repaginate()
 $doc.Fields.Update() | Out-Null
 $doc.Save()
 $doc.ExportAsFixedFormat($OutputFile,17)
} finally {
 if ($null -ne $doc) { $doc.Close(0) }
 $app.Quit()
}
''',encoding="utf-8-sig")
    result=subprocess.run(["powershell.exe","-NoProfile","-File",str(script),str(Path(doc_path).resolve()),str(out)],capture_output=True,text=True,timeout=240)
    if result.returncode or not out.exists():return "",result.stdout+result.stderr
    return str(out),"Exported through native Microsoft Word; packaged rasterizer generates page PNGs."


renderer.convert_to_pdf=word_pdf
if __name__=="__main__":renderer.main()
