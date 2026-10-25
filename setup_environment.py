"""Create an isolated local environment; never change global Python."""
from pathlib import Path
import argparse, subprocess, sys, venv
root=Path(__file__).resolve().parent
p=argparse.ArgumentParser();p.add_argument('--torch-index',default='https://download.pytorch.org/whl/cpu');p.add_argument('--with-llm',action='store_true');p.add_argument('--research',action='store_true');a=p.parse_args()
if sys.version_info[:2]!=(3,12):raise SystemExit('Use Python 3.12: py -3.12 setup_environment.py (Windows)')
target=root/'.venv'
venv.EnvBuilder(with_pip=True).create(target)
python=target/('Scripts/python.exe' if sys.platform=='win32' else 'bin/python')
def install(*args):subprocess.run([str(python),'-m','pip','install',*args],check=True,cwd=root)
install('--upgrade','pip')
install('torch==2.14.0','torchvision==0.29.0','--index-url',a.torch_index)
runtime_requirements='requirements-runtime-tested.txt' if (root/'requirements-runtime-tested.txt').exists() else 'code/server/requirements.txt'
install('-r',runtime_requirements)
if a.with_llm:
    llm_requirements='requirements-llm-tested.txt' if (root/'requirements-llm-tested.txt').exists() else 'code/llm/requirements.txt'
    install('-r',llm_requirements)
if a.research:install('-r','requirements-research.txt')
subprocess.run([str(python),'-m','pip','check'],check=True)
subprocess.run([str(python),'-c','import torch,torchvision; print(torch.__version__,torchvision.__version__)'],check=True)
print('Environment ready. Run .venv/Scripts/python.exe run.py on Windows, .venv/bin/python run.py on macOS/Linux.')
