"""Portable foreground launcher, local only; Ctrl+C stops its own process."""
import argparse, socket, sys
from pathlib import Path
root=Path(__file__).resolve().parent
if sys.version_info[:2]!=(3,12):raise SystemExit('Python 3.12 is required; use the project .venv interpreter.')
p=argparse.ArgumentParser();p.add_argument('--port',type=int,default=8002);a=p.parse_args()
if not 1<=a.port<=65535:raise SystemExit('Port must be 1..65535')
with socket.socket() as s:
    if s.connect_ex(('127.0.0.1',a.port))==0:raise SystemExit(f'Port {a.port} is occupied. Use --port 8003; no existing service was stopped.')
sys.path[:0]=[str(root/'code/server'),str(root/'code/models')]
import uvicorn
print(f'Open http://127.0.0.1:{a.port}/?v=handoff#document')
uvicorn.run('app:app',host='127.0.0.1',port=a.port)
