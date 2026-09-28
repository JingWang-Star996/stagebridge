"""Build deterministic source/node archives from the public allowlist."""
import hashlib,zipfile
from pathlib import Path

ROOT=Path(__file__).resolve().parents[1]
VERSION='0.1.0'
ROOT_FILES={'.gitignore','README.md','README.zh-CN.md','LICENSE','CHANGELOG.md','THIRD_PARTY_NOTICES.md','config.example.yaml','requirements-server.txt','requirements-server.lock.txt','requirements-test.txt','run_server.py'}
DIRS={'te_server','comfyui_nodes','tests','scripts','docs','results','provenance','workflows','assets'}
files=[]
for path in ROOT.rglob('*'):
    relative=path.relative_to(ROOT)
    if relative.parts[0] not in DIRS and str(relative) not in ROOT_FILES:continue
    if path.is_symlink():raise RuntimeError('Symlink in release: '+str(relative))
    if not path.is_file() or '__pycache__' in relative.parts:continue
    if path.suffix.lower() in {'.pyc','.pyo'}:continue
    if path.suffix.lower() in {'.safetensors','.pt','.pth','.mp4','.key','.pem'}:raise RuntimeError('Excluded asset type: '+str(relative))
    files.append(path)
files.sort(key=lambda p:p.relative_to(ROOT).as_posix())
manifest=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.relative_to(ROOT).as_posix()+'\n' for p in files)
(ROOT/'MANIFEST.sha256').write_text(manifest,encoding='utf-8')
files.append(ROOT/'MANIFEST.sha256')
dist=ROOT/'dist';dist.mkdir(exist_ok=True)
def archive(target,rows):
    with zipfile.ZipFile(target,'w',compression=zipfile.ZIP_DEFLATED,compresslevel=9) as z:
        for source,name in rows:
            info=zipfile.ZipInfo(name,(2026,9,28,0,0,0));info.compress_type=zipfile.ZIP_DEFLATED;info.external_attr=0o644<<16
            z.writestr(info,source.read_bytes())
source_zip=dist/f'stagebridge-v{VERSION}.zip'
node_zip=dist/f'stagebridge-comfyui-nodes-v{VERSION}.zip'
archive(source_zip,[(p,f'stagebridge-v{VERSION}/'+p.relative_to(ROOT).as_posix()) for p in files])
archive(node_zip,[(p,'ComfyUI-StageBridge/'+p.relative_to(ROOT/'comfyui_nodes').as_posix()) for p in files if p.is_relative_to(ROOT/'comfyui_nodes')]+[(ROOT/'LICENSE','ComfyUI-StageBridge/LICENSE')])
checks=''.join(hashlib.sha256(p.read_bytes()).hexdigest()+'  '+p.name+'\n' for p in [source_zip,node_zip])
(dist/'SHA256SUMS').write_text(checks,encoding='utf-8')
print(checks,end='')
