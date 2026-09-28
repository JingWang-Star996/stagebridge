"""Install a fresh node package. Never overwrite an existing installation."""
import argparse, json, shutil
from pathlib import Path

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--comfy-root',type=Path,required=True)
    p.add_argument('--dry-run',action='store_true')
    a=p.parse_args();root=a.comfy_root.resolve()
    if not (root/'main.py').is_file() or not (root/'custom_nodes').is_dir():
        p.error('--comfy-root must be a ComfyUI directory with main.py and custom_nodes')
    target=root/'custom_nodes'/'ComfyUI-StageBridge'
    if target.exists():p.error('Target already exists; back up and maintain it explicitly before installing')
    if (root/'custom_nodes'/'ComfyUI-RemoteTE').exists():p.error('Older ComfyUI-RemoteTE exists; avoid duplicate class registrations')
    source=Path(__file__).resolve().parents[1]/'comfyui_nodes'
    if any(f.is_symlink() for f in source.rglob('*')):p.error('Source contains a symlink')
    if not a.dry_run:shutil.copytree(source,target,ignore=shutil.ignore_patterns('__pycache__','*.pyc'))
    print(json.dumps({'status':'planned' if a.dry_run else 'installed','target':str(target),'restart_comfyui':True}))
if __name__=='__main__':main()
