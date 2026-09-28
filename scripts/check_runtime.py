"""Read-only source-presence probe; does not import ComfyUI or load CUDA."""
import argparse,json,sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--comfy-root',type=Path,required=True)
a=p.parse_args();root=a.comfy_root.resolve()
required=['main.py','comfy/sd.py','comfy/model_management.py','comfy/text_encoders/qwen_image21.py','comfy_extras/nodes_minimax_h3.py']
checks={name:(root/name).is_file() for name in required}
print(json.dumps({'source_presence':checks,'all_present':all(checks.values()),'note':'Presence only; quantized kernels, models, CUDA execution and compatibility still require validation.'},indent=2))
sys.exit(0 if all(checks.values()) else 1)
