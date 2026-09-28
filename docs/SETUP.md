# Setup and compatibility

## What to prepare

The validated deployment used Windows 11, Python 3.13.14, a ComfyUI build reporting 0.37.4, Torch 2.13.0+cu130, comfy-kitchen 0.2.35 and comfy-aimdo 0.5.5. These are recorded observations, not a claim that an arbitrary upstream checkout supports every tested quantized file. Package versions and key source hashes are archived in `provenance/validated-runtime.json`.

The host must provide `comfy.text_encoders.qwen_image21`, `comfy_extras.nodes_minimax_h3`, `CLIPType.QWEN_IMAGE`, `CLIPType.MINIMAX`, the required quantized loaders and CUDA kernels. A source-presence check is provided; it is not a model execution test. Obtain compatible model files separately under their applicable terms. We do not provide unofficial weight-download links or redistribute the runtime.

Tested files:

| Role | File | SHA-256 |
|---|---|---|
| Image TE | qwen3vl_8b_int8_convrot.safetensors | 8bfd0f6e12abf2d2d697ecc888e5e90b0d6741d6708f05799f53afa560452e8f |
| H3 TE | qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors | 35a88d51044231fe332301d7a62aa81e3f2cba62febeb446e2c1e3e0ef76f2c6 |
| H3 DiT | minimax_h3_fl2va_pruned_int8_convrot.safetensors | e889202c41dafb67b10d67b97f0d8541508036a6090af23425a5c2615d03c47a |
| H3 video VAE | minimax_h3_video_vae_fp16.safetensors | 7c1f131492e7eddacaac9069a61b81bdd39de5cc96561e677c5eab1cdce5e522 |
| H3 audio VAE | minimax_h3_audio_vae_fp32.safetensors | 8e505d95dd1561d47abd43d4238fd40d9bb1ae9e147ed0a4cba778d76ae4db48 |

TE files belong on the encoding host. DiT and VAE files belong on each generation worker. Worker system RAM also matters: unloading/offloading cannot remove the need to hold or move weights.

## Start the service

1. Use the Python executable from your compatible ComfyUI environment. In the Windows portable layout this is usually `python_embeded/python.exe`; replace `python` below with its full path when needed.
2. Install `requirements-server.txt`. Do not blindly replace the host's Torch, kernels or ComfyUI dependencies. `requirements-server.lock.txt` records the service-layer versions from the tested deployment; it is not a complete environment lock.
3. Copy `config.example.yaml` to `config.local.yaml`. Set absolute paths. Remove profiles whose files are unavailable. Every configured checkpoint is fingerprinted at startup, even though GPU loading is lazy.
4. For LAN use, change `host` to `0.0.0.0` and configure the selected port in your own firewall. The examples do not modify firewall rules, SSH, scheduled tasks or other services.
5. Set an optional `TE_AUTH_TOKEN` in the service environment, and run `python run_server.py --config config.local.yaml`.

Use a trusted LAN or a trusted TLS termination layer. The built-in server has no TLS, user accounts or internet-facing abuse controls. Bearer tokens over plain HTTP are not encrypted.

`GET /health` returns profile fingerprints, loaded/loading flags, queue lengths and activity. `POST /load` and `/unload` accept `{"model":"minimax_h3"}` or `{"model":"qwen3vl_8b"}`. `POST /encode` loads the selected profile automatically. Explicit preloading is optional. Image loading requires at least 10 GiB free GPU memory and H3 at least 14 GiB at the current guard; these are implementation checks, not guaranteed total-memory budgets.

Keep `exclusive_profiles: true` for a shared single GPU. Requests, maintenance, idle unloading and shutdown use the same serial lock. Switching profiles releases the previous model before loading the next. `idle_unload_min: 10` unloads after ten idle minutes; the reaper checks about every five seconds. A running or queued request prevents its profile from being reaped. `maintenance_interval_s: 0` disables keepalive computation.

The server's per-profile `vram_mb` and `cuda_reserved_mb` values currently describe process-wide CUDA allocator totals, not an isolated allocation count for each model. Use `loaded` and activity flags when interpreting them.

## Install the worker nodes

Run `python scripts/install_nodes.py --comfy-root <ComfyUI-directory>`, or extract the node ZIP into `custom_nodes`. Installation refuses to overwrite an existing `ComfyUI-StageBridge` directory. Back up and remove older `ComfyUI-RemoteTE` copies through your normal maintenance procedure first; do not load both packages.

The nodes are `QwenImage21Remote`, `TextEncodeQwenImage21Remote`, `TextEncodeQwenImageEditRemote` and `MiniMaxH3ImageToVideoRemote`. The default endpoint is loopback. `REMOTE_TE_SERVER_URL` sets the default, with `REMOTE_TE_H3_SERVER_URL` as an H3-specific override. Explicit workflow `server_url` values take precedence. `REMOTE_TE_API_TOKEN` carries the bearer token, never workflow JSON.

H3 examples use DynamicVRAM in the existing ComfyUI runtime. Do not infer that a high-VRAM flag used for a separate Image experiment is appropriate for H3. Check current GPU owners and ComfyUI queues before starting a production run.

## Submit a workflow

The included JSON files are ComfyUI **API prompt graphs**. They are not UI workflow exports. Upload your own input image to the worker's input directory through ComfyUI, edit `LoadImage.image`, choose the locally installed models and set the service's LAN URL. The supplied fingerprints identify the listed tested checkpoints; if using different weights, verify them and explicitly set their fingerprint instead of disabling mismatch checks.

An example PowerShell submission to an already running local ComfyUI worker:

```powershell
$graph = Get-Content workflows/h3-fl2va-39-api.json -Raw | ConvertFrom-Json
# Edit the graph or its file before submission.
$body = @{prompt=$graph; client_id='stagebridge-manual'} | ConvertTo-Json -Depth 100
$reply = Invoke-RestMethod -Method Post http://127.0.0.1:8188/prompt -ContentType application/json -Body $body
$reply.prompt_id
```

Save the returned prompt ID and query `/history/<prompt_id>`. Do not blindly repeat a timed-out POST; it may have been accepted. Verify successful history, the `REMOTE H3` or `REMOTE` receipt, the saved output and media decoding. Image examples disable local fallback. If you enable it, budget local TE memory and check whether the node reports `FALLBACK`.

Cold loads, request queueing and model switching can add latency. Use an appropriate node timeout; the H3 examples use 900 seconds. Changing timeout does not accelerate inference.

## Stop and roll back

Stop new submissions, wait for the owned queue/active request to finish, optionally unload profiles, then terminate the service using its normal supervisor. Restart only that worker when replacing node files. Keep the previous node directory and local config for rollback. No automatic worker startup, shutdown or fleet failover is supplied.
