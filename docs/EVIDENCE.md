# Evidence and limits

## Full H3 videos

`results/h3-runs.json` is a sanitized extraction of four successful local ComfyUI histories and available media inspections. It retains prompt IDs, source-history hashes, execution durations, checkpoint fingerprints, cached node IDs, output hashes and sampled resource minima. Paths, LAN addresses, account names and supervisor details were removed.

The execution durations recompute to 187.570 / 233.113 / 243.159 / 261.838 seconds for the 5090 / 5080 / 4080 / 3080 Ti Laptop workers. Driver polling durations are separately recorded where available. Each is one complete 124-frame sample. The 3080 Ti and 4080 full runs followed smaller 39-frame runs and reused cached loader nodes. The 5080 run began with its remote H3 TE unloaded; its 49.685-second encoding receipt includes approximately 47.362 seconds of lazy loading. The 5090/4080/3080 Ti runs used a different TE host. Do not average these into a fleet performance score.

The montage was extracted again directly from all four source MP4 files during packaging. All decoded to 124 video frames. Detailed local inspections found H.264 video and 32 kHz stereo AAC with nonzero audio. Still-image montages do not prove temporal smoothness or cross-machine perceptual equivalence. Raw videos and the original reference image are not bundled.

## Image transport parity and cross-GPU differences

`results/image-http-parity.json` records seven Image cases where HTTP results matched native encoding on the same 5090 bit-for-bit, including extras; a wrong fingerprint returned 409. This is a tested lossless-transport result, not universal model-quality equivalence.

For one historical Image 769-token case, 5090 versus 3080 Ti native encoding produced a maximum absolute difference of 0.0901489 across 3,149,824 float32 values. Mean absolute difference was 0.00018638 and relative RMS difference was approximately 0.00425%; 38 values met or exceeded 0.02. The number 0.09015 is not a percentage and does not mean an image is 9% wrong. The original maximum-difference target of 0.02 failed and is not retroactively marked as passed. Application acceptance later relied on same-host transport parity, explicit version checks and observed successful workflows. Long-prompt perceptual equivalence remains unproven.

H3 did not receive the same native-versus-HTTP numerical parity study in this release. Its completed videos establish functional operation for the tested first-frame workflows, not all conditioning modes or bitwise cross-GPU equivalence.

## On-demand and idle lifecycle

A live 5090 service was observed loading Image, then switching to H3 while Image became unloaded. A subsequent 5080 video request started with H3 unloaded and caused automatic loading. The last encoding completed at 22:17:38 local time; at 22:28:02 the service remained running with both profiles unloaded and queue length zero. Observed GPU usage fell from approximately 16,966 to 372 MiB. `results/idle-unload-observation.json` archives this receipt summary, not a continuous timing trace. CPU/GPU process overhead need not fall to zero when model weights are unloaded.

## Historical targets and newer research

The original Image service target of every long encode completing within 1.2 seconds also failed in some samples. A later local operating check used a three-second sampled long-request bound; neither target is a latency SLA for this release. The historical approximately 70-second H3 encode is not a universal GPU baseline. The separate CPU TE study found large improvements after GPU memory-headroom tuning: [cpu-te-bench](https://github.com/JingWang-Star996/cpu-te-bench). Those are TE-only experiments on a different loading path and must not be substituted into these archived full-video results.

## Reproducibility scope

Code, workflow settings, runtime hashes and numerical summaries are public. Historical reference images and output tensors are absent, so exact embeddings and media cannot be independently regenerated from this package alone. You can rerun the method with your own licensed files and compare your own native/HTTP results. Single-sample successes do not establish concurrency, long-running stability, long-video production, portable installation on other operating systems or automatic dispatch.
