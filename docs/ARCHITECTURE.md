# Architecture

StageBridge divides a generation pipeline by responsibility. It does not shard one DiT forward pass across cards and does not combine separate GPU memories into a single address space.

1. A ComfyUI remote node validates text and reference images.
2. The TE host validates the model fingerprint, selects the Image or H3 encoder and runs the native encoding implementation.
3. Float32 conditioning and supported metadata are packed as safetensors and transferred over HTTP.
4. The worker combines conditioning with local VAE keyframe/reference latents and performs sampling, decoding and saving.

Image hidden states have width 4096; H3 hidden states have width 5120 and retain `minimax_token_tags`. Image slots and supported extras preserve their native types. Strict protocol validation rejects unsupported fields rather than silently dropping semantic information. Input images are transported as float32 tensors and bounded by request/resize budgets.

Exclusive mode uses one shared serial lock across profiles. Queue reservations remain bounded per profile. A disconnected queued request releases its slot; an already running GPU operation completes safely. An idle reaper does not update business last-use time. There is no multi-GPU service pool, job placement algorithm, durable job queue or distributed sampler in this release.

Splitting stages can make deployment more flexible, but a worker's DiT/VAE and the encoder host both retain their own memory constraints. Network cost, model switching, CPU offload, disk loading and queueing can outweigh any benefit for a particular job. Throughput improvement from overlapping jobs is a future experiment, not a result established by the four sequential video runs.
