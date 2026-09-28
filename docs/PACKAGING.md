# Packaging and provenance

This is a clean export from the tested local `te-split` working tree, not a publication of that repository's private Git history. `provenance/code-manifest.json` records source and release hashes for every copied Python/JavaScript file. Encoding, transport, queue and lifecycle implementation files were copied unchanged. Only the two client endpoint defaults were generalized: environment variables with a loopback fallback replace private LAN addresses.

Private deployment/supervisor scripts, SSH aliases, firewall changes, machine configs, logs, raw media, reference images, model weights and legacy gateway code are excluded. The public test subset excludes the retired gateway tests. New install/check/result scripts operate on user-supplied paths and recorded data; they do not install models, stop services or dispatch production jobs.

API workflow copies replace service URLs with loopback, input filenames with `example.png`, disable Image fallback and allow a cold-load timeout. Checkpoint pins and generation settings remain explicit. The public workflow file hash therefore differs from each private historical prompt graph.

Build source and node ZIPs with `python scripts/build_release.py`. The builder only includes known release directories, rejects symlinks and excludes caches, local configs and weights. `dist/SHA256SUMS` hashes both ZIPs. `MANIFEST.sha256` inside the source ZIP lists the published source files.
