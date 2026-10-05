# Data sources

Retrieval date: 2026-10-04. Raw files are in `data/raw/` (gitignored; re-create with `analysis/fetch_data.py`). Per-file SHA-256 hashes are in the manifests `data/tb2_manifest.csv` and `data/tau2_manifest.csv` (columns: path, bytes, sha256).

## Terminal-Bench 2.0 leaderboard submissions

- URL: https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard
- Revision (commit): `572b2614be2c0cb2527e14f5b1e4026f1072e6c1` (last modified 2026-05-15)
- Licence: Apache-2.0 (dataset card)
- Subset downloaded: 33,367 files, 157.7 MB, from `submissions/terminal-bench/2.0/`: per-trial `result.json` (32,803), per-run `config.json` and `result.json` (245 run directories), per-submission `metadata.yaml`/`metadata.yml` (75 submissions). The full dataset is 121.6 GB (Hugging Face `usedStorage`); agent artifacts (logs, verifier output, trajectories) were not downloaded.
- Method: blob-less git fetch of the pinned revision, then fetch of the listed blobs only (`analysis/fetch_data.py tb2`).
- Manifest: `data/tb2_manifest.csv`, SHA-256 of the manifest file `e71ba7aad59799f20014d6e7c480bbba41a19a238b7ecec8598b27aec11825a1`.

## tau2-bench leaderboard submissions

- URL: https://github.com/sierra-research/tau2-bench, path `web/leaderboard/public/submissions/`
- Revision (commit): `5bfa7e37b36656b37dc6d022156be6563c1007f3` (2026-09-28)
- Licence: MIT (repository licence)
- Subset downloaded: 75 files, 0.2 MB (`submission.json` per submission, `manifest.json`, `schema.json`, `README.md`). Trajectory files are hosted outside the repository and were not downloaded.
- Method: GitHub raw files at the pinned commit (`analysis/fetch_data.py tau2`).
- Manifest: `data/tau2_manifest.csv`, SHA-256 of the manifest file `20606006fc1b7a41aa98f06c8c2ab2f4474a3271ee7046e02193c48ec062549f`.
