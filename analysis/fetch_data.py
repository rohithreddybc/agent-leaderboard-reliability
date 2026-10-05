"""Selective download of public data used by the RQ3 re-analysis.

Terminal-Bench 2.0 leaderboard (Hugging Face dataset, Apache-2.0) is 121.6 GB in
full. Only the small outcome files are fetched (about 158 MB): per-trial
``result.json``, per-run ``config.json`` / ``result.json`` and per-submission
``metadata.yaml``, via a blob-less git fetch of the pinned revision.

tau2-bench (GitHub, MIT): only ``web/leaderboard/public/submissions/**``.

Everything is pinned to a commit hash. A manifest with SHA-256 per file is
written next to the data. Network access is needed; nothing here is imported
by the analysis modules.

Usage:
    python fetch_data.py tb2   [--out ../data/raw/tb2]
    python fetch_data.py tau2  [--out ../data/raw/tau2]
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import os
import re
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

TB2_REPO = "harborframework/terminal-bench-2-leaderboard"
TB2_REV = "572b2614be2c0cb2527e14f5b1e4026f1072e6c1"
TB2_ROOT = "submissions/terminal-bench/2.0"
TAU2_REPO = "sierra-research/tau2-bench"
TAU2_REV = "5bfa7e37b36656b37dc6d022156be6563c1007f3"
TAU2_ROOT = "web/leaderboard/public/submissions"

_session = requests.Session()


def _get(url: str, **kw):
    err = None
    for _ in range(5):
        try:
            r = _session.get(url, timeout=120, **kw)
            r.raise_for_status()
            return r
        except requests.RequestException as e:  # retry transient failures
            err = e
    raise err


TB2_FILE = re.compile(
    r"^submissions/terminal-bench/2\.0/[^/]+/(metadata\.ya?ml|[^/]+/(result|config)\.json|[^/]+/[^/]+/result\.json)$")
BATCH = 16  # the Hugging Face git server rejects larger multi-object fetches


def _git(repo: Path, *args: str, check: bool = True, input: bytes | None = None) -> bytes:
    env = {**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}
    p = subprocess.run(["git", "-C", str(repo), *args], capture_output=True, env=env, input=input)
    if check and p.returncode:
        raise RuntimeError(f"git {' '.join(args)}: {p.stderr.decode(errors='replace')[:300]}")
    return p.stdout


def fetch_tb2(out: Path, workdir: Path) -> list[tuple[str, int, str]]:
    """Blob-less git fetch of the pinned revision, then fetch only the wanted
    blobs (the HF resolve endpoint rate-limits at ~1,000 requests per 5 min,
    far below the 33,367 files needed)."""
    workdir.mkdir(parents=True, exist_ok=True)
    _git(workdir, "init", "-q")
    _git(workdir, "remote", "add", "origin", f"https://huggingface.co/datasets/{TB2_REPO}", check=False)
    _git(workdir, "fetch", "-q", "--depth", "1", "--filter=blob:none", "origin", TB2_REV)
    listing = _git(workdir, "ls-tree", "-r", TB2_REV).decode().splitlines()
    wanted = {}
    for line in listing:
        meta, path = line.split("	", 1)
        if TB2_FILE.match(path):
            wanted[path] = meta.split()[2]
    oids = sorted(set(wanted.values()))
    print(f"{len(wanted)} files, {len(oids)} distinct blobs")
    batches = [oids[i:i + BATCH] for i in range(0, len(oids), BATCH)]

    def get(batch):
        for _ in range(3):
            if subprocess.run(["git", "-C", str(workdir), "fetch", "-q", "--filter=blob:none", "origin", *batch],
                              capture_output=True, env={**os.environ, "GIT_LFS_SKIP_SMUDGE": "1"}).returncode == 0:
                return
        for o in batch:  # fall back to single-object fetches
            _git(workdir, "fetch", "-q", "--filter=blob:none", "origin", o)

    with ThreadPoolExecutor(4) as ex:
        list(ex.map(get, batches))
    rows = []
    items = sorted(wanted.items())
    proc = subprocess.Popen(["git", "-C", str(workdir), "cat-file", "--batch"], stdin=subprocess.PIPE,
                            stdout=subprocess.PIPE, env={**os.environ, "GIT_NO_LAZY_FETCH": "1"})
    for path, oid in items:
        proc.stdin.write(oid.encode() + b"\n")
        proc.stdin.flush()
        header = proc.stdout.readline().split()
        if len(header) != 3:
            raise RuntimeError(f"blob {oid} for {path} not available locally: {header}")
        data = proc.stdout.read(int(header[2]))
        proc.stdout.read(1)  # trailing newline
        dest = out / path[len(TB2_ROOT):].lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(data)
        rows.append((path, len(data), hashlib.sha256(data).hexdigest()))
    proc.stdin.close()
    proc.wait()
    return rows


def tau2_paths() -> list[str]:
    r = _get(f"https://api.github.com/repos/{TAU2_REPO}/git/trees/{TAU2_REV}?recursive=1")
    tree = r.json()
    if tree.get("truncated"):
        sys.exit("GitHub tree listing truncated; cannot enumerate safely")
    return sorted(x["path"] for x in tree["tree"]
                  if x["type"] == "blob" and x["path"].startswith(TAU2_ROOT + "/"))


def download(paths: list[str], url_for, out: Path, strip: str) -> list[tuple[str, int, str]]:
    def one(p: str):
        dest = out / p[len(strip):].lstrip("/")
        dest.parent.mkdir(parents=True, exist_ok=True)
        data = _get(url_for(p)).content
        dest.write_bytes(data)
        return (p, len(data), hashlib.sha256(data).hexdigest())

    with ThreadPoolExecutor(16) as ex:
        return list(ex.map(one, paths))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", choices=["tb2", "tau2"])
    ap.add_argument("--out", type=Path)
    ap.add_argument("--workdir", type=Path, default=Path("tb2_git_tmp"), help="scratch git repo for tb2")
    a = ap.parse_args()
    if a.source == "tb2":
        out = a.out or Path("../data/raw/tb2")
        rows = fetch_tb2(out, a.workdir)
    else:
        out = a.out or Path("../data/raw/tau2")
        paths = tau2_paths()
        url_for = lambda p: f"https://raw.githubusercontent.com/{TAU2_REPO}/{TAU2_REV}/{p}"
        rows = download(paths, url_for, out, TAU2_ROOT)
    out.mkdir(parents=True, exist_ok=True)
    with open(out.parent.parent / f"{a.source}_manifest.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["path", "bytes", "sha256"])
        w.writerows(rows)
    print(f"{len(rows)} files, {sum(r[1] for r in rows)/1e6:.1f} MB")


if __name__ == "__main__":
    main()
