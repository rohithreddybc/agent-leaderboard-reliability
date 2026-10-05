"""Fetch selected files of a public git repo by blob (no checkout; works for paths that are invalid on Windows)
and write them to one JSON {commit, files: {path: parsed-json-or-text}}.
Usage: python fetch_blobs.py <repo_url> <ref> <path_regex> <workdir> <out_json>
No statistics are computed."""
import json, os, re, subprocess, sys
url, ref, rx, work, out = sys.argv[1:6]
rx = re.compile(rx)
os.makedirs(work, exist_ok=True)
def git(*a, check=True):
    p = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', work, *a], capture_output=True)
    if check and p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[:400])
    return p.stdout
if not os.path.exists(os.path.join(work, '.git')):
    git('init', '-q'); git('remote', 'add', 'origin', url)
git('fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', ref)
head = git('rev-parse', 'FETCH_HEAD').decode().strip()
rows = {}
for line in git('ls-tree', '-r', head).decode(errors='replace').splitlines():
    meta, path = line.split('\t', 1)
    if rx.search(path): rows[path] = meta.split()[2]
shas = sorted(set(rows.values()))
print('commit', head, 'files', len(rows), 'blobs', len(shas))
for i in range(0, len(shas), 100):
    git('fetch', '-q', '--depth', '1', '--no-tags', 'origin', *shas[i:i+100], check=False)
proc = subprocess.run(['git', '-C', work, 'cat-file', '--batch'], input=('\n'.join(shas)+'\n').encode(), capture_output=True)
buf, pos, blob = proc.stdout, 0, {}
while pos < len(buf):
    nl = buf.index(b'\n', pos); hdr = buf[pos:nl].decode().split()
    if hdr[1] == 'missing': pos = nl+1; continue
    size = int(hdr[2]); blob[hdr[0]] = buf[nl+1:nl+1+size]; pos = nl+1+size+1
print('blobs read', len(blob), 'missing', len(shas)-len(blob))
files = {}
for path, sha in rows.items():
    b = blob.get(sha)
    if b is None: files[path] = None; continue
    try: files[path] = json.loads(b)
    except Exception: files[path] = b.decode(errors='replace')
json.dump({'commit': head, 'repo': url, 'files': files}, open(out, 'w'), sort_keys=True)
