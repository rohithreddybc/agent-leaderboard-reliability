"""TheAgentCompany/experiments (GitHub): fetch per-task result files (evaluation/1.0.0/*/results/*.json) by blob.
Consolidates to one JSON. Usage: python fetch_tac.py <workdir> <out_json>. No statistics computed."""
import json, os, re, subprocess, sys
work, out = sys.argv[1], sys.argv[2]
os.makedirs(work, exist_ok=True)
def git(*a, inp=None, check=True):
    p = subprocess.run(['git', '-c', 'core.longpaths=true', '-C', work, *a], capture_output=True, input=inp)
    if check and p.returncode: raise RuntimeError(p.stderr.decode(errors='replace')[:400])
    return p.stdout
if not os.path.exists(os.path.join(work, '.git')):
    git('init', '-q'); git('remote', 'add', 'origin', 'https://github.com/TheAgentCompany/experiments')
git('fetch', '-q', '--depth', '1', '--filter=blob:none', 'origin', 'main')
head = git('rev-parse', 'FETCH_HEAD').decode().strip()
rows = {}
for line in git('ls-tree', '-r', head).decode().splitlines():
    meta, path = line.split('\t', 1)
    m = re.match(r'evaluation/([^/]+)/([^/]+)/results/(eval_.+\.json)$', path)
    if m: rows[path] = (meta.split()[2], m.groups())
shas = sorted({v[0] for v in rows.values()})
print('commit', head, 'files', len(rows), 'blobs', len(shas))
for i in range(0, len(shas), 100):
    git('fetch', '-q', '--depth', '1', '--no-tags', 'origin', *shas[i:i+100], check=False)
res = {}
proc = subprocess.run(['git', '-C', work, 'cat-file', '--batch'], input=('\n'.join(shas)+'\n').encode(), capture_output=True)
buf, pos, blob = proc.stdout, 0, {}
while pos < len(buf):
    nl = buf.index(b'\n', pos); hdr = buf[pos:nl].decode().split()
    if hdr[1] == 'missing': pos = nl+1; continue
    size = int(hdr[2]); blob[hdr[0]] = buf[nl+1:nl+1+size]; pos = nl+1+size+1
print('blobs read', len(blob))
for path, (sha, (ver, entry, fn)) in rows.items():
    d = json.loads(blob[sha]) if sha in blob else None
    res.setdefault(ver, {}).setdefault(entry, {})[fn[len('eval_'):-5]] = d and {'final': d.get('final_score'), 'checkpoints': d.get('checkpoints')}
json.dump({'commit': head, 'results': res}, open(out, 'w'), indent=0, sort_keys=True)
