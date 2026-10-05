import os, hashlib, csv
BS = chr(92)
rows = []
for root, _, fs in os.walk('data/raw/leaderboards'):
    for f in fs:
        p = os.path.join(root, f).replace(BS, '/')
        if p.endswith('wayback_lookup.json'):
            continue
        b = open(p, 'rb').read()
        rows.append((p[len('data/raw/'):], len(b), hashlib.sha256(b).hexdigest()))
rows.sort()
with open('data/leaderboards_manifest.csv', 'w', newline='') as g:
    w = csv.writer(g)
    w.writerow(['path', 'bytes', 'sha256'])
    w.writerows(rows)
print(len(rows), sum(r[1] for r in rows) / 1e6, 'MB')
print('manifest sha256', hashlib.sha256(open('data/leaderboards_manifest.csv', 'rb').read()).hexdigest())
for r in rows:
    if not r[0].endswith('.html') and 'browsergym/results' not in r[0]:
        print(r[2], r[0])
