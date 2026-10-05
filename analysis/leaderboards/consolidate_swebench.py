"""Consolidate swe-bench/experiments entry files into one JSON per split.
Usage: python consolidate_swebench.py <clone_dir> <out_dir>
Reads evaluation/<split>/<entry>/{metadata.yaml, results/results.json, per_instance_details.json}.
Uses \?\ paths because the clone sits at a long Windows path. No statistics are computed.
"""
import json, os, sys, yaml
BS = chr(92)
clone, out = os.path.abspath(sys.argv[1]), os.path.abspath(sys.argv[2])
def P(*a):
    return BS*2 + '?' + BS + os.path.join(clone, *a).replace('/', BS)
LISTS = ('resolved','no_generation','generated','no_logs','applied','test_errored',
         'test_timeout','install_fail','no_apply','reset_failed','with_logs')
for split in ['verified','lite','test','multimodal','multilingual']:
    rows = {}
    for entry in sorted(os.listdir(P('evaluation', split))):
        rec = {'dir': entry}
        mp = P('evaluation', split, entry, 'metadata.yaml')
        if os.path.exists(mp):
            y = yaml.safe_load(open(mp, encoding='utf-8')) or {}
            t = y.get('tags') or {}
            rec['meta'] = {'name': (y.get('info') or {}).get('name'),
                           'resolved_pct_reported': (y.get('info') or {}).get('resolved'),
                           'attempts': (t.get('system') or {}).get('attempts'),
                           'checked': t.get('checked'), 'org': t.get('org'),
                           'os_model': t.get('os_model'), 'os_system': t.get('os_system'),
                           'agent': t.get('agent'), 'model': t.get('model'),
                           'warning': y.get('warning')}
        rp = P('evaluation', split, entry, 'results', 'results.json')
        dp = P('evaluation', split, entry, 'per_instance_details.json')
        if os.path.exists(rp):
            r = json.load(open(rp, encoding='utf-8'))
            cov = set()
            for k in LISTS:
                if isinstance(r.get(k), list): cov |= set(r[k])
            rec['source'] = 'results.json'
            rec['resolved'] = sorted(r['resolved']) if isinstance(r.get('resolved'), list) else None
            rec['resolved_count_field'] = r['resolved'] if isinstance(r.get('resolved'), int) else None
            rec['covered_ids'] = sorted(cov)
        elif os.path.exists(dp):
            d = json.load(open(dp, encoding='utf-8'))
            rec['source'] = 'per_instance_details.json'
            rec['covered_ids'] = sorted(d)
            rec['resolved'] = sorted(k for k, v in d.items() if v.get('resolved'))
        else:
            rec['source'] = None
        rows[entry] = rec
    json.dump(rows, open(os.path.join(out, f'swebench_{split}.json'), 'w'), indent=0, sort_keys=True)
    print(split, len(rows), sum(1 for r in rows.values() if r['source']))
