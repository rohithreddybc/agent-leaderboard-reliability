"""Entry counts and entry-level exclusions for research/leaderboard-inventory.md.
Reads only files under data/raw/leaderboards. Counts only; no bound or statistic.
Writes research/leaderboard-entry-exclusions.csv, data/raw/leaderboards/inventory_counts.json
and cleaned HAL tables (hal/*_entries.csv)."""
import csv
import collections
import glob
import json
import re

import openpyxl
import pandas as pd
import yaml

R = 'data/raw/leaderboards'
excl = []   # (leaderboard, entry, reason)
cnt = {}


def add(lb, entry, reason):
    excl.append((lb, entry, reason))


def cons(v, n, dec=1):
    k = round(v * n / 100)
    return abs(round(k * 100 / n, dec) - v) < 1e-9


# SWE-bench splits
for s, n in {'verified': 500, 'lite': 300, 'test': 2294}.items():
    d = json.load(open(f'{R}/swe-bench/swebench_{s}.json'))
    inc = 0
    for k, r in d.items():
        if r['source'] is None:
            add(f'SWE-bench {s}', k, 'no per-instance result file in the repository')
            continue
        inc += 1
    att = collections.Counter(str((r.get('meta') or {}).get('attempts')) for r in d.values() if r['source'])
    cnt[f'SWE-bench {s}'] = dict(dirs=len(d), included=inc, n=n, attempts=dict(att))
d = json.load(open(f'{R}/swe-bench/swebench_multilingual.json'))
n = 300
inc = 0
for k, r in d.items():
    c = set(r.get('covered_ids') or [])
    if len(c) != n or 'mrdoob__three' in c:
        why = f'file lists {len(c)} instance ids, not the 300 listed by the other entries'
        if 'mrdoob__three' in c:
            why += '; includes id mrdoob__three, absent from all other entries'
        add('SWE-bench multilingual', k, why)
        continue
    inc += 1
cnt['SWE-bench multilingual'] = dict(dirs=len(d), included=inc, n=n)
d = json.load(open(f'{R}/swe-bench/swebench_multimodal.json'))
cnt['SWE-bench multimodal'] = dict(
    dirs=len(d), included=0, n=None,
    with_id_lists=sum(1 for r in d.values() if r.get('resolved')),
    count_only=sum(1 for r in d.values() if r.get('resolved_count_field') is not None))

# TheAgentCompany
t = json.load(open(f'{R}/theagentcompany/tac_results.json'))['results']['1.0.0']
inc = 0
for e, tasks in sorted(t.items()):
    ids = {x.replace('-image', '') for x in tasks}
    if len(ids) != 175:
        add('TheAgentCompany', e, f'{len(ids)} of 175 task result files present')
        continue
    inc += 1
cnt['TheAgentCompany'] = dict(entries=len(t), included=inc, n=175)

# AppWorld
a = json.load(open(f'{R}/appworld/_leaderboard.json'))
bad = [(e['id'], e['method']['name'], e['llm']['name'], e['test_challenge']['all']['task_goal_completion'])
       for e in a if not cons(e['test_challenge']['all']['task_goal_completion'], 417)]
cnt['AppWorld'] = dict(
    entries=len(a),
    test_normal_consistent_168=sum(cons(e['test_normal']['all']['task_goal_completion'], 168) for e in a),
    test_challenge_consistent_417=sum(cons(e['test_challenge']['all']['task_goal_completion'], 417) for e in a),
    dates=[min(e['date'] for e in a), max(e['date'] for e in a)])
for b in bad:
    add('AppWorld test_challenge', f'{b[1]} / {b[2]} (id {b[0]})',
        f'TGC {b[3]} is not a multiple of 1/417 at one decimal; reported value inconsistent with n=417')

# BrowserGym
bg = collections.defaultdict(list)
for f in glob.glob(f'{R}/browsergym/results/*/*.json'):
    for r in json.load(open(f, encoding='utf-8')):
        bg[r['benchmark']].append(r)
nn = {'WebArena': 812, 'WorkArena-L1': 330, 'WorkArena-L2': 235, 'WorkArena-L3': None, 'VisualWebArena': 910}
for b, rs in bg.items():
    n = nn.get(b)
    cnt['BrowserGym ' + b] = dict(
        entries=len(rs), n=n,
        consistent=(sum(cons(r['score'], n) for r in rs) if n else None),
        not_following_protocol=sum(r['followed_evaluation_protocol'] != 'Yes' for r in rs),
        tuned=sum(r['benchmark_tuned'] != 'No' for r in rs))
    if n:
        for r in rs:
            if not cons(r['score'], n):
                add('BrowserGym ' + b, r['agent_name'], f"score {r['score']} not a multiple of 1/{n} at one decimal")

# MLE-bench README table
md = open(f'{R}/mlebench/README.md', encoding='utf-8').read()
main = md.split('### Additional Leaderboard Submissions')[0]
rows = [l for l in main.splitlines() if l.startswith('| ') and not l.startswith('| Agent') and not l.startswith('|---')]
cnt['MLE-bench'] = dict(main_rows=len(rows), padded_incomplete_seeds=sum('[^3]' in l for l in rows))
add2 = md.split('### Additional Leaderboard Submissions')[1].split('[^2]')[0]
extra = [l for l in add2.splitlines() if l.startswith('| ') and not l.startswith('| Agent') and not l.startswith('|---')]
for l in extra:
    nm = re.sub(r'\]\([^)]*\)', '', l.split('|')[1]).replace('[', '').strip()
    add('MLE-bench', nm + ' / ' + l.split('|')[2].strip(),
        'listed under Additional Leaderboard Submissions: not directly comparable (test-set feedback)')
cnt['MLE-bench']['additional_rows'] = len(extra)

# HAL tables: drop the JavaScript template row
for f in sorted(glob.glob(f'{R}/hal/*_table.csv')):
    rows = list(csv.reader(open(f, encoding='utf-8')))
    h = rows[0]
    body = [r for r in rows[1:] if not any('${' in c for c in r)]
    names = [c.split(' ')[0] for c in h]
    with open(f.replace('_table.csv', '_entries.csv'), 'w', newline='', encoding='utf-8') as g:
        w = csv.writer(g)
        w.writerow(names)
        w.writerows(body)
    ri = names.index('Runs')
    cnt['HAL ' + f.replace('\\', '/').split('/')[-1].replace('_table.csv', '')] = dict(
        entries=len(body), runs=dict(collections.Counter(r[ri] for r in body)))

# GAIA
for s in ('test', 'validation'):
    df = pd.read_parquet(f'{R}/gaia/results_public/2023/{s}-00000-of-00001.parquet')
    cnt['GAIA ' + s] = dict(rows=len(df), distinct_model_org=int(df[['model', 'organisation']].drop_duplicates().shape[0]),
                            dates=[df['date'].min(), df['date'].max()])

# OSWorld-Verified
ws = openpyxl.load_workbook(f'{R}/osworld/osworld_verified_results.xlsx', data_only=True)['Eval Results']
rows = [r for r in ws.iter_rows(min_row=2, values_only=True) if r[0] is not None]
tot = collections.Counter()
frac = 0
for r in rows:
    try:
        a_, b_ = str(r[11]).split('/')
        tot[int(float(b_))] += 1
        frac += abs(float(a_) - round(float(a_))) > 1e-9
    except Exception:
        pass
cnt['OSWorld-Verified'] = dict(rows=len(rows), totals=dict(tot), fractional_numerators=int(frac))

# BFCL, AndroidWorld, WebArena sheet, Aider
cnt['BFCL'] = dict(rows=len(pd.read_csv(f'{R}/bfcl/data_overall.csv')))
a = pd.read_csv(f'{R}/androidworld/leaderboard_gid0.csv', skiprows=1)
a.columns = [c.replace('\n', ' ') for c in a.columns]
s = pd.to_numeric(a['Success Rate (pass@1)'], errors='coerce').dropna()
for _, r in a.iterrows():
    v = pd.to_numeric(r['Success Rate (pass@1)'], errors='coerce')
    if pd.isna(v):
        continue
    tr = r['Number of trials']
    ne = 116 * (int(tr) if pd.notna(tr) else 1)
    if not cons(v, ne):
        add('AndroidWorld', f"{r['Model']} / {r['Result Source']}", f'score {v} not a multiple of 1/{ne} at one decimal (n=116 x {int(tr) if pd.notna(tr) else 1} trials); possible different task set or rounding')
cnt['AndroidWorld'] = dict(rows_with_score=len(s), consistent_116=int(sum(cons(x, 116) for x in s)),
                           trials={str(k): int(v) for k, v in a['Number of trials'].value_counts(dropna=False).items()})
w = pd.read_csv(f'{R}/webarena/leaderboard_gid0.csv')
s = pd.to_numeric(w['Success Rate (%)'], errors='coerce').dropna()
for _, r in w.iterrows():
    v = pd.to_numeric(r['Success Rate (%)'], errors='coerce')
    if pd.notna(v) and not cons(v, 812):
        add('WebArena sheet', f"{r['Model']} / {r['Result Source']}", f'score {v} not a multiple of 1/812 at one decimal; possible subset of tasks or rounding')
cnt['WebArena sheet'] = dict(rows_with_score=len(s), consistent_812=int(sum(cons(x, 812) for x in s)))
y = yaml.safe_load(open(f'{R}/aider/polyglot_leaderboard.yml', encoding='utf-8'))
cnt['Aider polyglot'] = dict(entries=len(y), test_cases=dict(collections.Counter(str(x['test_cases']) for x in y)))

# tau2-bench (data/raw/tau2): text entries per domain; voice submissions and pre-1.0.1 banking_knowledge excluded
import os
T2 = 'data/raw/tau2'
man = json.load(open(f'{T2}/manifest.json'))
grp = {e: g for g, l in man.items() for e in l}
t2 = collections.defaultdict(list)
for e in sorted(os.listdir(T2)):
    f = os.path.join(T2, e, 'submission.json')
    if not os.path.exists(f) or e.startswith('A_EXAMPLE'):
        continue
    d = json.load(open(f, encoding='utf-8'))
    ver = str((d.get('methodology') or {}).get('tau2_bench_version'))
    for dom, r in d['results'].items():
        if not (r and r.get('pass_1') is not None):
            continue
        if grp.get(e) == 'voice_submissions':
            add('tau2-bench ' + dom, e, 'voice submission (different modality and protocol)')
            continue
        if dom == 'banking_knowledge' and ver != '1.0.1':
            add('tau2-bench ' + dom, e, f'tau2-bench version {ver}: banking_knowledge scores before 1.0.1 are not comparable (CHANGELOG 1.0.1)')
            continue
        t2[(dom, ver)].append(e)
cnt['tau2-bench (text entries included, by domain and version)'] = {f'{k[0]} | {k[1]}': len(v) for k, v in sorted(t2.items())}

json.dump(cnt, open(f'{R}/inventory_counts.json', 'w'), indent=1, default=str)
with open('research/leaderboard-entry-exclusions.csv', 'w', newline='', encoding='utf-8') as g:
    w = csv.writer(g)
    w.writerow(['leaderboard', 'entry', 'reason'])
    w.writerows(excl)
for k, v in cnt.items():
    print(k, v)
print(len(excl), 'entry-level exclusions')
