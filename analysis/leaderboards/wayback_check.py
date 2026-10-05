"""Look up existing Wayback Machine captures (read-only; never calls /save). Writes data/raw/leaderboards/wayback_lookup.json"""
import json, sys, time, urllib.parse, requests
urls = {
 'swe-bench': 'https://github.com/swe-bench/experiments',
 'tau2-bench': 'https://github.com/sierra-research/tau2-bench/tree/main/web/leaderboard/public/submissions',
 'tau2-bench-site': 'https://taubench.com',
 'terminal-bench-2.0': 'https://huggingface.co/datasets/harborframework/terminal-bench-2-leaderboard',
 'terminal-bench-1.0': 'https://github.com/laude-institute/terminal-bench-leaderboard',
 'theagentcompany': 'https://github.com/TheAgentCompany/experiments',
 'appworld': 'https://github.com/stonybrooknlp/appworld-leaderboard',
 'appworld-site': 'https://appworld.dev/leaderboard',
 'gaia-results': 'https://huggingface.co/datasets/gaia-benchmark/results_public',
 'browsergym': 'https://huggingface.co/spaces/ServiceNow/browsergym-leaderboard',
 'webarena-sheet': 'https://docs.google.com/spreadsheets/d/1M801lEpBbKSNwP-vDBkC_pF7LdyGU1f_ufZb_NWNBZQ/edit?usp=sharing',
 'androidworld-sheet': 'https://docs.google.com/spreadsheets/d/1cchzP9dlTZ3WXQTfYNhh3avxoLipqHN75v1Tb86uhHo/edit?gid=0',
 'agentbench-sheet': 'https://docs.google.com/spreadsheets/d/e/2PACX-1vRR3Wl7wsCgHpwUw1_eUXW_fptAPLL3FkhnW_rua0O1Ji_GIVrpTjY5LaKAhwO-WeARjnY_KNw0SYNJ/pubhtml',
 'mle-bench': 'https://github.com/openai/mle-bench',
 'bfcl': 'https://gorilla.cs.berkeley.edu/data_overall.csv',
 'bfcl-site': 'https://gorilla.cs.berkeley.edu/leaderboard.html',
 'osworld-site': 'https://os-world.github.io/',
 'osworld-xlsx': 'https://github.com/os-world/os-world.github.io/blob/main/static/data/osworld_verified_results.xlsx',
 'aider-polyglot': 'https://aider.chat/docs/leaderboards/',
}
for b in ['assistantbench','corebench_hard','gaia','online_mind2web','scicode','scienceagentbench','swebench_verified_mini','taubench_airline','usaco']:
    urls['hal-'+b] = 'https://hal.cs.princeton.edu/' + b
out = {}
for k, u in urls.items():
    r = None
    for _ in range(3):
        try:
            r = requests.get('https://archive.org/wayback/available', params={'url': u}, timeout=60); r.raise_for_status(); break
        except Exception as e:
            r = None; time.sleep(30)
    if r is None: out[k] = {'url': u, 'error': 'lookup failed'}; continue
    snap = (r.json().get('archived_snapshots') or {}).get('closest')
    out[k] = {'url': u, 'capture': snap and {'url': snap['url'], 'timestamp': snap['timestamp'], 'status': snap['status']}}
    time.sleep(4)
json.dump(out, open('data/raw/leaderboards/wayback_lookup.json', 'w'), indent=1)
for k, v in out.items(): print(k, '->', (v.get('capture') or {}).get('timestamp') if v.get('capture') else v.get('error', 'no existing capture'))
