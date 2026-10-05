# Run-to-run reliability harness (illustration)

A small, $0 harness that runs 20 deterministic tool-use tasks, 5 repeats each, against three
Groq free-tier models, and logs every run to JSONL. It is an **illustration**: it shows whether
run-to-run spread exists on pinned models under fixed settings. It is not a frequency estimate
and not a benchmark of the models. MIT licensed. It does not include or import code from any
other project.

## Task source and why

Decided on 2026-10-04 by feasibility check; the reasons are recorded here.

1. **tau2-bench (sierra-research/tau2-bench, MIT, commit `5bfa7e3`) rejected.** Its no-user-simulator
   mode (`llm_agent_solo`) needs tasks with a `ticket` field. In the checked-out data, 0 of 50 airline
   tasks and 0 of 114 retail tasks have one; the 20-task `telecom/tasks_small.json` does. Telecom solo
   would also need Python >= 3.12 plus litellm, and prompts built from a multi-KB policy and a large tool set.
   Groq free-tier limits are 8K tokens per minute and 200K tokens per day per model (Groq rate-limits
   page, fetched 2026-10-04). My estimate, not measured, is that 300 telecom runs would not fit in four days.
2. **`@modelcontextprotocol/server-everything` rejected.** Its tools are stateless demo tools, so
   there is no end state to check programmatically, and its tool list is large for the same token budget.
   This was not tested.
3. **Chosen: 20 deterministic tasks against nine local Python tools** (`world.py`, `tasks.py`): an
   in-memory order-management world (items, customers, orders, notes). Each run starts from a fresh deep
   copy of a fixed state. Success requires the complete final state to equal the state produced by a
   reference action sequence, plus an answer check for question tasks, so unrequested side effects
   fail a task. No randomness, no network apart from the model call.

Task mix (`category` in `tasks.py`): 5 lookup, 3 write, 3 refusal, 3 conditional, 6 multi-step.

### Task revision (disclosed)

The first 100-run pass used `t04` and `t15`. After seeing results I found both were defective
specifications: `t04` ("Order 2 gizmos for carol.") did not forbid restocking, and models restocked to
fulfil it; `t15` ("Which customers are gold tier?") could not be answered because no tool enumerates
customers. They were replaced by `t21` (adds "Do not change stock levels.") and `t22` (a yes/no question
answerable with `get_customer`). The 26 original runs of `t04` and `t15` are kept unmodified in
`results/superseded_runs.jsonl` and are excluded from `runs.jsonl` and `summary.csv`. This is a post-hoc
change to the task set, prompted by looking at outcomes.

## Models (pinned IDs)

`GET https://api.groq.com/openai/v1/models` on 2026-10-04 listed `openai/gpt-oss-20b`,
`openai/gpt-oss-120b`, `qwen/qwen3.8-27b`, `openai/gpt-oss-safeguard-20b` (a safety-classifier model,
not tested for tool use here), plus speech, guard and Arabic-model entries. `qwen/qwen3-32b` from the earlier smoke test returned
HTTP 404 `model_not_found` on that date, so `qwen/qwen3.8-27b` replaces it. All three tested models
returned tool calls in a smoke test. Models used:

| Model ID | Run date(s) (UTC) |
|---|---|
| `openai/gpt-oss-20b` | 2026-10-04 |
| `openai/gpt-oss-120b` | 2026-10-04 |
| `qwen/qwen3.8-27b` | 2026-10-04 (partial; resumes on later dates, see `date` in `runs.jsonl`) |

Free-tier limits per model at the time: 30 RPM, 1K RPD, 8K TPM, 200K TPD. No billing was enabled.

## Fixed settings

- Same system prompt for every run (`SYSTEM_PROMPT` in `tasks.py`), same nine tool schemas, `tool_choice: auto`.
- Condition `default`: the `temperature` parameter is **omitted**, so the provider default applies. Groq's API
  reference states the default is 1 ("Defaults to 1"); the response does not echo it. Reasoning settings are
  also provider defaults. Records carry `temperature: null` and `temperature_setting: "provider_default"`.
- Condition `temp0`: `temperature: 0` (see Status for coverage).
- `max_completion_tokens` 2048, at most 10 model calls per run (`steps`), 60 s request timeout, no seed.
- Transient failures: HTTP 429 per-minute limits are backed off and retried; 5xx and timeouts get 2 retries.

## Definitions

- **success**: final world state equals the reference end state and, where defined, the final answer passes the check. Runs that end in an API error are `success = false`.
- **steps**: number of model calls in the run. **tool_calls**: every tool call the model emitted, valid or not.
- **invalid tool call**: unknown tool name, arguments that are not a JSON object, arguments that violate the schema (missing, extra, or wrongly typed, including a string where an integer is required), or a Groq HTTP 400 `tool_use_failed`. The model receives an error message and can continue. A valid call that the world rejects (for example insufficient stock) is a `tool_error`, not invalid.
- **failed run** (`failed_run_count`): the run ended in an unrecovered API error (`error` is not null). These are logged, never dropped, and count as unsuccessful. Task failures (wrong end state, empty reply, `max_steps`) are not failed runs.
- **pass^k** (per task, c successes in n runs): C(c,k)/C(n,k); the model value is the mean over tasks with at least k runs.
- **share of tasks with mixed outcomes**: tasks with at least one success and at least one failure, divided by tasks with at least 2 runs.
- **invalid_tool_call_rate**: invalid tool calls divided by all tool calls emitted.
- A run interrupted by a daily-quota stop is not logged as a run; it is listed in `results/quota_events.jsonl` and re-run on resume.

## Commands

Python 3.11 was used. From this `harness/` directory:

```bash
pip install -r requirements.txt            # requests==2.33.1, pytest==8.4.2
python -m pytest tests -q                  # tests write only to pytest temp dirs; no network
export GROQ_API_KEY=...                    # PowerShell: $env:GROQ_API_KEY = "..."  (never stored by this code)
python run.py --condition default          # 20 tasks x 5 repeats x 3 models; resumable
python run.py --condition temp0            # optional second condition
python summarize.py                        # writes results/summary.csv and results/per_task.csv from runs.jsonl
```

`run.py` exits with code 3 if any model hit its daily quota; run it again after the quota resets and it
continues from `runs.jsonl`. Options: `--models`, `--tasks`, `--repeats`, `--limit`, `--out`.

## Outputs

- `results/runs.jsonl`: one record per run: model, UTC date and timestamp, condition, temperature setting, task_id, repeat,
  success, steps, tool_calls, invalid_tool_calls (with breakdown), tool_errors, prompt/completion/total tokens,
  API latency and wall time, retries and retry events, error and error detail, stop reason, final answer
  (truncated to 500 characters), and the list of calls.
- `results/summary.csv`: per condition and model: mean success, per-task success counts, pass^1..pass^5, mixed-outcome share, invalid-call rate, failed-run count, plus token and latency means.
- `results/per_task.csv`: successes and runs per task and model.
- `results/superseded_runs.jsonl`, `results/quota_events.jsonl`: audit trail. The console and resume logs of the original run directory are not part of this release.

## Status at 2026-10-04

Both conditions were run on 2026-10-04 (UTC). Groq's daily token limit behaved as a rolling window, so a
re-run a few hours after the first stop got further runs.

- `default`: `openai/gpt-oss-20b` 100 of 100, `openai/gpt-oss-120b` 100 of 100, `qwen/qwen3.8-27b` 66 of 100 (stopped at its 200K-token daily limit; 34 pending).
- `temp0`: `openai/gpt-oss-20b` 100 of 100, `openai/gpt-oss-120b` 100 of 100, `qwen/qwen3.8-27b` 0 of 100 (limit reached before any run completed).

Resume with `python run.py --condition default` and then `--condition temp0` once the quota allows. Read
current numbers from `results/summary.csv`; they are not repeated here so they cannot go stale. Before the
main run, a 9-run pilot (tasks `t03`, `t05`, `t17`, one repeat, three models) measured token use. It was
written outside `results/` and is not in `runs.jsonl`.

## Limitations

Twenty tasks in one synthetic domain with one prompt and nine tools; two models are from the same family; the
free tier changes without notice, and model IDs may be retired (`qwen/qwen3-32b` already was). Success depends on
strict end-state matching and a few exact-format answer checks. Results show that spread exists or does not exist
on these tasks; they do not estimate how often it occurs elsewhere.
