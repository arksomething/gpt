# Training observability and run control

Every paid probe enables a local, dependency-free observability layer. W&B and
TensorBoard remain optional mirrors; the files in the run directory are the
authoritative record.

## Durable run files

Each seed output directory contains:

| File | Purpose |
|---|---|
| `run_state.json` | Atomically replaced current state and heartbeat |
| `events.jsonl` | Append-only lifecycle, evaluation, checkpoint, budget, and control events |
| `metrics.jsonl` | Append-only training, evaluation, source-loss, throughput, gradient, and GPU metrics |
| `train.log` | Human-readable stdout and stderr |
| `checkpoint_manifest.json` | Local and uploaded checkpoint inventory |
| `runtime_cursor.json` | Last consumed command byte, preventing replay after resume |

`run_state.json` reports status, step, total steps, processed tokens, active
runtime, estimated cost, latest evaluation, latest checkpoint, PID, host, and
the experiment identity. If the process exits without recording completion,
the normal Python exit path marks it failed. An abrupt machine loss leaves a
running state with a stale heartbeat, which `run-status` detects.

## Inspecting runs

```bash
uv run run-status runs/probes/<group-id>/seed-1337
uv run run-status --json \
  runs/probes/<group-id>/seed-1337 \
  runs/probes/<group-id>/seed-2027 \
  runs/probes/<group-id>/seed-4099

uv run experiment status <group-id>
```

For automation, add `--require-healthy`; the command exits nonzero for missing,
invalid, stale, failed, or budget-stopped runs. The default stale threshold is
15 minutes and can be changed with `--stale-after-seconds`.
`experiment status` aggregates all seeds and their observed cost against the
group cap.

## Runtime budget behavior

`budget.max_cost` is the cap for the complete experiment group. Planning writes
an equal `budget.run_max_cost` allocation into each immutable seed config. The
trainer estimates active compute cost from elapsed runtime and the configured
hourly rate.

At 50%, 80%, and 95% of a seed allocation, it records budget-alarm events. At
100%, all ranks stop at a step boundary, write recovery/final state, attempt
the configured verified upload, mark the run `budget_stopped`, and exit
nonzero. The experiment launcher then refuses to start the next seed.

This is a second line of defense. Provider-native billing alarms and automatic
instance shutdown are still required because no in-process monitor can act
after the entire machine or network disappears.

## Runtime commands

`runtime_control.yaml` can adjust only allowlisted settings. `commands.jsonl`
supports the existing save, pin, sample, and stop controls. Every poll commits
its byte offset to `runtime_cursor.json`; exact resume consumes only commands
appended after that offset.

Applied updates, stop requests, evaluations, checkpoints, and terminal state
are reflected in the event log. Do not edit the cursor by hand.
