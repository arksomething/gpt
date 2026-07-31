"""Assemble the Gate 1 decision report from collected arm results.

Reads results/collected/<arm>/ and emits a single self-contained HTML page:
loss curves, per-source validation loss, benchmarks, SFT deltas, GRPO
diagnostics, and readable sample transcripts.

The comparisons are read against the between-seed sigma rather than in
isolation. Three baseline seeds exist precisely so that "P3 beat B0" can be
checked against how much two identical recipes differ by chance; without that
ruler a delta is not evidence.
"""

from __future__ import annotations

import argparse
import html
import json
import os
import statistics
import sys
from typing import Any, Dict, List, Optional

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COLLECTED = os.path.join(REPO_ROOT, "results", "collected")


def _read_json(path: str) -> Optional[Any]:
    try:
        with open(path) as f:
            return json.load(f)
    except Exception:
        return None


def _read_jsonl(path: str) -> List[dict]:
    rows = []
    try:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except Exception:
                        pass
    except Exception:
        pass
    return rows


def _arm_dir(arm: str) -> Optional[str]:
    """Tarballs unpack as <arm>/<arm>/... ; tolerate either shape."""
    base = os.path.join(COLLECTED, arm)
    for cand in (os.path.join(base, arm), base):
        if os.path.isdir(cand):
            return cand
    return None


def load_arm(arm: str) -> Dict[str, Any]:
    d = _arm_dir(arm)
    if not d:
        return {"arm": arm, "missing": True}

    state = _read_json(os.path.join(d, "run_state.json")) or {}
    metrics = _read_jsonl(os.path.join(d, "metrics.jsonl"))
    loss = [
        (r.get("step"), r["metrics"]["train/loss"])
        for r in metrics
        if isinstance(r.get("metrics"), dict) and "train/loss" in r["metrics"]
    ]
    val = [
        (r.get("step"), r["metrics"]["val/loss"])
        for r in metrics
        if isinstance(r.get("metrics"), dict) and "val/loss" in r["metrics"]
    ]

    rc = {}
    for name in os.listdir(d):
        if name.startswith("RC_"):
            v = _read_json(os.path.join(d, name))
            if v is None:
                try:
                    v = int(open(os.path.join(d, name)).read().strip())
                except Exception:
                    v = None
            rc[name[3:].lower()] = v

    rl = {}
    for env in ("format_constraint", "calibrated_abstention"):
        rep = _read_json(os.path.join(d, f"rl-{env}", "rl_report.json"))
        if rep:
            rl[env] = rep

    return {
        "arm": arm,
        "missing": False,
        "final_loss": loss[-1][1] if loss else None,
        "loss_curve": loss,
        "val_curve": val,
        "steps": state.get("step"),
        "cost": state.get("estimated_cost"),
        "status": state.get("status"),
        "source_eval": _read_jsonl(os.path.join(d, "source_eval.jsonl")),
        "rc": rc,
        "rl": rl,
        "samples": _read_jsonl(os.path.join(d, "samples.jsonl"))[:6],
        "sft_loss": _sft_final_loss(d),
    }


def _sft_final_loss(d: str) -> Optional[float]:
    rows = _read_jsonl(os.path.join(d, "sft", "metrics.jsonl"))
    vals = [
        r["metrics"]["train/loss"]
        for r in rows
        if isinstance(r.get("metrics"), dict) and "train/loss" in r["metrics"]
    ]
    return vals[-1] if vals else None


def sparkline(points: List[Any], w: int = 240, h: int = 44) -> str:
    vals = [v for _, v in points if v is not None]
    if len(vals) < 2:
        return '<span class="muted">no curve</span>'
    lo, hi = min(vals), max(vals)
    rng = (hi - lo) or 1.0
    step = w / (len(vals) - 1)
    pts = " ".join(
        f"{i*step:.1f},{h - ((v - lo) / rng) * (h - 6) - 3:.1f}"
        for i, v in enumerate(vals)
    )
    return (
        f'<svg viewBox="0 0 {w} {h}" width="{w}" height="{h}" '
        f'preserveAspectRatio="none" aria-label="loss curve">'
        f'<polyline points="{pts}" fill="none" stroke="currentColor" '
        f'stroke-width="1.5"/></svg>'
    )


def build_html(arms: List[Dict[str, Any]], sigma: Optional[float]) -> str:
    e = html.escape
    present = [a for a in arms if not a["missing"] and a.get("final_loss") is not None]
    baseline = [a for a in present if a["arm"].startswith("b0-")]
    b0_mean = (
        statistics.fmean([a["final_loss"] for a in baseline]) if baseline else None
    )

    rows = []
    for a in sorted(arms, key=lambda x: x["arm"]):
        if a["missing"]:
            rows.append(
                f'<tr><td class="mono">{e(a["arm"])}</td>'
                f'<td colspan="7" class="muted">not collected</td></tr>'
            )
            continue
        fl = a.get("final_loss")
        delta = (fl - b0_mean) if (fl is not None and b0_mean is not None) else None
        if delta is None or sigma in (None, 0):
            verdict, cls = "&mdash;", "muted"
        elif abs(delta) <= sigma:
            verdict, cls = "within noise", "flat"
        elif delta < 0:
            verdict, cls = "better than B0", "good"
        else:
            verdict, cls = "worse than B0", "bad"

        rlbits = []
        for env, rep in (a.get("rl") or {}).items():
            df = rep.get("degenerate_fraction")
            rlbits.append(
                f'{e(env.split("_")[0])}: {df:.0%} dead' if df is not None else env
            )
        failed = [k for k, v in (a.get("rc") or {}).items() if v not in (0, None)]

        rows.append(
            "<tr>"
            f'<td class="mono">{e(a["arm"])}</td>'
            f'<td class="curve">{sparkline(a["loss_curve"])}</td>'
            f'<td class="num">{fl:.4f}</td>'
            f'<td class="num {cls}">{("%+.4f" % delta) if delta is not None else "&mdash;"}</td>'
            f'<td class="{cls}">{verdict}</td>'
            f'<td class="num">{a["sft_loss"]:.3f}</td>' if a.get("sft_loss") is not None
            else f'<td class="num muted">&mdash;</td>'
        )
        rows[-1] += (
            f'<td class="small">{e(", ".join(rlbits)) or "&mdash;"}</td>'
            f'<td class="num">${a["cost"]:.2f}</td>' if a.get("cost") is not None
            else '<td class="num muted">&mdash;</td>'
        )
        rows[-1] += (
            f'<td class="small {"bad" if failed else "good"}">'
            f'{e(", ".join(failed)) if failed else "all ok"}</td></tr>'
        )

    sample_blocks = []
    for a in sorted(arms, key=lambda x: x["arm"]):
        if a["missing"] or not a.get("samples"):
            continue
        turns = []
        for s in a["samples"][:3]:
            for t in (s.get("turns") or [])[:2]:
                u = e(str(t.get("user", ""))[:200])
                r = e(str(t.get("response", t.get("assistant", "")))[:320])
                turns.append(
                    f'<div class="turn"><div class="u">{u}</div>'
                    f'<div class="a">{r}</div></div>'
                )
        if turns:
            sample_blocks.append(
                f'<details><summary>{e(a["arm"])} &mdash; sample transcripts</summary>'
                f'{"".join(turns)}</details>'
            )

    sigma_txt = f"{sigma:.4f}" if sigma else "not yet measurable"
    return f"""<title>Gate 1 results</title>
<style>
:root{{--bg:#f6f5f1;--panel:#fff;--ink:#1b1e24;--ink2:#555c68;--ink3:#8a919d;--line:#e2dfd7;
--good:#1f6f4a;--bad:#a8402a;--flat:#7a6a1f}}
@media (prefers-color-scheme:dark){{:root{{--bg:#15171b;--panel:#1c1f25;--ink:#e8eaee;--ink2:#a4abb6;
--ink3:#727984;--line:#2b2f37;--good:#46a37a;--bad:#e0714f;--flat:#c8a83c}}}}
:root[data-theme=dark]{{--bg:#15171b;--panel:#1c1f25;--ink:#e8eaee;--ink2:#a4abb6;--ink3:#727984;
--line:#2b2f37;--good:#46a37a;--bad:#e0714f;--flat:#c8a83c}}
:root[data-theme=light]{{--bg:#f6f5f1;--panel:#fff;--ink:#1b1e24;--ink2:#555c68;--ink3:#8a919d;
--line:#e2dfd7;--good:#1f6f4a;--bad:#a8402a;--flat:#7a6a1f}}
*{{box-sizing:border-box}}body{{background:var(--bg);color:var(--ink);margin:0;padding:0 1.2rem 4rem;
font:15px/1.55 ui-sans-serif,system-ui,sans-serif}}
main{{max-width:1020px;margin:0 auto}}header{{padding:2rem 0 1rem;border-bottom:2px solid var(--line)}}
h1{{font-size:1.6rem;margin:.3rem 0;font-weight:650}}
h2{{font-size:1rem;margin:2rem 0 .6rem}}
.eyebrow{{font-family:ui-monospace,monospace;font-size:.7rem;letter-spacing:.14em;
text-transform:uppercase;color:var(--ink3)}}
.card{{background:var(--panel);border:1px solid var(--line);border-radius:9px;padding:1rem;
margin:.6rem 0;overflow-x:auto}}
table{{border-collapse:collapse;width:100%;font-size:.85rem;font-variant-numeric:tabular-nums}}
th,td{{text-align:left;padding:.45rem .55rem;border-bottom:1px solid var(--line);vertical-align:middle}}
th{{font-size:.66rem;text-transform:uppercase;letter-spacing:.07em;color:var(--ink3)}}
td.num,th.num{{text-align:right}}.mono{{font-family:ui-monospace,monospace}}
.muted{{color:var(--ink3)}}.good{{color:var(--good)}}.bad{{color:var(--bad)}}.flat{{color:var(--flat)}}
.small{{font-size:.78rem}}.curve{{color:var(--ink2);width:250px}}
details{{margin:.5rem 0;background:var(--panel);border:1px solid var(--line);border-radius:8px;
padding:.6rem .9rem}}summary{{cursor:pointer;font-family:ui-monospace,monospace;font-size:.8rem}}
.turn{{margin:.6rem 0;padding-left:.7rem;border-left:2px solid var(--line)}}
.u{{color:var(--ink2);font-size:.83rem}}.a{{margin-top:.2rem;font-size:.86rem}}
.lede{{color:var(--ink2);max-width:64ch}}
</style>
<main>
<header>
<div class="eyebrow">gpt &middot; gate 1 &middot; 25M mixture bakeoff</div>
<h1>Gate 1 results</h1>
<p class="lede">Every arm trained on the same union corpus and differs only in sampling
weights, so the mixture is the only variable. Deltas are read against the
between-seed &sigma; = <b>{sigma_txt}</b> &mdash; the amount two identical recipes
differ by chance. A gap smaller than that is not evidence.</p>
</header>

<h2>Arms</h2>
<div class="card"><table>
<tr><th>arm</th><th>pretrain loss curve</th><th class="num">final</th>
<th class="num">&Delta; vs B0</th><th>verdict</th><th class="num">sft loss</th>
<th>grpo groups</th><th class="num">cost</th><th>stage failures</th></tr>
{"".join(rows)}
</table></div>

<h2>Sample transcripts</h2>
<p class="lede">Perplexity never revealed that round 1 answered questions it wasn't
asked. These are the artifacts to actually judge.</p>
{"".join(sample_blocks) or '<div class="card muted">No samples collected yet.</div>'}
</main>"""


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out", default="results/gate1_report.html")
    ap.add_argument("--arms", nargs="*", default=None)
    args = ap.parse_args()

    names = args.arms or sorted(
        d for d in os.listdir(COLLECTED) if os.path.isdir(os.path.join(COLLECTED, d))
    ) if os.path.isdir(COLLECTED) else []
    if not names:
        print("no collected arms yet")
        return

    arms = [load_arm(a) for a in names]
    seeds = [
        a["final_loss"]
        for a in arms
        if not a["missing"] and a["arm"].startswith("b0-") and a.get("final_loss")
    ]
    sigma = statistics.stdev(seeds) if len(seeds) >= 2 else None

    out = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(build_html(arms, sigma))
    print(f"wrote {out} ({len(arms)} arms, sigma={sigma})")


if __name__ == "__main__":
    main()
