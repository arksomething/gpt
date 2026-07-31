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


def load_eval_prompts() -> Dict[str, List[str]]:
    """case_id -> user turns. samples.jsonl stores responses but not prompts."""
    out: Dict[str, List[str]] = {}
    for row in _read_jsonl(os.path.join(REPO_ROOT, "evals", "conversation", "v1.jsonl")):
        cid = row.get("id")
        if cid:
            out[cid] = [str(t.get("user", "")) for t in (row.get("turns") or [])]
    return out


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
        "uniform_loss": uniform_source_loss(d)[0],
        "per_source": uniform_source_loss(d)[1],
    }


CONVERSATION_SOURCES = {
    "stackexchange", "youtube", "ubuntu_irc", "github_archive", "uk_hansard",
}


def uniform_source_loss(d: str) -> tuple[Optional[float], Dict[str, float]]:
    """Mean loss over every source, each scored on equal tokens.

    This is the only cross-arm-comparable number here. Training loss and the
    mixture-weighted validation loss are each measured on that arm's *own*
    sampling distribution (validation_source_weights inherits source_weights
    when unset), so an arm that trains on more predictable text scores lower
    without being a better model. source_eval evaluates all sources with
    163,840 tokens each, identically for every arm.
    """
    rows = _read_jsonl(os.path.join(d, "source_eval.jsonl"))
    if not rows:
        return None, {}
    srcs = rows[-1].get("sources") or {}
    per = {k: v["loss"] for k, v in srcs.items() if isinstance(v, dict) and "loss" in v}
    if not per:
        return None, {}
    return statistics.fmean(per.values()), per


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
    present = [a for a in arms if not a["missing"] and a.get("uniform_loss") is not None]
    baseline = [a for a in present if a["arm"].startswith("b0-")]
    b0_mean = (
        statistics.fmean([a["uniform_loss"] for a in baseline]) if baseline else None
    )
    b0_per_source: Dict[str, float] = {}
    if baseline:
        keys = set(baseline[0].get("per_source") or {})
        for k in keys:
            vals = [b["per_source"][k] for b in baseline if k in (b.get("per_source") or {})]
            if vals:
                b0_per_source[k] = statistics.fmean(vals)

    rows = []
    for a in sorted(arms, key=lambda x: x["arm"]):
        if a["missing"]:
            rows.append(
                f'<tr><td class="mono">{e(a["arm"])}</td>'
                f'<td colspan="7" class="muted">not collected</td></tr>'
            )
            continue
        fl = a.get("uniform_loss")
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

        # Built as a list of cells: an inline ternary over a concatenated
        # f-string applies to the whole string, so one missing value silently
        # collapsed the entire row to a single cell.
        sft = a.get("sft_loss")
        cost = a.get("cost")
        cells = [
            f'<td class="mono">{e(a["arm"])}</td>',
            f'<td class="curve">{sparkline(a["loss_curve"])}</td>',
            f'<td class="num">{fl:.4f}</td>',
            f'<td class="num {cls}">'
            f'{("%+.4f" % delta) if delta is not None else "&mdash;"}</td>',
            f'<td class="{cls}">{verdict}</td>',
            f'<td class="num">{sft:.3f}</td>' if sft is not None
            else '<td class="num muted">&mdash;</td>',
            f'<td class="small">{e(", ".join(rlbits)) if rlbits else "&mdash;"}</td>',
            f'<td class="num">${cost:.2f}</td>' if cost is not None
            else '<td class="num muted">&mdash;</td>',
            f'<td class="small {"bad" if failed else "good"}">'
            f'{e(", ".join(failed)) if failed else "all ok"}</td>',
        ]
        rows.append("<tr>" + "".join(cells) + "</tr>")

    per_source_arms = [
        a for a in sorted(arms, key=lambda x: x["arm"])
        if not a["missing"] and a.get("per_source") and not a["arm"].startswith("b0-")
    ]
    ps_rows = []
    for src in sorted(b0_per_source, key=lambda k: -abs(
        (per_source_arms[0]["per_source"].get(k, b0_per_source[k]) - b0_per_source[k])
        if per_source_arms else 0.0)):
        dot = "&bull; " if src in CONVERSATION_SOURCES else ""
        cells = ""
        for a in per_source_arms:
            v = (a.get("per_source") or {}).get(src)
            if v is None:
                cells += '<td class="num muted">&mdash;</td>'
            else:
                dv = v - b0_per_source[src]
                c = "good" if dv < -0.02 else ("bad" if dv > 0.02 else "muted")
                cells += f'<td class="num {c}">{dv:+.4f}</td>'
        ps_rows.append(f'<tr><td>{dot}{e(src)}</td>{cells}</tr>')
    per_source_rows = "".join(ps_rows)

    prompts = load_eval_prompts()
    sample_blocks = []
    for a in sorted(arms, key=lambda x: x["arm"]):
        if a["missing"] or not a.get("samples"):
            continue
        turns = []
        for s in a["samples"][:4]:
            cid = s.get("case_id") or s.get("id") or ""
            ptxt = prompts.get(cid, [])
            for i, t in enumerate((s.get("turns") or [])[:2]):
                u = e(ptxt[i][:200] if i < len(ptxt) else f"[{cid}]")
                r = e(str(t.get("assistant", t.get("response", "")))[:340])
                det = t.get("deterministic") or {}
                wc = det.get("word_count")
                ok = det.get("passed")
                badge = ""
                if ok is not None:
                    badge = (
                        f'<span class="{"good" if ok else "bad"} small">'
                        f'{"checks passed" if ok else "checks failed"}'
                        f'{f" &middot; {wc}w" if wc else ""}</span>'
                    )
                turns.append(
                    f'<div class="turn"><div class="u">{u} {badge}</div>'
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

<h2>Per-source deltas vs baseline</h2>
<p class="lede">Where each arm gained and lost. Conversation sources are marked
&bull;. This is the view that decides a mixture: an arm can be flat overall while
moving a register you care about.</p>
<div class="card"><table>
<tr><th>source</th>{"".join(f'<th class="num">{e(a["arm"])}</th>' for a in per_source_arms)}</tr>
{per_source_rows}
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
        a["uniform_loss"]
        for a in arms
        if not a["missing"] and a["arm"].startswith("b0-") and a.get("uniform_loss")
    ]
    sigma = statistics.stdev(seeds) if len(seeds) >= 2 else None

    out = os.path.join(REPO_ROOT, args.out)
    os.makedirs(os.path.dirname(out), exist_ok=True)
    with open(out, "w") as f:
        f.write(build_html(arms, sigma))
    print(f"wrote {out} ({len(arms)} arms, sigma={sigma})")


if __name__ == "__main__":
    main()
