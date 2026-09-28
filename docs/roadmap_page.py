# SPDX-FileCopyrightText: 2026 Jan Owoc
# SPDX-License-Identifier: AGPL-3.0-or-later

"""Render the roadmap as one HTML page from the live GitHub issues.

Usage, from the repository root with ``gh`` authenticated::

    python docs/roadmap_page.py                 # writes docs/roadmap.html (gitignored)
    python docs/roadmap_page.py --standalone    # a full document for a browser
    python docs/roadmap_page.py --out path      # somewhere else

Without ``--standalone`` the output is a page fragment starting at ``<title>``,
the form the claude.ai artifact publisher expects. ``EXECUTION_ORDER`` is the
one thing here a human maintains: it is the order to work the issues in, which
is a decision, not something the tracker can supply.
"""

from __future__ import annotations

import argparse
import html
import json
import re
import subprocess
from pathlib import Path

REPO = "janowoc/northplan"

#: Every issue, in the order to work them. Done issues are listed in the order
#: they were completed. Append new issues where they belong, not at the end.
EXECUTION_ORDER = [
    1,
    2,
    3,
    25,
    4,
    26,
    5,
    27,
    9,
    6,
    28,
    7,
    8,
    10,
    11,
    12,
    13,
    30,
    31,
    37,
    32,
    38,
    33,
    29,
    34,
    14,
    39,
    15,
    16,
    40,
    41,
    42,
    43,
    44,
    45,
    46,
    47,
    17,
    48,
    18,
    49,
    51,
    52,
    54,
    58,
    59,
    53,
    19,
    50,
    55,
    60,
    35,
    36,
    57,
    20,
    61,
    62,
    63,
    64,
    70,
    65,
    66,
    21,
    67,
    56,
    68,
    22,
    23,
    24,
    69,
]

LIST_ITEM = re.compile(r"^(\s*)(- \[ \] |- |\d+\. )(.*)$")


def fetch_issues() -> dict[int, dict]:
    """Every issue in the repository, keyed by number, via ``gh``."""
    out = subprocess.run(
        [
            "gh",
            "issue",
            "list",
            "--repo",
            REPO,
            "--state",
            "all",
            "--limit",
            "200",
            "--json",
            "number,title,body,state,labels",
        ],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    return {issue["number"]: issue for issue in json.loads(out)}


def inline(text: str) -> str:
    """Inline Markdown: code, bold, ``#N`` links, and ``L12`` limitation cites."""
    text = html.escape(text, quote=False)
    text = re.sub(r"`([^`]+)`", r"<code>\1</code>", text)
    text = re.sub(r"\*\*([^*]+)\*\*", r"<strong>\1</strong>", text)
    text = re.sub(r"(?<![\w/])#(\d{1,3})\b", r'<a class="ref" href="#issue-\1">#\1</a>', text)
    text = re.sub(r"\bL(\d{1,2})\b", r'<span class="lim">L\1</span>', text)
    return text


def _cells(row: str) -> list[str]:
    return [cell.strip() for cell in row.strip().strip("|").split("|")]


def render_body(lines: list[str]) -> str:
    """Block Markdown of the subset the issue bodies use."""
    out: list[str] = []
    paragraph: list[str] = []
    i, n = 0, len(lines)

    def flush() -> None:
        if paragraph:
            out.append("<p>" + inline(" ".join(paragraph)) + "</p>")
            paragraph.clear()

    while i < n:
        line = lines[i]
        if line.startswith("```"):
            flush()
            block: list[str] = []
            i += 1
            while i < n and not lines[i].startswith("```"):
                block.append(lines[i])
                i += 1
            out.append("<pre><code>" + html.escape("\n".join(block)) + "</code></pre>")
            i += 1
            continue
        if line.startswith("## "):
            flush()
            out.append(f"<h3>{inline(line[3:])}</h3>")
            i += 1
            continue
        if line.strip() == "---":
            flush()
            out.append("<hr>")
            i += 1
            continue
        if line.startswith("|") and i + 1 < n and lines[i + 1].startswith("|--"):
            flush()
            rows: list[str] = []
            while i < n and lines[i].startswith("|"):
                rows.append(lines[i])
                i += 1
            head = "".join(f"<th>{inline(c)}</th>" for c in _cells(rows[0]))
            body = "".join(
                "<tr>" + "".join(f"<td>{inline(c)}</td>" for c in _cells(r)) + "</tr>"
                for r in rows[2:]
            )
            out.append(
                f'<div class="tablewrap small"><table><thead><tr>{head}</tr></thead>'
                f"<tbody>{body}</tbody></table></div>"
            )
            continue
        match = LIST_ITEM.match(line)
        if match:
            flush()
            kind = _list_kind(match.group(2))
            items: list[str] = []
            while i < n:
                item = LIST_ITEM.match(lines[i])
                if item:
                    if _list_kind(item.group(2)) != kind:
                        break
                    items.append(item.group(3))
                    i += 1
                elif (
                    lines[i].strip()
                    and not lines[i].startswith(("```", "## ", "|"))
                    and lines[i].strip() != "---"
                ):
                    items[-1] += " " + lines[i].strip()
                    i += 1
                else:
                    break
            tag = "ol" if kind == "ol" else "ul"
            cls = ' class="checks"' if kind == "check" else ""
            rendered = "".join(f"<li>{inline(t)}</li>" for t in items)
            out.append(f"<{tag}{cls}>{rendered}</{tag}>")
            continue
        if not line.strip():
            flush()
            i += 1
            continue
        paragraph.append(line.strip())
        i += 1
    flush()
    return "\n".join(out)


def _list_kind(marker: str) -> str:
    if marker.startswith("- ["):
        return "check"
    return "ol" if marker[0].isdigit() else "ul"


def parse(issue: dict) -> tuple[str, list[str], str]:
    """``(owner, dependency numbers, rendered body)`` from an issue body."""
    lines = issue["body"].replace("\r\n", "\n").splitlines()
    head = lines[0]
    owner_match = re.search(r"\*\*Owner:\*\*\s*(\w+)", head)
    owner = owner_match.group(1) if owner_match else "agent"
    deps_match = re.search(r"\*\*Depends on:\*\*\s*([^·\n]+)", head)
    deps = re.findall(r"\d+", deps_match.group(1)) if deps_match else []
    return owner, deps, render_body(lines[1:])


def dep_chips(deps: list[str]) -> str:
    if not deps:
        return '<span class="dep none">no dependencies</span>'
    return "".join(f'<a class="dep" href="#issue-{d}">#{d}</a>' for d in deps)


STYLE = """
:root {
  --paper:#f5f6f3; --paper-2:#eceee9; --ink:#1c2530; --ink-2:#4a5561; --ink-3:#77818c;
  --rule:#d5d9d3; --accent:#2b5a7a; --accent-soft:#dfe8ee;
  --human:#8a5a1e; --human-soft:#f1e6d3; --agent:#2f6b4f; --agent-soft:#dcebe1;
  --code-bg:#e9ece7; --lim:#5b4a86;
  --serif:"Newsreader", Georgia, "Times New Roman", serif;
  --sans:"Public Sans", "Segoe UI", Helvetica, Arial, sans-serif;
  --mono:"JetBrains Mono", ui-monospace, SFMono-Regular, Menlo, monospace;
}
@media (prefers-color-scheme: dark) {
  :root:not([data-theme="light"]) {
    --paper:#151a1f; --paper-2:#1c2228; --ink:#e4e8ea; --ink-2:#aab3bb; --ink-3:#7f8992;
    --rule:#2c343c; --accent:#7fb0d1; --accent-soft:#1f3140;
    --human:#d9a55b; --human-soft:#3a2d18; --agent:#7cc3a0; --agent-soft:#1c3328;
    --code-bg:#1f262d; --lim:#b3a4dd;
  }
}
:root[data-theme="dark"] {
  --paper:#151a1f; --paper-2:#1c2228; --ink:#e4e8ea; --ink-2:#aab3bb; --ink-3:#7f8992;
  --rule:#2c343c; --accent:#7fb0d1; --accent-soft:#1f3140;
  --human:#d9a55b; --human-soft:#3a2d18; --agent:#7cc3a0; --agent-soft:#1c3328;
  --code-bg:#1f262d; --lim:#b3a4dd;
}
* { box-sizing:border-box; }
body { background:var(--paper); color:var(--ink); font-family:var(--sans); font-size:15.5px; line-height:1.55; margin:0; }
a { color:var(--accent); text-decoration:none; }
a:hover, a:focus-visible { text-decoration:underline; }
a:focus-visible { outline:2px solid var(--accent); outline-offset:2px; border-radius:2px; }
code { font-family:var(--mono); font-size:0.86em; background:var(--code-bg); padding:0.08em 0.35em; border-radius:3px; }
pre { background:var(--code-bg); border:1px solid var(--rule); border-radius:4px; padding:0.9rem 1rem; overflow-x:auto; font-size:0.82rem; line-height:1.5; }
pre code { background:none; padding:0; font-size:inherit; }
h1, h2, h3 { font-family:var(--serif); font-weight:600; text-wrap:balance; margin:0; letter-spacing:-0.01em; }
h1 { font-size:2.4rem; line-height:1.1; }
h2 { font-size:1.55rem; line-height:1.2; }
h3 { font-size:1.05rem; font-family:var(--sans); font-weight:600; text-transform:uppercase; letter-spacing:0.06em; color:var(--ink-2); margin-top:1.6rem; }
hr { border:0; border-top:1px solid var(--rule); margin:1.4rem 0; }
.wrap { display:grid; grid-template-columns:280px minmax(0,1fr); gap:3rem; max-width:1260px; margin:0 auto; padding:2.5rem 1.5rem 5rem; }
nav { position:sticky; top:1.5rem; align-self:start; max-height:calc(100vh - 3rem); overflow-y:auto; font-size:0.8rem; }
nav .navtitle { font-weight:600; text-transform:uppercase; letter-spacing:0.08em; font-size:0.7rem; color:var(--ink-3); margin-bottom:0.6rem; }
nav ul { list-style:none; margin:0; padding:0; display:flex; flex-direction:column; gap:0.15rem; }
nav a { display:grid; grid-template-columns:10px 30px 1fr; gap:0.5rem; align-items:baseline; color:var(--ink-2); padding:0.2rem 0.3rem; border-radius:3px; }
nav a:hover { background:var(--paper-2); text-decoration:none; color:var(--ink); }
nav li.done a { color:var(--ink-3); }
nav .num { font-family:var(--mono); color:var(--ink-3); font-size:0.72rem; }
.dot { width:8px; height:8px; border-radius:50%; display:inline-block; position:relative; top:-1px; }
.dot.human { background:var(--human); } .dot.agent { background:var(--agent); }
main { min-width:0; }
.brief { max-width:68ch; }
.brief p { margin:0.9rem 0; color:var(--ink-2); }
.eyebrow { font-size:0.72rem; text-transform:uppercase; letter-spacing:0.1em; color:var(--ink-3); font-weight:600; margin-bottom:0.5rem; }
.legend { display:flex; gap:1.2rem; font-size:0.82rem; color:var(--ink-2); margin-top:1rem; flex-wrap:wrap; align-items:center; }
.legend > span { display:inline-flex; align-items:center; gap:0.4rem; }
.tablewrap { overflow-x:auto; margin:2rem 0 3rem; border-top:2px solid var(--ink); }
.tablewrap.small { margin:1rem 0; border-top:1px solid var(--rule); }
table { border-collapse:collapse; width:100%; font-size:0.86rem; }
.tablewrap.small table { font-size:0.8rem; }
th { text-align:left; font-weight:600; text-transform:uppercase; letter-spacing:0.06em; font-size:0.68rem; color:var(--ink-3); padding:0.6rem 0.6rem 0.4rem; border-bottom:1px solid var(--rule); }
td { padding:0.5rem 0.6rem; border-bottom:1px solid var(--rule); vertical-align:top; }
td.num { font-family:var(--mono); color:var(--ink-3); font-variant-numeric:tabular-nums; width:3rem; white-space:nowrap; }
td.deps { white-space:nowrap; }
tr.done td, tr.done td a { color:var(--ink-3); }
.owner, .state { display:inline-block; font-size:0.7rem; font-weight:600; text-transform:uppercase; letter-spacing:0.08em; padding:0.15rem 0.5rem; border-radius:2px; }
.owner.human { color:var(--human); background:var(--human-soft); }
.owner.agent { color:var(--agent); background:var(--agent-soft); }
.state.done { color:var(--ink-3); background:var(--paper-2); }
.state.open { color:var(--accent); background:var(--accent-soft); }
.dep { display:inline-block; font-family:var(--mono); font-size:0.75rem; padding:0.05rem 0.4rem; border:1px solid var(--rule); border-radius:3px; margin-right:0.3rem; color:var(--accent); }
.dep.none { font-family:var(--sans); color:var(--ink-3); border-style:dashed; }
.issue { border-top:1px solid var(--rule); padding:2.2rem 0 1.5rem; max-width:76ch; }
.issue.done { opacity:0.72; }
.issue header { margin-bottom:0.8rem; }
.kicker { font-family:var(--mono); font-size:0.75rem; color:var(--ink-3); display:block; margin-bottom:0.3rem; }
.meta { display:flex; flex-wrap:wrap; align-items:center; gap:0.6rem 1rem; margin-top:0.7rem; font-size:0.8rem; color:var(--ink-2); }
.meta .label { text-transform:uppercase; letter-spacing:0.08em; font-size:0.66rem; color:var(--ink-3); margin-right:0.4rem; font-weight:600; }
.gh { font-size:0.78rem; margin-left:auto; }
.issue p { margin:0.7rem 0; }
.issue ul, .issue ol { padding-left:1.3rem; margin:0.5rem 0; display:flex; flex-direction:column; gap:0.35rem; }
ul.checks { list-style:none; padding-left:0; }
ul.checks li { padding-left:1.6rem; position:relative; }
ul.checks li::before { content:""; position:absolute; left:0; top:0.35em; width:0.8em; height:0.8em; border:1.5px solid var(--accent); border-radius:2px; }
.ref { font-family:var(--mono); font-size:0.9em; }
.lim { font-family:var(--mono); font-size:0.85em; color:var(--lim); }
@media (max-width: 960px) {
  .wrap { grid-template-columns:1fr; gap:1.5rem; }
  nav { position:static; max-height:none; }
  nav ul { display:grid; grid-template-columns:repeat(auto-fill, minmax(240px,1fr)); }
  .gh { margin-left:0; }
}
@media (prefers-reduced-motion: no-preference) { html { scroll-behavior:smooth; } }
"""

FONTS = (
    '<link rel="preconnect" href="https://fonts.googleapis.com">\n'
    '<link rel="stylesheet" href="https://fonts.googleapis.com/css2?family=Newsreader:opsz,wght@6..72,500;6..72,600'
    '&family=Public+Sans:wght@400;500;600&family=JetBrains+Mono:wght@400;500&display=swap">'
)


def render_page(issues: dict[int, dict], order: list[int], generated_on: str) -> str:
    missing = sorted(set(issues) - set(order))
    unknown = sorted(set(order) - set(issues))
    if missing or unknown:
        raise SystemExit(
            f"EXECUTION_ORDER is out of step with the tracker: not listed {missing}, "
            f"listed but not on GitHub {unknown}."
        )

    parsed = {n: parse(issues[n]) for n in order}

    def state(n: int) -> str:
        return "done" if issues[n]["state"] == "CLOSED" else "open"

    def title(n: int) -> str:
        return html.escape(issues[n]["title"])

    done_n = sum(1 for n in order if state(n) == "done")
    human_n = sum(1 for n in order if parsed[n][0] == "human")
    open_order = ", ".join(str(n) for n in order if state(n) == "open")

    nav = "".join(
        f'<li class="{state(n)}"><a href="#issue-{n}"><span class="dot {parsed[n][0]}"></span>'
        f'<span class="num">#{n}</span><span>{title(n)}</span></a></li>'
        for n in order
    )
    rows = "".join(
        f'<tr class="{state(n)}"><td class="num">{k}</td><td class="num">#{n}</td>'
        f'<td><a href="#issue-{n}">{title(n)}</a></td>'
        f'<td><span class="owner {parsed[n][0]}">{parsed[n][0]}</span></td>'
        f'<td><span class="state {state(n)}">{state(n)}</span></td>'
        f'<td class="deps">{dep_chips(parsed[n][1])}</td></tr>'
        for k, n in enumerate(order, 1)
    )
    sections = "".join(
        f'<section class="issue {state(n)}" id="issue-{n}"><header>'
        f'<span class="kicker">Step {k:02d} · issue #{n} · {state(n)}</span>'
        f"<h2>{title(n)}</h2>"
        f'<div class="meta"><span class="owner {parsed[n][0]}">{parsed[n][0]}</span>'
        f'<span class="depends"><span class="label">depends on</span>{dep_chips(parsed[n][1])}</span>'
        f'<a class="gh" href="https://github.com/{REPO}/issues/{n}">on GitHub</a></div></header>'
        f"{parsed[n][2]}</section>"
        for k, n in enumerate(order, 1)
    )

    return f"""<title>Northplan Roadmap</title>
{FONTS}
<style>{STYLE}</style>
<div class="wrap">
<nav aria-label="Steps"><div class="navtitle">Execution order</div><ul>{nav}</ul></nav>
<main>
  <div class="brief">
    <div class="eyebrow">northplan · roadmap · generated {generated_on} from the live tracker</div>
    <h1>Northplan Roadmap</h1>
    <p>{len(order)} issues from the scaffold to a first end-to-end version, in execution order rather than by number. {done_n} are done and shown dimmed, in the order they were completed. {human_n} issues are the human's: parameter sourcing and verification. The rest go to a coding agent with success criteria the verifier can check.</p>
    <p>Open issues, in order: {open_order}. Every simplification the issues rely on is cited as an <span class="lim">L-number</span> into <code>docs/limitations.md</code>. Bodies are the live GitHub text.</p>
    <div class="legend"><span><i class="dot human"></i>human</span><span><i class="dot agent"></i>agent</span><span class="state done">done</span><span class="state open">open</span><span><a class="dep" href="#issue-1">#1</a>dependency, links to the issue</span></div>
  </div>
  <div class="tablewrap"><table>
    <thead><tr><th>Order</th><th>#</th><th>Step</th><th>Owner</th><th>Status</th><th>Depends on</th></tr></thead>
    <tbody>{rows}</tbody>
  </table></div>
  {sections}
</main>
</div>
"""


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--out", type=Path, default=Path(__file__).with_name("roadmap.html"))
    parser.add_argument(
        "--standalone",
        action="store_true",
        help="wrap the fragment in a full HTML document for opening in a browser",
    )
    args = parser.parse_args()

    import datetime as dt

    page = render_page(fetch_issues(), EXECUTION_ORDER, dt.date.today().isoformat())
    if args.standalone:
        page = (
            '<!doctype html>\n<html lang="en"><head><meta charset="utf-8">'
            '<meta name="viewport" content="width=device-width, initial-scale=1">\n'
            + page.replace("</style>", "</style></head><body>", 1)
            + "</body></html>\n"
        )
    args.out.write_text(page, encoding="utf-8")
    print(f"wrote {args.out} ({len(page):,} bytes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
