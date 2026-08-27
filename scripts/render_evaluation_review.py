"""Render an internal evaluation suite and advisory judge report as review HTML."""

from __future__ import annotations

import argparse
import html
import json
from pathlib import Path
from typing import Any


def _escape(value: Any) -> str:
    return html.escape(str(value))


def _list(items: list[str], empty: str = "None") -> str:
    if not items:
        return f'<p class="muted">{_escape(empty)}</p>'
    return "<ul>" + "".join(f"<li>{_escape(item)}</li>" for item in items) + "</ul>"


def _case_card(case: dict[str, Any], result: dict[str, Any]) -> str:
    expected = case["expected"]
    decision = expected["decision"]
    verdict = result["verdict"]
    criteria = expected.get("answer_criteria", [])
    evidence = case.get("gold_evidence", [])
    findings = result.get("findings", [])
    scores = result["scores"]
    papers = case["papers"]

    criteria_html = "".join(
        "<li><code>{}</code> — {} <span class=\"pill\">{}</span></li>".format(
            _escape(item["criterion_id"]),
            _escape(item["description"]),
            "required" if item["required"] else "optional",
        )
        for item in criteria
    ) or '<p class="muted">No answer criteria: expected decision is abstention.</p>'

    evidence_html = "".join(
        """
        <div class="evidence">
          <div><strong>{evidence_id}</strong> · {paper} · {location}</div>
          <blockquote>{quote}</blockquote>
          <div class="muted">Supports: {supports}</div>
        </div>
        """.format(
            evidence_id=_escape(item["evidence_id"]),
            paper=_escape(item["versioned_id"]),
            location=_escape(
                f"page {item['page']} · {item.get('section') or 'section unknown'}"
                if item["source_type"] == "full_text"
                else item.get("section") or "Abstract"
            ),
            quote=_escape(item["quote"]),
            supports=_escape(", ".join(item["supports"])),
        )
        for item in evidence
    ) or '<p class="muted">No gold passage. A human must check the full paper for absence.</p>'

    findings_html = "".join(
        '<li class="{}"><strong>{}</strong> · {}: {}</li>'.format(
            _escape(item["severity"]),
            _escape(item["severity"].upper()),
            _escape(item["category"]),
            _escape(item["message"]),
        )
        for item in findings
    ) or '<p class="pass-text">No judge findings.</p>'

    challenge = case.get("challenge")
    challenge_html = (
        "<p><strong>Kinds:</strong> {}<br><strong>Description:</strong> {}</p>".format(
            _escape(", ".join(challenge["kinds"])), _escape(challenge["description"])
        )
        if challenge
        else '<p class="muted">No challenge label.</p>'
    )
    answer = expected.get("reference_answer")
    abstention = expected.get("abstention_reason")

    return f"""
    <article class="case {verdict}" id="{_escape(case['case_id'])}">
      <header>
        <div>
          <div class="eyebrow">{_escape(case['case_id'])}</div>
          <h2>{_escape(case['question'])}</h2>
        </div>
        <div class="badges">
          <span class="badge decision">{_escape(decision)}</span>
          <span class="badge verdict">{_escape(verdict)}</span>
        </div>
      </header>
      <div class="meta">
        <span>Type: {_escape(case['question_type'])}</span>
        <span>Papers: {_escape(', '.join(p['versioned_id'] for p in papers))}</span>
        <span>Human review: {_escape(result['human_review_required'])}</span>
      </div>
      <section>
        <h3>Expected response</h3>
        <p><strong>{'Reference answer' if answer else 'Expected abstention'}:</strong>
        {_escape(answer or abstention or 'No reason recorded')}</p>
        <h4>Answer criteria</h4>
        <ul>{criteria_html}</ul>
        <h4>Forbidden claims</h4>
        {_list(expected.get('forbidden_claims', []))}
      </section>
      <section>
        <h3>Gold evidence</h3>
        {evidence_html}
      </section>
      <section>
        <h3>Challenge design</h3>
        {challenge_html}
      </section>
      <section class="judge">
        <h3>LLM judge review</h3>
        <div class="scores">
          {''.join(f'<div><b>{_escape(name.replace("_", " "))}</b><span>{score}/5</span></div>' for name, score in scores.items())}
        </div>
        <p><strong>Rationale:</strong> {_escape(result['rationale'])}</p>
        <h4>Findings</h4>
        <ul>{findings_html}</ul>
      </section>
    </article>
    """


def render(suite: dict[str, Any], report: dict[str, Any]) -> str:
    result_by_id = {item["case_id"]: item for item in report["results"]}
    cases = sorted(
        suite["cases"],
        key=lambda item: (result_by_id[item["case_id"]]["verdict"] == "pass", item["case_id"]),
    )
    cards = "".join(_case_card(case, result_by_id[case["case_id"]]) for case in cases)
    counts = report["verdict_counts"]
    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width">
<title>Evaluation review · {_escape(suite['suite_id'])}</title>
<style>
:root {{ color-scheme: light; --ink:#182024; --muted:#66737a; --line:#dce3e6; --paper:#fff;
  --bg:#f3f6f5; --green:#157347; --amber:#a65d00; --red:#b3261e; --blue:#3157b7; }}
* {{ box-sizing:border-box }} body {{ margin:0; font:15px/1.55 Inter,Segoe UI,sans-serif;
  color:var(--ink); background:var(--bg) }} main {{ width:min(1120px,94vw); margin:36px auto 80px }}
.hero {{ background:#17211e; color:white; border-radius:20px; padding:30px; margin-bottom:22px }}
.hero h1 {{ margin:4px 0 8px; font-size:32px }} .hero p {{ margin:0; color:#cdd8d4 }}
.summary {{ display:flex; gap:10px; flex-wrap:wrap; margin-top:18px }}
.summary span,.badge,.pill {{ border-radius:999px; padding:5px 10px; font-weight:700 }}
.summary span {{ background:#293733 }} .case {{ background:var(--paper); border:1px solid var(--line);
  border-left:7px solid var(--green); border-radius:16px; margin:18px 0; overflow:hidden; box-shadow:0 3px 14px #20302a0b }}
.case.needs_revision {{ border-left-color:var(--amber) }} .case.fail {{ border-left-color:var(--red) }}
.case>header {{ display:flex; justify-content:space-between; gap:18px; padding:24px 26px 16px }}
h2 {{ margin:3px 0; font-size:23px; line-height:1.3 }} h3 {{ margin:0 0 11px; font-size:17px }} h4 {{ margin:16px 0 5px }}
.eyebrow {{ color:var(--muted); font:700 12px/1.2 ui-monospace,monospace; text-transform:uppercase }}
.badges {{ display:flex; align-items:flex-start; gap:7px }} .badge {{ font-size:12px; text-transform:uppercase }}
.decision {{ background:#e7eefc; color:var(--blue) }} .verdict {{ background:#e4f4eb; color:var(--green) }}
.needs_revision .verdict {{ background:#fff0d9; color:var(--amber) }} .fail .verdict {{ background:#fde8e7; color:var(--red) }}
.meta {{ display:flex; flex-wrap:wrap; gap:8px 20px; padding:0 26px 18px; color:var(--muted) }}
section {{ padding:20px 26px; border-top:1px solid var(--line) }} .judge {{ background:#fafbf8 }}
ul {{ margin:7px 0; padding-left:22px }} blockquote {{ margin:10px 0 5px; padding:11px 14px;
  border-left:3px solid #9aa9a3; background:#f5f7f6 }} .evidence {{ margin:12px 0 }}
.muted {{ color:var(--muted) }} .pill {{ font-size:11px; padding:2px 7px; background:#edf1f2 }}
.scores {{ display:grid; grid-template-columns:repeat(auto-fit,minmax(165px,1fr)); gap:9px }}
.scores div {{ display:flex; justify-content:space-between; gap:10px; padding:9px 11px; background:white; border:1px solid var(--line); border-radius:9px }}
.scores b {{ text-transform:capitalize }} .scores span {{ font-weight:800 }} .warning {{ color:var(--amber) }}
.error {{ color:var(--red) }} .pass-text {{ color:var(--green); font-weight:700 }} code {{ font-size:13px }}
@media(max-width:700px) {{ .case>header {{ display:block }} .badges {{ margin-top:12px }} }}
</style></head><body><main>
<div class="hero"><div class="eyebrow">Internal development suite · advisory review</div>
<h1>{_escape(suite['suite_id'])}</h1><p>Question, expected behavior, gold evidence, and LLM findings in one place. Needs-revision cases appear first.</p>
<div class="summary"><span>{_escape(report['case_count'])} cases</span>
<span>{_escape(counts.get('pass',0))} pass</span><span>{_escape(counts.get('needs_revision',0))} needs revision</span>
<span>Judge: {_escape(report['judge_model'])}</span></div></div>{cards}</main></body></html>"""


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    suite = json.loads(args.suite.read_text(encoding="utf-8"))
    report = json.loads(args.report.read_text(encoding="utf-8"))
    suite_ids = {case["case_id"] for case in suite["cases"]}
    report_ids = {result["case_id"] for result in report["results"]}
    if suite_ids != report_ids:
        raise ValueError("Suite and judge report case IDs do not match.")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(render(suite, report), encoding="utf-8")
    print(args.output.resolve())


if __name__ == "__main__":
    main()
