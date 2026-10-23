"""Pick a project up where an experiment ran into trouble.

Given a lab result logged against a roadmap step, this gathers only the context that matters
(the step, earlier attempts at it, quote-checked evidence from papers reviewed in full, and
papers newly found for this specific problem through Valency Bond), then proposes likely causes,
quick checks, and a revised plan from that step onward. Citations and quotes are validated.
"""
import json
import re

from research import _ask, _contains, _model_client

STOP = set("the and for with that this from were was are has have had not but its into than then them they our "
           "using used use which when where what how can could would should also only very more most some such "
           "after before about over under between did does done get got one two all any each".split())

QUERY_PROMPT = """Write literature search queries for a problem a scientist hit in the lab.
Return a JSON object {"queries": [one or two short keyword queries]}. Combine the technique or material with the
failure itself, for example "dynamical scattering thick samples electron diffraction neural network"."""

DIAGNOSE_PROMPT = """You help a scientist who hit a problem while carrying out one step of a research plan.
You receive: the research question; the plan step; earlier attempts at this point in the plan, with what was suggested then;
the new lab result; evidence items from papers they reviewed in full (paper id, section, quote); and abstracts of
papers found for this problem.
Return a JSON object with keys:
- summary: 2-3 sentences on what most likely went wrong and what to do next
- likely_causes: list of {cause, explanation, evidence_ids, evidence_quotes, confidence ("high"|"medium"|"low"),
  status ("literature-backed"|"hypothesis")}, most likely first
- checks: list of {check, tells_apart, expected_result}: quick diagnostic experiments or analyses, cheapest first
- revised_steps: list of {title, details, evidence_ids}: the plan from this step onward, starting with the most
  useful next action, at most 6 steps
Cite only supplied paper ids. evidence_quotes must be exact text from the supplied quotes or abstracts. A cause is
"literature-backed" only if a supplied paper supports it. Do not repeat suggestions that earlier attempts already
tried or ruled out. Flag anything that needs a lab safety review instead of giving hazardous procedures."""


def _tokens(text):
    return {w for w in re.findall(r"[a-z0-9]{3,}", str(text).lower()) if w not in STOP}


def relevant_evidence(problem_text, step, reviews, limit=18):
    """Quote-checked review items that share the most words with the problem, limitations first."""
    want = _tokens(f'{problem_text} {step.get("title", "")} {step.get("details", "")}')
    scored = []
    for pid, rv in (reviews or {}).items():
        for kind in ("limitations", "results", "methods", "future_work"):
            for x in rv.get(kind, []):
                if not x.get("verified"):
                    continue
                text = " ".join(str(x.get(k, "")) for k in ("item", "finding", "detail", "metric", "value", "condition", "quote"))
                overlap = len(want & _tokens(text))
                if not overlap:
                    continue
                score = overlap + (2 if kind == "limitations" else 0) + (1 if pid in step.get("evidence_ids", []) else 0)
                scored.append((score, {"paper_id": pid, "kind": kind, "section": x.get("section"),
                                       "item": x.get("item") or x.get("finding"), "quote": x.get("quote")}))
    scored.sort(key=lambda t: -t[0])
    return [item for _, item in scored[:limit]]


def _queries(client, model, question, step, problem):
    fallback = [re.sub(r"\s+", " ", f'{step.get("title", "")} {problem}')[:120]]
    try:
        out = _ask(client, model, [
            {"role": "system", "content": QUERY_PROMPT},
            {"role": "user", "content": json.dumps({"research_question": question, "step": step.get("title"),
                                                    "problem": problem[:1500]}) + _no_think(model)},
        ])
        qs = [str(q).strip() for q in out.get("queries", []) if str(q).strip()][:2]
        return qs or fallback
    except (ValueError, AttributeError):
        return fallback


def _no_think(model):
    return " /no_think" if "qwen3" in str(model).lower() else ""


def _problem_text(entry):
    return f'What was done: {entry.get("did", "")}\nWhat happened: {entry.get("observed", "")}'


def find_problem_papers(question, step, entry, papers, api_key="", model="qwen3:8b", provider="Local Ollama",
                        base_url="http://localhost:11434/v1", find_papers=None, progress=None):
    """Stage 1: search Valency Bond for papers about this specific problem.
    Returns (new papers not already in the project, queries used, notes about failed searches)."""
    say = progress or (lambda text: None)
    if not find_papers:
        return [], [], []
    client = _model_client(provider, api_key, base_url)
    say("Searching Valency Bond for papers about this problem")
    queries = _queries(client, model, question, step, _problem_text(entry))
    found, notes = [], []
    for q in queries:
        try:
            found += find_papers(q)
        except Exception as exc:  # a failed search shouldn't stop the diagnosis
            notes.append(f"Bond search for '{q}' failed: {type(exc).__name__}")
    known = {(p.get("doi") or "").lower() or p["title"].lower() for p in papers}
    fresh, seen = [], set()
    for p in found:
        k = (p.get("doi") or "").lower() or p["title"].lower()
        if k not in known and k not in seen:
            seen.add(k)
            fresh.append(p)
    return fresh[:10], queries, notes


def diagnose(question, step, step_number, entry, lab_log, papers, new_papers, reviews, api_key="", model="qwen3:8b",
             provider="Local Ollama", base_url="http://localhost:11434/v1", progress=None):
    """Stage 2: likely causes, quick checks and a revised plan, from the step, earlier attempts at this point in
    the plan, full-text evidence (including any new papers already reviewed) and new papers' abstracts."""
    say = progress or (lambda text: None)
    client = _model_client(provider, api_key, base_url)
    problem = _problem_text(entry)
    attachments = [{"file": a["name"], "excerpt": a["excerpt"][:1500]} for a in entry.get("attachments", []) if a.get("excerpt")]
    earlier = []
    for e in lab_log:
        same_point = e.get("step_title") == step.get("title") or e.get("step_index") == entry.get("step_index")
        if e["id"] == entry["id"] or not same_point:  # same step, or the step that replaced it after a revision
            continue
        d = e.get("diagnosis") or {}
        earlier.append({"when": e.get("created"), "did": e.get("did"), "observed": e.get("observed"),
                        "outcome": e.get("outcome"), "suggested_then": [c.get("cause") for c in d.get("likely_causes", [])][:4]})

    say("Checking the papers read in full for the same problem")
    evidence = relevant_evidence(problem, step, reviews)
    new_ids = {p["id"] for p in new_papers}
    say("Working out likely causes, quick checks and a revised plan")
    payload = {
        "research_question": question,
        "step": {"number": step_number, "title": step.get("title"), "details": step.get("details"),
                 "evidence_ids": step.get("evidence_ids", [])},
        "earlier_attempts_at_this_point_in_the_plan": earlier,
        "new_lab_result": {"outcome": entry.get("outcome"), "problem": problem, "attached_data": attachments},
        "evidence_from_full_text_reviews": evidence,
        "papers_found_for_this_problem": [
            {"id": p["id"], "title": p["title"], "year": p.get("year"),
             "read_in_full": p["id"] in (reviews or {}),
             "summary_or_abstract": ((reviews or {}).get(p["id"], {}).get("summary") or p.get("abstract") or "")[:1200]}
            for p in new_papers],
    }
    result = _ask(client, model, [
        {"role": "system", "content": DIAGNOSE_PROMPT},
        {"role": "user", "content": json.dumps(payload, ensure_ascii=False, default=str)[:24000] + _no_think(model)},
    ])
    if not isinstance(result, dict):
        result = {}

    # Validate citations and quotes against what the model was actually given.
    text_for = {p["id"]: p.get("abstract") or "" for p in papers}
    for pid, rv in (reviews or {}).items():
        text_for[pid] = text_for.get(pid, "") + " " + " ".join(
            str(x.get("quote", "")) for k in ("results", "methods", "limitations", "future_work") for x in rv.get(k, []) if x.get("verified"))
    for key in ("likely_causes", "checks", "revised_steps"):
        items = result.get(key)
        result[key] = [x for x in items if isinstance(x, dict)] if isinstance(items, list) else []
    for item in result["likely_causes"] + result["revised_steps"]:
        ids = item.get("evidence_ids") if isinstance(item.get("evidence_ids"), list) else []
        item["evidence_ids"] = [i for i in ids if i in text_for]
        quotes = item.get("evidence_quotes") if isinstance(item.get("evidence_quotes"), list) else []
        item["evidence_quotes"] = [q for q in quotes if isinstance(q, str) and any(_contains(q, text_for[i]) for i in item["evidence_ids"])]
    for cause in result["likely_causes"]:
        if not cause["evidence_ids"]:
            cause["status"] = "hypothesis"
    for s in result["revised_steps"]:
        s["status"] = "proposed"
    result["summary"] = str(result.get("summary", ""))
    result.update({"evidence_items": len(evidence), "earlier_attempts": len(earlier),
                   "new_paper_ids": sorted(new_ids),
                   "new_papers_read_in_full": sorted(i for i in new_ids if i in (reviews or {}))})
    return result


def troubleshoot(question, steps, step_index, entry, lab_log, papers, reviews, api_key="", model="qwen3:8b",
                 provider="Local Ollama", base_url="http://localhost:11434/v1", find_papers=None, progress=None,
                 step_number=None):
    """Both stages in one call, with new papers given temporary ids (N01...). Returns (diagnosis, new_papers)."""
    llm = dict(api_key=api_key, model=model, provider=provider, base_url=base_url)
    step = steps[step_index - 1]
    (progress or (lambda t: None))("Checking the papers you reviewed in full for the same problem")
    fresh, queries, notes = find_problem_papers(question, step, entry, papers, find_papers=find_papers,
                                                progress=progress, **llm)
    fresh = [{**p, "id": f"N{i:02d}"} for i, p in enumerate(fresh, 1)]
    result = diagnose(question, step, step_number or step_index, entry, lab_log, papers + fresh, fresh, reviews,
                      progress=progress, **llm)
    result.update({"queries": queries, "notes": notes})
    return result, fresh


def renumber(diagnosis, id_map):
    """After merging new papers into the project, swap temporary ids (N01...) for project ids (P025...)."""
    for key in ("likely_causes", "revised_steps"):
        for item in diagnosis.get(key, []):
            item["evidence_ids"] = [id_map.get(i, i) for i in item.get("evidence_ids", [])]
    diagnosis["new_paper_ids"] = list(id_map.values())
    return diagnosis
