"""Section-by-section full-paper review. Every extracted item carries a quote that is checked
against the paper's own text, so unsupported claims are visible instead of hidden."""
import json
import re
import unicodedata

from quotes import normalize, quote_found, quote_match  # noqa: F401  (quote_found re-exported)
from research import _ask, _model_client

REVIEW_KINDS = {"abstract", "methods", "results", "discussion", "limitations", "conclusion", "availability", "body"}
PRIORITY = ["methods", "results", "limitations", "discussion", "conclusion", "availability", "body", "abstract"]
LISTS = ("methods", "results", "limitations", "future_work", "resources")

EXTRACT_PROMPT = """You review ONE passage from a scientific paper. Extract only what this passage states; never use outside knowledge.
Return a JSON object with these keys, each a list (empty when the passage says nothing relevant):
- methods: {item, detail, quote} - techniques, instruments and settings, samples or materials, datasets, models, training or analysis procedures
- results: {finding, metric, value, condition, quote} - quantitative or qualitative outcomes; copy numbers exactly
- limitations: {item, quote} - weaknesses or failure cases the authors state
- future_work: {item, quote} - open problems or next steps the authors state
- resources: {name, kind, url, quote} - kind is one of dataset, software, code, instrument, material, model
Every quote must be copied word for word from the passage, at most 30 words. Keep item texts short."""

SYNTH_PROMPT = """You write a structured review of one scientific paper from verified extractions of its full text.
Use only the supplied extractions and abstract. Return a JSON object with keys:
summary (3-4 sentences), contribution (1 sentence), relevance (1-2 sentences on how it bears on the research question),
strengths (list of short strings), weaknesses (list of short strings), open_questions (list of short strings),
reproducibility: {code: "available" | "not mentioned" | "unclear", data: "available" | "not mentioned" | "unclear", notes}.
Strengths and weaknesses are your assessment; ground each in the extractions. Do not invent numbers."""


def _norm(text):
    return normalize(text)


def _sentences(text):
    return re.split(r"(?<=[.!?])\s+(?=[A-Z0-9(\[])", text)


def make_passages(sections, size):
    passages = []
    for order, sec in enumerate(sections):
        if sec.get("kind") not in REVIEW_KINDS:
            continue
        buf = ""
        for sentence in _sentences(sec["text"]):
            if buf and len(buf) + len(sentence) + 1 > size:
                passages.append({"section": sec["title"], "kind": sec["kind"], "text": buf, "order": order})
                buf = sentence
            else:
                buf = f"{buf} {sentence}".strip()
            while len(buf) > size * 1.5:  # very long sentence runs, e.g. tables
                passages.append({"section": sec["title"], "kind": sec["kind"], "text": buf[:size], "order": order})
                buf = buf[size:]
        if buf:
            passages.append({"section": sec["title"], "kind": sec["kind"], "text": buf, "order": order})
    return passages


def _select(passages, cap):
    if len(passages) <= cap:
        return passages
    ranked = sorted(range(len(passages)), key=lambda i: (PRIORITY.index(passages[i]["kind"]), i))
    keep = sorted(ranked[:cap])
    return [passages[i] for i in keep]


def _no_think(model):
    # Qwen3 models "think" at length by default; this soft switch makes local reviews several times faster.
    return " /no_think" if "qwen3" in str(model).lower() else ""


def review_paper(paper, fulltext, question, api_key="", model="qwen3:8b", provider="Local Ollama",
                 base_url="http://localhost:11434/v1", progress=None):
    """Review a paper's full text. `progress(done, total, label)` is called as passages are read."""
    client = _model_client(provider, api_key, base_url)
    local = provider == "Local Ollama"
    size, cap = (5000, 12) if local else (16000, 10)  # small passages fit local models' default context
    all_passages = make_passages(fulltext["sections"], size)
    passages = _select(all_passages, cap)
    if not passages:
        raise ValueError("The full text had no methods, results or discussion sections to review.")

    found = {key: [] for key in LISTS}
    seen = set()
    for i, passage in enumerate(passages, 1):
        if progress:
            progress(i - 1, len(passages) + 1, f"Reading {passage['section']} ({i} of {len(passages)})")
        payload = {"research_question": question, "paper_title": paper["title"],
                   "section": passage["section"], "passage": passage["text"]}
        try:
            result = _ask(client, model, [
                {"role": "system", "content": EXTRACT_PROMPT},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False) + _no_think(model)},
            ])
        except ValueError:
            continue  # one unparseable passage shouldn't sink the review
        for key in LISTS:
            items = result.get(key, []) if isinstance(result, dict) else []
            for item in items if isinstance(items, list) else []:
                if not isinstance(item, dict):
                    continue
                label = _norm(item.get("item") or item.get("finding") or item.get("name") or "")
                if not label or (key, label) in seen:
                    continue
                seen.add((key, label))
                item["section"] = passage["section"]
                item["match"] = quote_match(item.get("quote"), passage["text"])
                item["verified"] = item["match"] is not None
                found[key].append(item)

    if progress:
        progress(len(passages), len(passages) + 1, "Writing the review")
    verified = {k: [x for x in v if x["verified"]] for k, v in found.items()}
    digest = {k: [{kk: vv for kk, vv in x.items() if kk not in ("verified", "quote", "match")} for x in v][:15]
              for k, v in verified.items()}
    synthesis = {}
    try:
        synthesis = _ask(client, model, [
            {"role": "system", "content": SYNTH_PROMPT},
            {"role": "user", "content": json.dumps({
                "research_question": question, "title": paper["title"], "abstract": paper["abstract"][:3000],
                "verified_extractions": digest}, ensure_ascii=False)[:14000] + _no_think(model)},
        ])
    except ValueError:
        synthesis = {"summary": "The model couldn't write a summary; the extracted items below are still usable."}
    if not isinstance(synthesis, dict):
        synthesis = {}

    links = list(fulltext.get("links") or [])
    for res in found["resources"]:
        url = str(res.get("url") or "")
        if url and url not in links and url.startswith("http") and quote_found(url, " ".join(s["text"] for s in fulltext["sections"])):
            links.append(url)
    return {
        "paper_id": paper["id"], "title": paper["title"], "source": fulltext["source"], "url": fulltext.get("url"),
        "license": fulltext.get("license"), "sections": [s["title"] for s in fulltext["sections"]],
        "passages_reviewed": len(passages), "passages_total": len(all_passages), "model": model,
        **{k: found[k] for k in LISTS}, "links": links,
        "summary": str(synthesis.get("summary", "")), "contribution": str(synthesis.get("contribution", "")),
        "relevance": str(synthesis.get("relevance", "")),
        "strengths": [str(x) for x in synthesis.get("strengths", []) if x] if isinstance(synthesis.get("strengths"), list) else [],
        "weaknesses": [str(x) for x in synthesis.get("weaknesses", []) if x] if isinstance(synthesis.get("weaknesses"), list) else [],
        "open_questions": [str(x) for x in synthesis.get("open_questions", []) if x] if isinstance(synthesis.get("open_questions"), list) else [],
        "reproducibility": synthesis.get("reproducibility") if isinstance(synthesis.get("reproducibility"), dict) else {},
    }


def review_markdown(review):
    """Plain Markdown for the exported report."""
    lines = [f"### [{review['paper_id']}] {review['title']}",
             f"Full text from {review['source']}; {review['passages_reviewed']} of {review['passages_total']} passages reviewed.",
             "", review.get("summary", "")]
    for label, key, field in (("Key results", "results", "finding"), ("Methods", "methods", "item"),
                              ("Stated limitations", "limitations", "item"), ("Stated future work", "future_work", "item")):
        items = [x for x in review.get(key, []) if x.get("verified")]
        if items:
            lines += ["", f"**{label}**"]
            for x in items[:8]:
                extra = " ".join(str(x.get(k)) for k in ("metric", "value", "condition") if x.get(k)) if key == "results" else str(x.get("detail") or "")
                lines.append(f"- {x.get(field, '')}{': ' + extra if extra.strip() else ''} ({x.get('section')}): \"{x.get('quote', '')}\"")
    if review.get("links"):
        lines += ["", "**Code and data links found in the paper**"] + [f"- {u}" for u in review["links"]]
    if review.get("weaknesses"):
        lines += ["", "**Reviewer-model assessment of weaknesses**"] + [f"- {w}" for w in review["weaknesses"]]
    return "\n".join(lines)
