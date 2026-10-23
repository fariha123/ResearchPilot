"""ResearchPilot: evidence-grounded research planning on top of Valency Bond."""
import html
import json
import os
import textwrap
from collections import Counter
from datetime import date, datetime
from pathlib import Path
from urllib.parse import quote

from dotenv import load_dotenv

load_dotenv()

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from bond import search
from fulltext import FullTextUnavailable, get_full_text, match_pdf
from project import list_projects, load_project, merge_papers, new_project_dir, save_attachment, save_project
from research import analyze, github_search, markdown_report, normalize, plan_for_gap
from review import review_markdown, review_paper
from quotes import quote_match
from troubleshoot import diagnose, find_problem_papers

st.set_page_config(page_title="ResearchPilot | Literature to Laboratory", page_icon="🔬", layout="wide")

# Lab-notebook palette: graph paper, blueprint ink, and a highlighter reserved for the user's own choices.
PAPER = "#F2F5F3"
GRID = "#DFE7E2"
INK = "#1A2744"
BLUE = "#2449A8"
PENCIL = "#5D6A7A"
HIGHLIGHT = "#F5E05A"
HIGHLIGHT_SOFT = "#FBF3B5"
ANALYSIS_CAP = 30  # research.analyze reads at most this many papers

EXAMPLES = {
    "Phase ID from SAED patterns": "AI-based crystal phase identification from selected area electron diffraction patterns",
    "4D-STEM strain mapping": "Machine learning for strain mapping in 4D-STEM datasets",
    "Low-dose cryo-EM denoising": "Deep learning denoising of low-dose cryo-EM micrographs",
    "Neural XRD phase analysis": "Automated phase analysis of powder X-ray diffraction patterns with neural networks",
}

ss = st.session_state
ss.setdefault("question", EXAMPLES["Phase ID from SAED patterns"])
ss.setdefault("shortlist", set())
ss.setdefault("nonce", 0)       # bumps on every new search to reset per-search widgets
ss.setdefault("tl_nonce", 0)    # bumps to clear the timeline selection
ss.setdefault("analysis_id", 0)
ss.setdefault("reviews", {})     # paper id -> full-text review
ss.setdefault("fulltexts", {})   # paper id -> fetched full text (sections)
ss.setdefault("ft_errors", {})   # paper id -> why no full text was found
ss.setdefault("lab_log", [])     # results logged from the lab, with any diagnosis
ss.setdefault("timeline", [])    # what happened in this project, in order
ss.setdefault("read_mode", "full")  # "full": every step reads papers in full; "abstract": faster, abstracts only

st.markdown(
    f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Source+Serif+4:ital,opsz,wght@0,8..60,400;0,8..60,600;1,8..60,400&display=swap');
.stApp {{
  background-color: {PAPER};
  background-image:
    linear-gradient({GRID} 1px, transparent 1px),
    linear-gradient(90deg, {GRID} 1px, transparent 1px);
  background-size: 32px 32px;
}}
[data-testid="stHeader"] {{ background: transparent; }}
[class*="st-key-panel"] {{ background: #FFFFFF; }}
.rp-head {{ display: flex; align-items: baseline; flex-wrap: wrap; gap: .25rem 1rem; margin: 0 0 .5rem; }}
.rp-mark {{ font-weight: 600; font-size: 1.75rem; letter-spacing: -0.015em; color: {INK}; }}
.rp-tag {{ color: {PENCIL}; font-size: 1rem; }}
.st-key-question textarea {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.35rem; line-height: 1.45; color: {INK};
}}
.rp-title {{
  font-family: 'Source Serif 4', Georgia, serif; font-weight: 600; font-size: 1.3rem;
  line-height: 1.35; color: {INK}; margin: 0 0 .35rem;
}}
.rp-title-sm {{ font-size: 1.08rem; }}
.rp-hl {{
  background: linear-gradient(transparent 55%, {HIGHLIGHT} 55%, {HIGHLIGHT} 92%, transparent 92%);
  -webkit-box-decoration-break: clone; box-decoration-break: clone;
}}
.rp-meta {{ color: {PENCIL}; font-size: .92rem; margin-bottom: .6rem; }}
.rp-abstract {{
  font-family: 'Source Serif 4', Georgia, serif; font-size: 1.02rem; line-height: 1.62;
  color: {INK}; max-width: 68ch;
}}
.rp-quote {{
  font-family: 'Source Serif 4', Georgia, serif; font-style: italic; margin: .25rem 0 .5rem;
  border-left: 3px solid {HIGHLIGHT}; padding: .1rem 0 .1rem .8rem; color: {INK};
}}
.rp-summary {{ font-size: 1.05rem; margin: .25rem 0 0; max-width: 90ch; }}
.st-key-panel_welcome, .st-key-panel_revision {{ box-shadow: inset 6px 0 0 {HIGHLIGHT}; }}
</style>
""",
    unsafe_allow_html=True,
)

if "flash" in ss:
    text, icon = ss.pop("flash")
    st.toast(text, icon=icon)

esc = html.escape


# ---------- helpers ----------
def clear_keys(*prefixes):
    for key in [k for k in list(ss.keys()) if isinstance(k, str) and k.startswith(prefixes)]:
        del ss[key]


def short_authors(authors):
    names = [a.strip() for a in str(authors or "").split(",") if a.strip()]
    if not names:
        return ""
    return names[0] + (" et al." if len(names) > 1 else "")


def papers_count(n, word="paper"):
    return f"{n} {word}{'' if n == 1 else 's'}"


def status_color(status):
    return {"reported": "blue", "literature-backed": "blue", "hypothesis": "violet", "unverified": "orange",
            "worked": "green", "partly worked": "orange", "problem": "red"}.get(str(status).lower(), "gray")


def confidence_color(confidence):
    text = str(confidence).lower()
    if "high" in text:
        return "green"
    if "med" in text or "moderate" in text:
        return "orange"
    return "gray"


def use_example():
    choice = ss.get("example")
    if choice:
        ss.question = EXAMPLES[choice]
        ss.example = None


def toggle_shortlist(pid):
    if pid in ss.shortlist:
        ss.shortlist.discard(pid)
    else:
        ss.shortlist.add(pid)


def shortlist_many(ids):
    ss.shortlist |= set(ids)
    ss.flash = (f"Shortlisted {len(ids)} papers", ":material/star:")


def clear_shortlist():
    ss.shortlist = set()
    ss.flash = ("Shortlist cleared", ":material/delete_sweep:")


def show_all_years():
    ss.tl_nonce += 1


def run_connection_test(token):
    try:
        from bond import diagnose

        with st.spinner("Connecting and listing Bond's tools"):
            ss.conn_check = {"ok": True, "data": diagnose(token)}
    except Exception as exc:  # shown to the user below
        ss.conn_check = {"ok": False, "error": exc}


SHARED = "__shared__"  # step tag for setup steps every pursued gap needs


def _dot_label(text, width=30):
    raw = str(text).replace("\\", "\\\\").replace('"', '\\"')
    return "\\n".join(textwrap.wrap(raw, width)) or " "


def roadmap_dot(steps, done, pursued=None):
    """Draft plan: one chain. With pursued gaps: shared setup, then a branch per gap ending in its success test."""
    lines = [
        "digraph roadmap {",
        'graph [rankdir=TB, bgcolor="transparent", nodesep=0.3, ranksep=0.32, pad=0.15, compound=true];',
        f'node [shape=box, fontname="IBM Plex Sans", fontsize=11, color="{INK}", fontcolor="{INK}", '
        'penwidth=1.2, margin="0.2,0.1"];',
        f'edge [color="{PENCIL}", arrowsize=0.7];',
    ]

    def node(i, step):
        style = "rounded,filled" + ("" if step.get("evidence_ids") else ",dashed")
        fill = HIGHLIGHT if done[i - 1] else "#FFFFFF"
        label = _dot_label(f'{i}. {step.get("title", "Task")}')
        return f'n{i} [label="{label}", style="{style}", fillcolor="{fill}"];'

    def chain(members):
        return [f"n{a} -> n{b};" for (a, _), (b, _) in zip(members, members[1:])]

    numbered = list(enumerate(steps, 1))
    pursued = pursued or []
    if not pursued:
        lines += [node(i, st_) for i, st_ in numbered] + chain(numbered)
    else:
        shared = [(i, st_) for i, st_ in numbered if st_.get("gap") == SHARED]
        lines += [node(i, st_) for i, st_ in shared] + chain(shared)
        titles = [g["title"] for g in pursued]
        titles += [t for t in dict.fromkeys(st_.get("gap") for _, st_ in numbered) if t and t != SHARED and t not in titles]
        for k, title in enumerate(titles):
            members = [(i, st_) for i, st_ in numbered if st_.get("gap") == title]
            if not members:
                continue
            gap = next((g for g in pursued if g["title"] == title), {})
            lines.append(f'subgraph cluster_{k} {{ label="{_dot_label("Gap: " + title, 44)}"; labeljust="l"; labelloc="b"; '
                         f'style="rounded"; color="{PENCIL}"; fontname="IBM Plex Sans"; fontsize=11; fontcolor="{INK}";')
            lines += [node(i, st_) for i, st_ in members] + chain(members)
            if gap.get("success_criteria"):
                lines.append(f'goal{k} [shape=note, style="filled", fillcolor="{HIGHLIGHT_SOFT}", '
                             f'label="{_dot_label("Closed when: " + gap["success_criteria"], 34)}"];')
                lines.append(f"n{members[-1][0]} -> goal{k};")
            lines.append("}")
            if shared:
                lines.append(f"n{shared[-1][0]} -> n{members[0][0]};")
        loose = [(i, st_) for i, st_ in numbered if not st_.get("gap")]
        lines += [node(i, st_) for i, st_ in loose] + chain(loose)
    lines.append("}")
    return "\n".join(lines)


def mermaid_source(steps, pursued=None):
    def clean(s):
        return str(s).replace('"', "'").replace("\n", " ")[:100]

    numbered = list(enumerate(steps, 1))
    lines = ["flowchart TD"] + [f'  N{i}["{i}. {clean(s.get("title", "Task"))}"]' for i, s in numbered]
    if not pursued:
        lines += [f"  N{i} --> N{i + 1}" for i in range(1, len(steps))]
        return "\n".join(lines)
    shared = [i for i, s in numbered if s.get("gap") == SHARED]
    lines += [f"  N{a} --> N{b}" for a, b in zip(shared, shared[1:])]
    for k, g in enumerate(pursued):
        members = [i for i, s in numbered if s.get("gap") == g["title"]]
        if not members:
            continue
        lines.append(f'  subgraph G{k}["Gap: {clean(g["title"])}"]')
        lines += [f"    N{m}" for m in members] + [f"    N{a} --> N{b}" for a, b in zip(members, members[1:])]
        if g.get("success_criteria"):
            lines += [f'    D{k}["Closed when: {clean(g["success_criteria"])}"]', f"    N{members[-1]} --> D{k}"]
        lines.append("  end")
        if shared:
            lines.append(f"  N{shared[-1]} --> N{members[0]}")
    return "\n".join(lines)


# ---------- projects ----------
PROGRESS_KEYS = ("step_done_",)  # pursued gaps live in the analysis itself


def now_str():
    return datetime.now().isoformat(timespec="seconds")


def log_event(kind, text):
    ss.timeline.append({"time": now_str(), "kind": kind, "text": text})


def project_snapshot():
    return {
        "name": ss.get("project_name") or "", "question": ss.get("research_question", ""),
        "search": ss.get("project_search") or {"start": str(ss.get("start", "")), "end": str(ss.get("end", ""))},
        "papers": ss.get("papers") or [], "shortlist": sorted(ss.shortlist), "focus_id": ss.get("focus_id"),
        "reviews": ss.reviews, "ft_errors": ss.ft_errors, "analysis": ss.get("analysis"),
        "flags": {k: bool(v) for k, v in ss.items() if isinstance(k, str) and k.startswith(PROGRESS_KEYS)},
        "lab_log": ss.lab_log, "timeline": ss.timeline, "pending_revision": ss.get("pending_revision"),
        "read_mode": ss.get("read_mode", "full"),
    }


def apply_snapshot(data, project_dir):
    """Restore a saved project. Only called from callbacks, which run before any widget is drawn."""
    clear_keys(*PROGRESS_KEYS)
    ss.papers = data.get("papers") or []
    ss.research_question = data.get("question", "")
    if ss.research_question:
        ss.question = ss.research_question
    ss.shortlist = set(data.get("shortlist") or [])
    ss.focus_id = data.get("focus_id") or (ss.papers[0]["id"] if ss.papers else None)
    ss.reviews, ss.fulltexts, ss.ft_errors = data.get("reviews") or {}, {}, data.get("ft_errors") or {}
    if data.get("analysis"):
        ss.analysis = data["analysis"]
    else:
        ss.pop("analysis", None)
    for key, value in (data.get("flags") or {}).items():
        ss[key] = value
    ss.lab_log, ss.timeline = data.get("lab_log") or [], data.get("timeline") or []
    ss.pending_revision = data.get("pending_revision")
    ss.read_mode = data.get("read_mode") or "full"
    ss.project_dir, ss.project_name = str(project_dir), data.get("name") or Path(project_dir).name
    ss.project_search = data.get("search") or {}
    ss.nonce += 1
    ss.analysis_id += 1
    for key in ("repos", "pdf_match_cache", "last_batch", "_saved_hash", "show_diagnosis"):
        ss.pop(key, None)
    ss.welcome = {"saved_at": data.get("saved_at", "")}


def create_project(name_key="new_project_name"):
    name = (ss.get(name_key) or "").strip() or (ss.get("research_question") or "Research project")[:50]
    ss.project_dir, ss.project_name = str(new_project_dir(name)), name
    ss.project_search = {"start": str(ss.get("start", "")), "end": str(ss.get("end", ""))}
    log_event("project", f'Saved as project "{name}"')
    ss.saved_at = save_project(ss.project_dir, project_snapshot())
    ss.flash = (f'Saved as project "{name}". From now on it saves automatically.', ":material/save:")


def open_project(project_dir):
    if not project_dir:
        return
    try:
        data = load_project(project_dir)
    except (OSError, ValueError) as exc:
        ss.flash = (f"Couldn't open that project: {exc}", ":material/error:")
        return
    apply_snapshot(data, project_dir)
    log_event("project", "Reopened the project")
    ss.flash = (f'Opened "{ss.project_name}"', ":material/folder_open:")


def open_uploaded_project(key):
    upload = ss.get(key)
    if not upload:
        return
    try:
        data = json.loads(upload.getvalue().decode("utf-8"))
        assert isinstance(data, dict) and "papers" in data
    except Exception:
        ss.flash = ("That file isn't a ResearchPilot project.", ":material/error:")
        return
    project_dir = new_project_dir(data.get("name") or "Imported project")
    save_project(project_dir, data)
    open_project(str(project_dir))


def close_project():
    for key in ("project_dir", "project_name", "project_search", "welcome", "pending_revision", "saved_at",
                "_saved_hash", "papers", "analysis", "show_diagnosis", "last_batch"):
        ss.pop(key, None)
    ss.shortlist, ss.reviews, ss.fulltexts, ss.ft_errors, ss.lab_log, ss.timeline = set(), {}, {}, {}, [], []
    clear_keys(*PROGRESS_KEYS)
    ss.nonce += 1
    ss.flash = ("Project closed. It's saved, so you can open it again any time.", ":material/folder:")


def step_toggled(i, title):
    log_event("plan", f'Marked step {i} "{title}" as {"done" if ss.get(f"step_done_{i}") else "not done"}')


def done_titles():
    current = (ss.get("analysis") or {}).get("steps", [])
    return {s.get("title") for i, s in enumerate(current, 1) if ss.get(f"step_done_{i}")}


def set_steps(new_steps, keep_done):
    """Replace the roadmap. Steps are matched by title, so anything already done stays ticked.
    Only call before the step checkboxes are drawn (callbacks, or the tabs that come before the Roadmap)."""
    ss.analysis["steps"] = new_steps
    clear_keys("step_done_")
    for i, step in enumerate(new_steps, 1):
        if step.get("title") in keep_done:
            ss[f"step_done_{i}"] = True


def pursued_gaps():
    return (ss.get("analysis") or {}).get("pursued") or []


def is_pursued(gap):
    return any(g["title"] == gap.get("title") for g in pursued_gaps())


def stop_pursuing(title):
    a = ss.get("analysis")
    if not a:
        return
    keep = done_titles()
    a["pursued"] = [g for g in a.get("pursued", []) if g["title"] != title]
    (a.get("plan_resources") or {}).pop(title, None)
    if a["pursued"]:
        new_steps = [s for s in a.get("steps", []) if s.get("gap") != title]
    else:
        new_steps = a.pop("draft_steps", None) or []
    set_steps(new_steps, keep)
    ss.pop("pending_revision", None)
    log_event("plan", f'Stopped pursuing "{title}"; its steps were removed from the roadmap')
    ss.flash = (f'Stopped pursuing "{title}". Its branch is gone from the roadmap; everything else is unchanged.',
                ":material/flag:")


def apply_revision():
    pending = ss.pop("pending_revision", None)
    if not pending or not ss.get("analysis"):
        return
    current, k = ss.analysis.get("steps", []), pending["from_step"]
    flags = {str(i): bool(ss.get(f"step_done_{i}")) for i in range(1, len(current) + 1)}
    ss.analysis.setdefault("revisions", []).append(
        {"time": now_str(), "entry_id": pending["entry_id"], "from_step": k, "old_steps": current, "old_flags": flags})
    branch = current[k - 1].get("gap") if 0 < k <= len(current) else None
    revised = [{**s, "added_after": pending["entry_id"], **({"gap": branch} if branch else {})} for s in pending["steps"]]
    # Replace this step and the rest of its own branch; other gaps' branches stay as they are.
    after = [s for s in current[k:] if branch and s.get("gap") != branch]
    set_steps(current[:k - 1] + revised + after, done_titles())
    log_event("plan", f'Revised the roadmap from step {k} after lab note {pending["entry_id"]}')
    ss.pop("show_diagnosis", None)  # it stays in the Lab notebook; the revised steps are what matters now
    ss.flash = (f"Roadmap revised from step {k}. Steps before it are unchanged; the diagnosis is in the Lab notebook.",
                ":material/route:")


def dismiss_revision():
    pending = ss.pop("pending_revision", None)
    if pending:
        log_event("plan", f'Kept the current roadmap instead of the plan proposed after {pending["entry_id"]}')


def undo_revision():
    revisions = (ss.get("analysis") or {}).get("revisions") or []
    if not revisions:
        return
    last = revisions.pop()
    ss.analysis["steps"] = last["old_steps"]
    clear_keys("step_done_")
    for i, flag in (last.get("old_flags") or {}).items():
        ss[f"step_done_{i}"] = flag
    log_event("plan", f'Undid the roadmap revision made after {last["entry_id"]}')
    ss.flash = ("Previous roadmap restored.", ":material/undo:")


# ---------- sidebar ----------
with st.sidebar:
    st.subheader("Project")
    if ss.get("project_dir"):
        st.markdown(f"**{ss.project_name}**")
        saved = ss.get("saved_at") or ""
        st.caption(f"Saves automatically. Last saved at {saved[11:16]} on {saved[:10]}." if saved else "Saves automatically.")
        st.caption(f"Saved in projects/{Path(ss.project_dir).name}, next to the app.")
        st.button("Close project", icon=":material/folder:", on_click=close_project, width="stretch")
    else:
        if ss.get("papers"):
            st.text_input("Project name", key="new_project_name",
                          placeholder=(ss.get("research_question") or "")[:50])
            st.button("Save as project", icon=":material/save:", on_click=create_project, type="primary",
                      width="stretch")
            st.caption("Save to come back later, log lab results, and pick up where you stopped.")
        saved_projects = list_projects()
        if saved_projects:
            labels = {p["dir"]: f'{p["name"]} ({p["saved_at"][:10]})' for p in saved_projects}
            choice = st.selectbox("Open a saved project", list(labels), format_func=labels.get, index=None,
                                  placeholder="Choose a project", key="open_choice")
            st.button("Open project", icon=":material/folder_open:", on_click=open_project, args=(choice,),
                      disabled=not choice, width="stretch")
        upload_key = f"project_upload_{ss.nonce}"
        st.file_uploader("Or open a project file", type=["json"], key=upload_key,
                         on_change=open_uploaded_project, args=(upload_key,))
    st.divider()
    st.subheader("Valency Bond")
    bond_token = st.text_input(
        "Bearer token",
        value=os.getenv("VALENCY_BEARER_TOKEN", ""),
        type="password",
        help="Paste the raw tok_... value from Valency Settings > API Token, without the word Bearer.",
    )
    if bond_token.strip():
        st.badge("Token added", icon=":material/check:", color="green")
    else:
        st.badge("Add a token to search", icon=":material/key:", color="orange")
    if st.button("Test connection", icon=":material/cable:", width="stretch"):
        run_connection_test(bond_token)
    check = ss.get("conn_check")
    if check and check["ok"]:
        data = check["data"]
        st.success(f"Connected. Bond offers {len(data['tools'])} tools.")
        with st.expander("Tools and their parameters"):
            for name in data["tools"]:
                params = ", ".join(data["parameters"].get(name, [])) or "No parameters"
                st.markdown(f"**{name}**  \n{params}")
            if data["sources"] is not None:
                st.caption("list_sources result")
                st.json(data["sources"], expanded=False)
    elif check:
        st.error("Connection failed. Check the token, then test again.")
        with st.expander("Error details"):
            st.exception(check["error"])

    st.divider()
    st.subheader("Analysis model")
    provider = st.segmented_control(
        "Provider", ["Local Ollama", "OpenAI API"], default="Local Ollama", required=True, key="provider"
    ) or "Local Ollama"
    if provider == "OpenAI API":
        openai_key = st.text_input("OpenAI API key", value=os.getenv("OPENAI_API_KEY", ""), type="password")
        model = st.text_input("Model", value="gpt-4.1-mini", key="model_openai")
        base_url = ""
        st.caption("Billed per request by OpenAI, separately from a ChatGPT subscription.")
    else:
        openai_key = ""
        model = st.text_input("Model", value="qwen3:8b", key="model_ollama")
        base_url = st.text_input("Ollama API base URL", value="http://localhost:11434/v1")
        st.caption("Runs on your own computer, with no per-request fee.")

    st.divider()
    st.subheader("Reading depth")
    st.segmented_control("How should papers be read?", ["full", "abstract"], required=True, key="read_mode",
                         format_func={"full": "Full papers", "abstract": "Abstracts only"}.get)
    if ss.read_mode == "full":
        st.caption("Gap analysis, resources, roadmap and troubleshooting all read papers in full, fetching the full "
                   "text when needed. Papers with no open copy use their abstract until you upload the PDF.")
    else:
        st.caption("Faster: papers are read by abstract. Full-text reviews you've already done are still used.")

    st.divider()
    st.subheader("Full text")
    unpaywall_email = st.text_input(
        "Your email for Unpaywall (optional)", value=os.getenv("UNPAYWALL_EMAIL", ""),
        help="Unpaywall finds legal open-access copies of journal papers and asks for a contact email. "
             "Without it, full text comes from arXiv and PDFs you upload.",
    )
    st.caption("Full text is fetched only for papers you choose to review, and stays on this computer.")

    st.divider()
    st.subheader("GitHub")
    github_token = st.text_input(
        "Token (optional)", value=os.getenv("GITHUB_TOKEN", ""), type="password",
        help="Raises GitHub's search rate limit.",
    )
    st.caption(
        "Keys are only used for requests from this running session. Don't deploy a shared "
        "instance with secrets in environment variables."
    )
    st.markdown("[Valency setup guide](https://docs.valency.io/)")


# ---------- header and search ----------
st.markdown(
    '<div class="rp-head"><span class="rp-mark">ResearchPilot</span>'
    '<span class="rp-tag">From literature to laboratory, with evidence from Valency Bond</span></div>',
    unsafe_allow_html=True,
)
if ss.get("project_dir"):
    st.badge(f"Project: {ss.project_name}", icon=":material/folder_open:", color="blue")

with st.container(border=True, key="panel_query"):
    st.text_area("What do you want to research?", key="question", height=96)
    st.pills("Or start from an example", list(EXAMPLES), key="example", on_change=use_example)
    c1, c2, c3, c4 = st.columns([1, 1, 1.4, 1.2], vertical_alignment="bottom")
    start = c1.date_input("Published after", value=date(2020, 1, 1), key="start")
    end = c2.date_input("Published before", value=date.today(), key="end")
    limit = c3.slider("Results per search tool", 5, 100, 25, 5, key="limit")
    go_search = c4.button("Search and add to project" if ss.get("project_dir") and ss.get("papers") else "Search Valency Bond",
                          type="primary", icon=":material/search:", width="stretch")
    st.toggle("Look up missing years in Crossref (slower)", value=True, key="enrich")

if go_search:
    question = ss.question.strip()
    if not question:
        st.error("Enter a research question, then search again.")
    elif start > end:
        st.error("The 'Published after' date must be earlier than 'Published before'.")
    elif not bond_token.strip():
        st.error("Add your Valency token in the sidebar, then search again.")
    else:
        with st.status("Searching Valency Bond", expanded=True) as status:
            try:
                st.write("Connecting and calling Bond's search tools")
                results = search(bond_token, question, start.isoformat(), end.isoformat(), limit)
                st.write("Removing duplicates and checking publication years"
                         + (", with Crossref for missing years" if ss.enrich else ""))
                found = normalize(results, start.isoformat(), end.isoformat(), enrich_doi=ss.enrich)
                if ss.get("project_dir") and ss.get("papers"):
                    # In a project, a new search adds papers and keeps reviews, plan and lab log.
                    ss.papers, added = merge_papers(ss.papers, found, note=f'search "{question[:60]}"')
                    log_event("search", f'Searched Bond for "{question}": {len(found)} papers, {len(added)} new')
                    status.update(label=f"Found {len(found)} papers; {len(added)} new added to the project",
                                  state="complete", expanded=False)
                else:
                    ss.papers = found
                    ss.research_question = question
                    ss.shortlist = set()
                    ss.focus_id = found[0]["id"] if found else None
                    ss.nonce += 1
                    ss.pop("analysis", None)
                    ss.pop("repos", None)
                    ss.reviews, ss.fulltexts, ss.ft_errors = {}, {}, {}
                    ss.pop("pdf_match_cache", None)
                    ss.pop("last_batch", None)
                    clear_keys("step_done_")
                    ss.timeline = []
                    log_event("search", f'Searched Bond for "{question}": {len(found)} papers')
                    status.update(label=f"Found {len(found)} papers", state="complete", expanded=False)
            except Exception as exc:
                status.update(label="Search failed", state="error", expanded=True)
                st.error("Bond search failed. The details below don't include your token.")
                st.exception(exc)

papers = ss.get("papers")
if papers is None:
    st.markdown(
        "Results appear here after you search. Nothing is pre-filled: every paper, gap and "
        "citation comes from your live Bond search."
    )
    st.stop()
if not papers:
    st.warning("Bond returned no paper records this app could read. Try a narrower question or a wider date range.")
    st.stop()

question = ss.get("research_question", ss.question)
index = {p["id"]: p for p in papers}
shortlist = ss.shortlist
years = [p["year"] for p in papers if p["year"]]
with_abstract = sum(1 for p in papers if p["abstract"])
span = f", published {min(years)} to {max(years)}" if years else ""
reviews = {pid: rv for pid, rv in ss.reviews.items() if pid in index}
if shortlist:
    chosen_text = f'<span class="rp-hl">{len(shortlist)} {"is" if len(shortlist) == 1 else "are"} shortlisted</span> for analysis'
else:
    chosen_text = "none are shortlisted yet"
reviewed_text = f" {len(reviews)} {'has' if len(reviews) == 1 else 'have'} been reviewed in full." if reviews else ""
st.markdown(
    f'<p class="rp-summary"><strong>{papers_count(len(papers))}</strong> found for '
    f"“{esc(question)}”{span}. {with_abstract} include an abstract, and {chosen_text}.{reviewed_text}</p>",
    unsafe_allow_html=True,
)
st.caption("Bond search is selective, not exhaustive. Unretrieved work may exist.")

OLLAMA_HINT = " Check that Ollama is running and the model is pulled, for example `ollama pull qwen3:8b`."


def run_review(paper, uploaded_pdf=None):
    """Fetch a paper's full text (or use an uploaded PDF), review it, and store the result. Returns True on success."""
    pid = paper["id"]
    with st.status(f"Reviewing {pid} in full", expanded=True) as status:
        try:
            if uploaded_pdf is not None or pid not in ss.fulltexts:
                st.write("Reading your PDF" if uploaded_pdf is not None
                         else "Looking for the full text: arXiv first, then open-access copies")
                ss.fulltexts[pid] = get_full_text(paper, unpaywall_email, uploaded_pdf)
            ft = ss.fulltexts[pid]
            ss.ft_errors.pop(pid, None)
            st.write(f"Got {len(ft['sections'])} sections ({ft['chars']:,} characters) from {ft['source']}")
            bar = st.progress(0.0, text="Starting the review")
            ss.reviews[pid] = review_paper(
                paper, ft, question, openai_key, model, provider=provider, base_url=base_url,
                progress=lambda done, total, label: bar.progress(min(done / total, 1.0), text=label),
            )
            status.update(label=f"Reviewed {pid} from {ft['source']}", state="complete", expanded=False)
            log_event("review", f"Reviewed {pid} in full from {ft['source']}")
            return True
        except FullTextUnavailable as exc:
            ss.ft_errors[pid] = str(exc)
            status.update(label=f"No open full text found for {pid}", state="error", expanded=False)
        except Exception as exc:
            status.update(label=f"Review of {pid} failed", state="error", expanded=True)
            st.error(f"{exc}" + (OLLAMA_HINT if provider == "Local Ollama" else ""))
    return False


def run_batch(batch, review_it=True, uploads=None):
    """Find full text for (and optionally review) many papers in one go, saving each as it finishes."""
    uploads = uploads or {}
    total = len(batch)
    counts, lines = Counter(), []
    verb = "Reviewing" if review_it else "Finding full text for"
    with st.status(f"{verb} {papers_count(total)}", expanded=True) as status:
        overall = st.progress(0.0, text="Starting")
        current = st.progress(0.0, text="Waiting for the first paper") if review_it else None
        for i, p in enumerate(batch):
            pid = p["id"]
            overall.progress(i / total, text=f"Paper {i + 1} of {total}: {p['title'][:90]}")
            try:
                if pid in uploads:
                    ss.fulltexts[pid] = get_full_text(p, unpaywall_email, uploads[pid])
                elif pid not in ss.fulltexts:
                    ss.fulltexts[pid] = get_full_text(p, unpaywall_email)
                ss.ft_errors.pop(pid, None)
                source = ss.fulltexts[pid]["source"]
                if review_it:
                    current.progress(0.0, text=f"{pid}: starting the review")
                    ss.reviews[pid] = review_paper(
                        p, ss.fulltexts[pid], question, openai_key, model, provider=provider, base_url=base_url,
                        progress=lambda done, n, label: current.progress(min(done / n, 1.0), text=f"{pid}: {label}"),
                    )
                    line = f"{pid}: reviewed from {source}"
                else:
                    line = f"{pid}: full text found ({source})"
                counts["ok"] += 1
            except FullTextUnavailable as exc:
                ss.ft_errors[pid] = str(exc)
                counts["missing"] += 1
                line = f"{pid}: no open full text"
            except Exception as exc:
                counts["failed"] += 1
                line = f"{pid}: failed ({str(exc)[:200]})"
            lines.append(line)
            st.write(line)
            if counts["failed"] >= 2 and not counts["ok"]:
                lines.append("Stopped early: the first papers all failed, so the model may be unreachable."
                             + (OLLAMA_HINT if provider == "Local Ollama" else ""))
                st.error(lines[-1])
                break
        overall.progress(1.0, text="Finished")
        label = f"{'Reviewed' if review_it else 'Found full text for'} {counts['ok']} of {papers_count(total)}"
        if counts["missing"]:
            label += f"; {counts['missing']} without an open full text"
        if counts["failed"]:
            label += f"; {counts['failed']} failed"
        status.update(label=label, state="error" if counts["failed"] else "complete", expanded=False)
    ss.last_batch = {"label": label, "lines": lines, "failed": counts["failed"]}
    if counts["ok"]:
        log_event("review", label)
    return counts


def review_one(p, progress=None):
    """Fetch and review one paper inside another step's progress box. Raises FullTextUnavailable."""
    pid = p["id"]
    try:
        if pid not in ss.fulltexts:
            ss.fulltexts[pid] = get_full_text(p, unpaywall_email)
    except FullTextUnavailable as exc:
        ss.ft_errors[pid] = str(exc)
        raise
    ss.ft_errors.pop(pid, None)
    ss.reviews[pid] = review_paper(p, ss.fulltexts[pid], question, openai_key, model, provider=provider,
                                   base_url=base_url, progress=progress)
    log_event("review", f"Reviewed {pid} in full from {ss.fulltexts[pid]['source']}")
    return ss.fulltexts[pid]["source"]
    ss.flash = (label, ":material/menu_book:")


def render_items(items, field, detail_fields=()):
    """List extracted items with their quote, section, and whether the quote was found in the paper."""
    for item in sorted(items, key=lambda x: not x.get("verified")):
        detail = " ".join(str(item.get(f)) for f in detail_fields if item.get(f))
        st.markdown(f'**{item.get(field, "")}**' + (f": {detail}" if detail else ""))
        if item.get("match") == "close":
            check = "Quote closely matches the paper (minor wording differences; numbers checked)."
        elif item.get("verified"):
            check = "Quote found in the paper."
        else:
            check = "Quote not found in the paper's text; check it before relying on it."
        if item.get("quote"):
            st.markdown(f'<blockquote class="rp-quote">“{esc(str(item["quote"]))}”</blockquote>', unsafe_allow_html=True)
        st.caption(f'{item.get("section", "")}. {check}')


def evidence_row(ids, key):
    """Each cited paper opens in a popover, so evidence can be read without leaving the page."""
    ids = [pid for pid in ids if pid in index]
    if not ids:
        st.caption("No supporting paper in these results. Treat this as a proposal.")
        return
    with st.container(horizontal=True, gap="small"):
        for pid in ids:
            p = index[pid]
            rv = reviews.get(pid)
            icon = ":material/menu_book:" if rv else ":material/description:"
            hint = ("Reviewed in full. " if rv else "Abstract only. ") + p["title"][:140]
            with st.popover(pid, icon=icon, help=hint, key=f"pop_{key}_{pid}"):
                meta = str(p["year"] or "Year unknown")
                if p["authors"]:
                    meta += ", " + short_authors(p["authors"])
                st.markdown(
                    f'<div class="rp-title rp-title-sm">{esc(p["title"])}</div>'
                    f'<div class="rp-meta">{esc(meta)}</div>',
                    unsafe_allow_html=True,
                )
                abstract = p["abstract"]
                if rv:
                    st.badge(f'Reviewed in full from {rv["source"]}', icon=":material/menu_book:", color="green")
                    st.markdown(f'<div class="rp-abstract">{esc(rv.get("summary") or "")}</div>', unsafe_allow_html=True)
                    st.caption("Open the Full-text reviews tab for its methods, results and limitations.")
                elif abstract:
                    clipped = abstract[:1200] + ("…" if len(abstract) > 1200 else "")
                    st.markdown(f'<div class="rp-abstract">{esc(clipped)}</div>', unsafe_allow_html=True)
                else:
                    st.caption("No abstract was returned for this paper.")
                st.link_button("Open paper", p["url"], icon=":material/open_in_new:")


analysis = ss.get("analysis") or {}
gaps = [g for g in analysis.get("gaps", []) if isinstance(g, dict)]
resources = [r for r in analysis.get("resources", []) if isinstance(r, dict)]
steps = [s for s in analysis.get("steps", []) if isinstance(s, dict)]
done_flags = [bool(ss.get(f"step_done_{i}")) for i in range(1, len(steps) + 1)]

# ---------- lab log and troubleshooting ----------
OUTCOMES = ["Worked", "Partly worked", "Problem"]


def add_log_entry(step_i, did, observed, outcome, files):
    entry_id = f"L{len(ss.lab_log) + 1:03d}"
    step = steps[step_i - 1]
    attachments = [save_attachment(ss.project_dir, entry_id, f.name, f.getvalue()) for f in files or []]
    entry = {"id": entry_id, "created": now_str(), "step_index": step_i, "step_title": step.get("title", "Task"),
             "step": {k: step.get(k) for k in ("title", "details", "evidence_ids")},
             "gap": None if step.get("gap") in (None, SHARED) else step.get("gap"),
             "did": did.strip(), "observed": observed.strip(), "outcome": outcome or "Problem",
             "attachments": attachments, "diagnosis": None}
    ss.lab_log.append(entry)
    ss.pop("welcome", None)  # they've picked the project back up
    log_event("lab", f'Logged a result for step {step_i} ({entry["outcome"].lower()}): {entry["observed"][:120]}')
    if entry["outcome"] == "Worked":
        ss[f"step_done_{step_i}"] = True  # the step checkboxes are drawn later in this run, so this is allowed
    return entry


NEW_PAPERS_TO_READ = 3  # newly found papers read in full when troubleshooting (each takes minutes locally)


def read_in_full_first(candidates, why):
    """With reading depth set to full papers, read any of these papers that aren't read yet (inside a status box)."""
    if ss.read_mode != "full":
        return
    to_read = [p for p in candidates if p["id"] not in ss.reviews and p["id"] not in ss.ft_errors]
    if not to_read:
        return
    st.write(f"Reading {papers_count(len(to_read))} in full first ({why}): {', '.join(p['id'] for p in to_read)}")
    bar = st.progress(0.0, text="Starting")
    for n, p in enumerate(to_read, 1):
        try:
            source = review_one(p, progress=lambda d, t, label, n=n, pid=p["id"]: bar.progress(
                min(d / t, 1.0), text=f"{pid} ({n} of {len(to_read)}): {label}"))
            st.write(f"{p['id']}: read in full from {source}")
        except FullTextUnavailable:
            st.write(f"{p['id']}: no open full text, so its abstract is used")


def pursue_gap(gap):
    """Build (or extend) the roadmap, flowchart and resources around one gap. Returns True on success."""
    llm = dict(api_key=openai_key, model=model, provider=provider, base_url=base_url)
    with st.status(f'Planning how to close "{gap["title"]}"', expanded=True) as status:
        try:
            by_id = {p["id"]: p for p in ss.papers}
            read_in_full_first([by_id[i] for i in gap.get("evidence_ids", []) if i in by_id], "the papers behind this gap")
            a = ss.analysis
            current = a.get("steps", [])
            first = not a.get("pursued")
            shared = [s for s in current if s.get("gap") == SHARED]
            st.write("Writing the steps, resources and a success test for this gap"
                     + ("" if first else ", reusing the shared setup steps already in your roadmap"))
            out = plan_for_gap(gap, question, ss.papers, ss.reviews, shared, a.get("resources"), **llm)
            if not out["steps"]:
                raise ValueError("The model returned no steps for this gap. Try again, or use another model.")
            keep = done_titles()
            if first:
                a["draft_steps"] = current
                new_steps = [{**x, "gap": SHARED} for x in out["shared_steps"]] + [{**x, "gap": gap["title"]} for x in out["steps"]]
            else:
                new_steps = current + [{**x, "gap": gap["title"]} for x in out["steps"]]
            a.setdefault("pursued", []).append({
                "title": gap["title"], "explanation": gap.get("explanation", ""), "evidence_ids": gap.get("evidence_ids", []),
                "success_criteria": out["success_criteria"], "since": now_str()})
            a.setdefault("plan_resources", {})[gap["title"]] = out["resources"]
            set_steps(new_steps, keep)
            extra = f' and {len(out["shared_steps"])} shared setup steps' if first and out["shared_steps"] else ""
            log_event("plan", f'Now pursuing "{gap["title"]}": {len(out["steps"])} steps{extra} added to the roadmap')
            status.update(label=f'Plan ready for "{gap["title"]}"', state="complete", expanded=False)
            return True
        except Exception as exc:
            status.update(label="Planning failed", state="error", expanded=True)
            st.error(f"{exc}" + (OLLAMA_HINT if provider == "Local Ollama" else ""))
            return False


def run_troubleshoot(entry):
    """Resume from the step where the experiment ran into trouble. Returns True on success."""
    current = steps[entry["step_index"] - 1] if 0 < entry["step_index"] <= len(steps) else None
    plan_unchanged = bool(current) and current.get("title") == entry["step_title"]
    step = current if plan_unchanged else (entry.get("step") or {"title": entry["step_title"]})
    llm = dict(api_key=openai_key, model=model, provider=provider, base_url=base_url)
    with st.status(f'Picking up from step {entry["step_index"]}: {entry["step_title"]}', expanded=True) as status:
        try:
            find = None
            if bond_token.strip():
                since = str((ss.get("project_search") or {}).get("start") or ss.get("start") or "2015-01-01")[:10]
                today = date.today().isoformat()
                find = lambda q: normalize(search(bond_token, q, since, today, 10), since, today)  # noqa: E731
            else:
                st.write("No Valency token in the sidebar, so no new papers are searched; using this project's papers.")
            fresh, queries, notes = find_problem_papers(question, step, entry, ss.papers, find_papers=find,
                                                        progress=st.write, **llm)
            added = []
            if fresh:
                ss.papers, added = merge_papers(ss.papers, fresh, note=f'troubleshooting {entry["id"]}')
                st.write(f"Found {papers_count(len(added), 'new paper')} about this problem: {', '.join(p['id'] for p in added)}")

            by_id = {p["id"]: p for p in ss.papers}
            read_in_full_first([by_id[i] for i in step.get("evidence_ids", []) if i in by_id] + added[:NEW_PAPERS_TO_READ],
                               "this step's papers and new ones about the problem")

            diagnosis = diagnose(question, step, entry["step_index"], entry, ss.lab_log, ss.papers, added, ss.reviews,
                                 progress=st.write, **llm)
            diagnosis.update({"queries": queries, "notes": notes, "read_mode": ss.read_mode})
            entry["diagnosis"] = diagnosis
            if diagnosis.get("revised_steps") and plan_unchanged:
                ss.pending_revision = {"entry_id": entry["id"], "from_step": entry["step_index"],
                                       "steps": diagnosis["revised_steps"], "created": now_str()}
            log_event("diagnosis", f'Troubleshot step {entry["step_index"]}: {len(diagnosis["likely_causes"])} likely '
                                   f'causes, {len(diagnosis["new_paper_ids"])} new papers from Bond')
            ss.show_diagnosis = entry["id"]
            status.update(label="Diagnosis ready", state="complete", expanded=False)
            return True
        except Exception as exc:
            status.update(label="Troubleshooting failed", state="error", expanded=True)
            st.error(f"{exc}" + (OLLAMA_HINT if provider == "Local Ollama" else ""))
            return False


def render_diagnosis(entry, key):
    d = entry.get("diagnosis") or {}
    if d.get("summary"):
        st.markdown(f'<div class="rp-abstract">{esc(d["summary"])}</div>', unsafe_allow_html=True)
    used = [f'{d.get("evidence_items", 0)} quote-checked items from your full-text reviews',
            f'{papers_count(d.get("earlier_attempts", 0), "earlier attempt")} at this step']
    if d.get("new_paper_ids"):
        used.append(f'{papers_count(len(d["new_paper_ids"]), "new paper")} found in Valency Bond '
                    f'({", ".join(d["new_paper_ids"])}), now added to the project')
    if d.get("new_papers_read_in_full"):
        used.append(f'of which {", ".join(d["new_papers_read_in_full"])} read in full')
    st.caption("Context used: " + "; ".join(used) + ".")
    if d.get("likely_causes"):
        st.markdown("###### Likely causes")
        for j, cause in enumerate(d["likely_causes"], 1):
            with st.container(border=True, key=f"panel_cause_{key}_{j}"):
                st.markdown(f'**{cause.get("cause", "Possible cause")}**')
                with st.container(horizontal=True, gap="small"):
                    st.badge(str(cause.get("status", "hypothesis")).capitalize(), color=status_color(cause.get("status")))
                    st.badge(f'{cause.get("confidence", "unrated")} confidence'.capitalize(),
                             color=confidence_color(cause.get("confidence", "")))
                st.write(cause.get("explanation", ""))
                for q in cause.get("evidence_quotes", []):
                    st.markdown(f'<blockquote class="rp-quote">“{esc(str(q))}”</blockquote>', unsafe_allow_html=True)
                evidence_row(cause.get("evidence_ids", []), f"{key}c{j}")
    if d.get("checks"):
        st.markdown("###### Quick checks to tell the causes apart")
        st.dataframe(pd.DataFrame([{"Check": c.get("check", ""), "Tells apart": c.get("tells_apart", ""),
                                    "Expected result": c.get("expected_result", "")} for c in d["checks"]]),
                     hide_index=True, width="stretch")
    for note in d.get("notes", []):
        st.caption(note)
    st.caption("These are suggestions to test, not instructions. Check anything involving hazardous materials "
               "or equipment with your lab's safety procedures.")


def next_step_index():
    flags = [bool(ss.get(f"step_done_{i}")) for i in range(1, len(steps) + 1)]
    return next((i for i, done in enumerate(flags, 1) if not done), len(steps) or 1)


if ss.get("welcome"):
    with st.container(border=True, key="panel_welcome"):
        saved = ss.welcome.get("saved_at") or ""
        st.markdown(f"**Welcome back to {ss.project_name}.**"
                    + (f" You last worked on it on {saved[:10]} at {saved[11:16]}." if saved else ""))
        lines = []
        if pursued_gaps():
            lines.append("You're pursuing: " + "; ".join(f'"{g["title"]}"' for g in pursued_gaps()) + ".")
        if steps:
            k = next_step_index()
            lines.append(f"{sum(done_flags)} of {papers_count(len(steps), 'roadmap step')} done. "
                         f'Next up: step {k}, "{steps[k - 1].get("title", "Task")}".')
        if ss.lab_log:
            last = ss.lab_log[-1]
            lines.append(f'Your last lab note was on step {last["step_index"]} ({last["outcome"].lower()}): '
                         f'"{last["observed"][:160].rstrip()}"')
        if ss.get("pending_revision"):
            lines.append("A revised plan is waiting for your decision in the Roadmap tab.")
        if reviews:
            lines.append(f"{papers_count(len(reviews))} reviewed in full are ready to use.")
        st.markdown("\n".join(f"- {line}" for line in lines) or "Nothing logged yet.")
        st.caption("Log what happened in the Roadmap tab, and ResearchPilot will pick up from that step using your "
                   "papers, full-text reviews and earlier attempts.")
        st.button("Dismiss", type="tertiary", key="dismiss_welcome", on_click=lambda: ss.pop("welcome", None))


def next_action():
    scope_now = ([p for p in papers if p["id"] in shortlist] or papers)[:ANALYSIS_CAP]
    unread = [p for p in scope_now if p["id"] not in reviews and p["id"] not in ss.ft_errors]
    if not analysis:
        if ss.read_mode == "full" and unread:
            return (f"Find research gaps in 3. Research gaps: it reads {papers_count(len(unread))} in full first. "
                    + ("" if shortlist else "Shortlist papers in 1. Papers to focus it."))
        return "Find research gaps in 3. Research gaps."
    if not pursued_gaps():
        return "Pick a gap to pursue in 3. Research gaps. The roadmap and resources will be built around it."
    open_problem = next((e for e in reversed(ss.lab_log) if e["outcome"] != "Worked" and not e.get("diagnosis")), None)
    if open_problem:
        return f'Troubleshoot lab note {open_problem["id"]} in 6. Lab notebook, so the plan can pick up from there.'
    if ss.get("pending_revision"):
        return "A revised plan is waiting for your decision in 4. Roadmap."
    if steps and not all(done_flags):
        k = next_step_index()
        return (f'Work on step {k}, "{steps[k - 1].get("title", "Task")}", in 4. Roadmap'
                + ("" if ss.get("project_dir") else ", and save this as a project so you can log results and come back")
                + ".")
    return "Every step is done. Export your research packet, or pursue another gap."


st.markdown(f"<p class='rp-summary'><strong>Next:</strong> {esc(next_action())}</p>", unsafe_allow_html=True)

# Fixed labels plus a key keep the open tab across reruns (changing labels would jump back to Papers),
# and switching tabs stays instant, so it never interrupts a running batch. Counts appear inside each tab.
t_papers, t_reviews, t_gaps, t_roadmap, t_resources, t_notebook, t_export = st.tabs([
    "1. Papers",
    "2. Full-text reviews",
    "3. Research gaps",
    "4. Roadmap",
    "5. Resources",
    "6. Lab notebook",
    ":material/download: Export",
], key="main_tabs")

# ---------- papers ----------
with t_papers:
    picked_years = []
    counts = Counter(years)
    if counts:
        with st.container(border=True, key="panel_timeline"):
            xs = sorted(counts)
            fig = go.Figure(go.Bar(
                x=xs, y=[counts[x] for x in xs],
                marker=dict(color=BLUE, line=dict(color=INK, width=1)),
                selected=dict(marker=dict(color=HIGHLIGHT)),
                unselected=dict(marker=dict(opacity=0.4)),
                hovertemplate="%{x}: %{y} papers<extra></extra>",
            ))
            fig.update_layout(
                height=210, margin=dict(l=8, r=8, t=10, b=8), bargap=0.35,
                clickmode="event+select", dragmode="zoom",  # axes are fixed, so drag never zooms
                paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)", showlegend=False,
                font=dict(family="IBM Plex Sans, sans-serif", color=INK, size=13),
                xaxis=dict(type="category", showgrid=False, linecolor=PENCIL, fixedrange=True, automargin=True),
                yaxis=dict(gridcolor=GRID, zeroline=False, fixedrange=True, tickformat="d", automargin=True,
                           title=dict(text="Papers", font=dict(size=12, color=PENCIL))),
                hoverlabel=dict(bgcolor="#FFFFFF", font=dict(family="IBM Plex Sans, sans-serif", color=INK),
                                bordercolor=INK),
            )
            event = st.plotly_chart(
                fig, key=f"timeline_{ss.nonce}_{ss.tl_nonce}", on_select="rerun",
                selection_mode="points", theme=None, config={"displayModeBar": False},
            )
            if event and event.selection:
                picked_years = sorted({int(pt["x"]) for pt in event.selection.points if pt.get("x") is not None})
            with st.container(horizontal=True, vertical_alignment="center"):
                if picked_years:
                    label = ", ".join(map(str, picked_years))
                    st.badge(f"Showing {label}", icon=":material/filter_alt:", color="yellow")
                    st.button("Show all years", type="tertiary", on_click=show_all_years)
                else:
                    st.caption("Click a bar to see that year's papers. Shift-click to add more years.")

    left, right = st.columns([3, 2], gap="large")
    with left:
        with st.container(horizontal=True, vertical_alignment="bottom"):
            needle = st.text_input(
                "Filter papers", key=f"filter_{ss.nonce}", placeholder="Filter by title, abstract or author",
                icon=":material/filter_list:", label_visibility="collapsed",
            )
            order = st.selectbox(
                "Sort", ["Newest first", "Oldest first", "Title A to Z"], key="sort",
                label_visibility="collapsed", width=170,
            )
        with st.container(horizontal=True):
            only_abstracts = st.toggle("Only with abstracts", key=f"only_abs_{ss.nonce}")
            only_shortlisted = st.toggle("Only shortlisted", key=f"only_sl_{ss.nonce}")

        needle = (needle or "").lower()
        shown = [
            p for p in papers
            if (not picked_years or p["year"] in picked_years)
            and (not only_abstracts or p["abstract"])
            and (not only_shortlisted or p["id"] in shortlist)
            and needle in f'{p["title"]} {p["abstract"]} {p["authors"]}'.lower()
        ]
        if order == "Oldest first":
            shown.sort(key=lambda p: (p["year"] or 9999, p["title"]))
        elif order == "Title A to Z":
            shown.sort(key=lambda p: p["title"].lower())

        if not shown:
            st.info("No papers match these filters. Clear the filter text, switch off a toggle, or show all years.")
        else:
            df = pd.DataFrame([{
                "id": p["id"],
                "Shortlisted": "★" if p["id"] in shortlist else "",
                "Year": p["year"],
                "Title": p["title"],
                "Full text": "✓" if p["id"] in reviews else "",
            } for p in shown])
            df["Year"] = df["Year"].astype("Int64")
            styled = df.style.apply(
                lambda row: [f"background-color: {HIGHLIGHT_SOFT}" if row["Shortlisted"] else ""] * len(row), axis=1
            )
            # The key follows the visible rows, so a selected row index always maps to the paper on screen.
            table_key = f"papers_{ss.nonce}_{abs(hash(tuple(df['id'])))}"
            table = st.dataframe(
                styled, key=table_key, on_select="rerun", selection_mode="single-row", hide_index=True,
                column_order=["Shortlisted", "Year", "Full text", "Title"],
                column_config={
                    "Shortlisted": st.column_config.TextColumn("★", width=36, help="Shortlisted for analysis"),
                    "Year": st.column_config.NumberColumn("Year", format="%d", width=64),
                    "Title": st.column_config.TextColumn("Title", width="large"),
                    "Full text": st.column_config.TextColumn("Full text", width=70, help="Reviewed in full"),
                },
                height=min(38 * (len(df) + 1) + 4, 460),
            )
            if table.selection.rows:
                ss.focus_id = df.iloc[table.selection.rows[0]]["id"]
            st.caption(f"{len(shown)} of {len(papers)} papers shown. Tick the box at the left of a row to read it.")
            with st.container(horizontal=True):
                st.button(
                    f"Shortlist all {len(shown)} shown", icon=":material/star:",
                    on_click=shortlist_many, args=([p["id"] for p in shown],),
                )
                st.button("Clear shortlist", icon=":material/delete_sweep:", on_click=clear_shortlist,
                          disabled=not shortlist, type="tertiary")

    with right:
        paper = index.get(ss.get("focus_id"))
        with st.container(border=True, key="panel_detail"):
            if not paper:
                st.markdown("Select a paper in the table to read its abstract here.")
            else:
                chosen = paper["id"] in shortlist
                meta = f'{paper["year"] or "Year unknown"}, {paper["authors"] or "authors not reported"}'
                st.markdown(
                    f'<div class="rp-title"><span class="{"rp-hl" if chosen else ""}">{esc(paper["title"])}</span></div>'
                    f'<div class="rp-meta">{esc(meta)}</div>',
                    unsafe_allow_html=True,
                )
                with st.container(height=300, border=False):
                    if paper["abstract"]:
                        st.markdown(f'<div class="rp-abstract">{esc(paper["abstract"])}</div>', unsafe_allow_html=True)
                    else:
                        st.caption("No abstract was returned, so no methods or claims are inferred for this paper.")
                with st.container(horizontal=True):
                    st.button(
                        "Remove from shortlist" if chosen else "Add to shortlist",
                        type="secondary" if chosen else "primary",
                        icon=":material/star_border:" if chosen else ":material/star:",
                        on_click=toggle_shortlist, args=(paper["id"],), key="toggle_shortlist",
                    )
                    st.link_button("Open paper", paper["url"], icon=":material/open_in_new:")
                pid = paper["id"]
                if pid in reviews:
                    st.badge(f'Reviewed in full from {reviews[pid]["source"]}', icon=":material/menu_book:", color="green")
                    st.caption("Read the review in the Full-text reviews tab.")
                else:
                    if st.button("Review full paper", icon=":material/menu_book:", key="review_one",
                                 help="Fetch the full text and review its methods, results and limitations."):
                        if run_review(paper):
                            ss.flash = (f"Reviewed {pid} in full", ":material/menu_book:")
                            st.rerun()
                    if pid in ss.ft_errors:
                        st.warning(f"No open full text found. Tried: {ss.ft_errors[pid]}. If you have access through "
                                   "your library, download the PDF and upload it here.")
                    with st.expander("Upload the PDF instead", expanded=pid in ss.ft_errors):
                        upload = st.file_uploader("Paper PDF", type=["pdf"], key=f"upload_{ss.nonce}_{pid}",
                                                  label_visibility="collapsed")
                        if upload and st.button("Review uploaded PDF", type="primary", key="review_upload"):
                            if run_review(paper, upload.getvalue()):
                                ss.flash = (f"Reviewed {pid} from your PDF", ":material/menu_book:")
                                st.rerun()
                st.caption(
                    f'{paper["id"]}. DOI: {paper["doi"] or "not reported"}. '
                    f'Year from {paper.get("year_source", "unknown source")}.'
                )

# ---------- full-text reviews ----------
NOT_MATCHED = "Not matched"
with t_reviews:
    find_clicked = review_clicked = retry_clicked = False
    with st.container(border=True, key="panel_reviews_intro"):
        st.markdown("Valency Bond finds the papers. This step reads the **full text** of the ones you choose, "
                    "section by section, and pulls out methods, results, limitations and code links. Every item "
                    "keeps a quote, checked against the paper's own text.")
        if ss.read_mode == "full":
            st.caption("Reading depth is set to full papers, so the gap analysis and troubleshooting read any paper "
                       "they need in full automatically. Reviewing here first just does it ahead of time.")
        default_scope = "shortlisted" if shortlist else "all"
        scope_choice = st.segmented_control(
            "Papers to review", ["shortlisted", "all"], default=default_scope, required=True,
            key=f"review_scope_{ss.nonce}_{bool(shortlist)}",  # switches to the shortlist once there is one
            format_func=lambda s: f"Shortlisted ({len(shortlist)})" if s == "shortlisted" else f"All papers ({len(papers)})",
        ) or default_scope
        batch = [p for p in papers if p["id"] in shortlist] if scope_choice == "shortlisted" else papers
        to_find = [p for p in batch if p["id"] not in reviews and p["id"] not in ss.fulltexts and p["id"] not in ss.ft_errors]
        to_review = [p for p in batch if p["id"] not in reviews and p["id"] not in ss.ft_errors]
        if not batch:
            st.markdown("Nothing is shortlisted yet. Choose **All papers**, or shortlist papers in the Papers tab.")
        else:
            n_done = sum(p["id"] in reviews for p in batch)
            n_ready = sum(p["id"] in ss.fulltexts and p["id"] not in reviews for p in batch)
            n_missing = sum(p["id"] in ss.ft_errors and p["id"] not in reviews for p in batch)
            parts = [f"**{n_done} of {papers_count(len(batch))}** reviewed"]
            if n_ready:
                parts.append(f"{n_ready} with full text ready")
            if n_missing:
                parts.append(f"{n_missing} without an open full text")
            st.markdown(", ".join(parts) + ".")
            with st.container(horizontal=True):
                find_clicked = st.button(
                    f"Find full text for {papers_count(len(to_find))}" if to_find else "Full text already checked",
                    icon=":material/travel_explore:", disabled=not to_find,
                    help="Quick: downloads only, no AI. Shows which papers will need a PDF before you spend time reviewing.",
                )
                review_clicked = st.button(
                    f"Review {papers_count(len(to_review))}" if to_review else "Nothing left to review",
                    type="primary", icon=":material/menu_book:", disabled=not to_review,
                )
            notes = "Each paper is saved as soon as it's done, so stopping part-way keeps the finished ones. " \
                    "Clicking other buttons while a batch runs stops it; switching tabs is fine."
            if provider == "Local Ollama":
                notes = "Local models take a few minutes per paper; the OpenAI provider is much faster. " + notes
            st.caption(notes)
            if provider == "Local Ollama" and len(to_review) > 10:
                st.warning(f"Reviewing {len(to_review)} papers with a local model could take hours. "
                           "Consider starting with your shortlist.")
    if find_clicked:
        run_batch(to_find, review_it=False)
        st.rerun()
    if review_clicked:
        run_batch(to_review)
        st.rerun()

    last = ss.get("last_batch")
    if last:
        with st.expander(f"Last run: {last['label']}", expanded=bool(last["failed"]), icon=":material/history:"):
            st.markdown("\n".join(f"- {line}" for line in last["lines"]))

    if batch:
        with st.expander("Each paper's status", icon=":material/list_alt:"):
            st.dataframe(pd.DataFrame([{
                "ID": p["id"], "Title": p["title"],
                "Full text": ss.fulltexts[p["id"]]["source"] if p["id"] in ss.fulltexts
                else "Not found" if p["id"] in ss.ft_errors else "Not checked yet",
                "Review": "Done" if p["id"] in reviews else "",
            } for p in batch]), hide_index=True, width="stretch",
                column_config={"Title": st.column_config.TextColumn(width="large")})

    with st.container(border=True, key="panel_bulk_upload"):
        st.markdown("##### Add PDFs in bulk")
        missing_ids = [p["id"] for p in papers if p["id"] in ss.ft_errors and p["id"] not in reviews]
        st.caption((f"{papers_count(len(missing_ids))} {'has' if len(missing_ids) == 1 else 'have'} no open full text. "
                    if missing_ids else "")
                   + "If you have the PDFs, for example from your library, drop them all here at once. Each file is "
                     "matched to its paper by title, DOI or arXiv ID; check the matches, then review.")
        files = st.file_uploader("PDFs", type=["pdf"], accept_multiple_files=True,
                                 key=f"bulk_upload_{ss.nonce}_{ss.get('upload_round', 0)}", label_visibility="collapsed")
        if files:
            cache = ss.setdefault("pdf_match_cache", {})
            label_for = {p["id"]: f'{p["id"]}: {p["title"][:80]}' for p in papers}
            rows = []
            for f in files:
                cache_key = (f.name, f.size)
                if cache_key not in cache:
                    cache[cache_key] = match_pdf(f.getvalue(), f.name, papers)
                pid, how = cache[cache_key]
                note = how if pid else how.capitalize()
                if pid in reviews:
                    note += " (already reviewed; will be replaced)"
                rows.append({"File": f.name, "Paper": label_for.get(pid, NOT_MATCHED), "Matched by": note})
            edited = st.data_editor(
                pd.DataFrame(rows), key=f"bulk_matches_{ss.nonce}_{len(files)}", hide_index=True, width="stretch",
                disabled=["File", "Matched by"],
                column_config={"Paper": st.column_config.SelectboxColumn(
                    "Paper", options=[NOT_MATCHED] + list(label_for.values()), required=True, width="large")},
            )
            pairs, dupes = {}, set()
            for f, paper_label in zip(files, edited["Paper"]):
                if paper_label == NOT_MATCHED:
                    continue
                pid = paper_label.split(":", 1)[0]
                if pid in pairs:
                    dupes.add(pid)
                pairs[pid] = f
            unmatched = sum(label == NOT_MATCHED for label in edited["Paper"])
            if unmatched:
                st.caption(f"{papers_count(unmatched, 'file')} couldn't be matched. Pick the paper in the table, "
                           "or leave it out.")
            if dupes:
                st.warning(f"More than one file is matched to {', '.join(sorted(dupes))}; only the last one is used.")
            if st.button(f"Review {papers_count(len(pairs), 'matched PDF')}", type="primary",
                         icon=":material/menu_book:", disabled=not pairs, key="review_bulk_pdfs"):
                run_batch([index[pid] for pid in pairs], uploads={pid: f.getvalue() for pid, f in pairs.items()})
                ss.upload_round = ss.get("upload_round", 0) + 1  # empties the upload box once reviewed
                st.rerun()
        if missing_ids:
            retry_clicked = st.button(
                f"Search again for {papers_count(len(missing_ids))} without full text", type="tertiary",
                icon=":material/refresh:", help="For example after adding your Unpaywall email in the sidebar.",
            )
    if retry_clicked:
        for pid in missing_ids:
            ss.ft_errors.pop(pid, None)
        run_batch([index[pid] for pid in missing_ids], review_it=False)
        st.rerun()

    for p in [p for p in papers if p["id"] in reviews]:
        rv = reviews[p["id"]]
        with st.container(border=True, key=f"panel_review_{p['id']}"):
            st.markdown(f'<div class="rp-title">{esc(p["title"])}</div>'
                        f'<div class="rp-meta">{esc(p["id"])}, {p["year"] or "year unknown"}. Full text from '
                        f'{esc(rv["source"])}; {rv["passages_reviewed"]} of {rv["passages_total"]} passages reviewed '
                        f'with {esc(str(rv.get("model", "")))}.</div>', unsafe_allow_html=True)
            if rv.get("summary"):
                st.markdown(f'<div class="rp-abstract">{esc(rv["summary"])}</div>', unsafe_allow_html=True)
            if rv.get("relevance"):
                st.caption(f'Relevance to your question: {rv["relevance"]}')
            verified = sum(1 for k in ("methods", "results", "limitations", "future_work", "resources")
                           for x in rv.get(k, []) if x.get("verified"))
            total = sum(len(rv.get(k, [])) for k in ("methods", "results", "limitations", "future_work", "resources"))
            st.badge(f"{verified} of {total} extracted items have a quote found in the paper",
                     icon=":material/fact_check:", color="green" if total and verified == total else "orange")
            r_tab, m_tab, l_tab, c_tab, a_tab = st.tabs([
                f'Results ({len(rv.get("results", []))})', f'Methods ({len(rv.get("methods", []))})',
                f'Limitations and future work ({len(rv.get("limitations", [])) + len(rv.get("future_work", []))})',
                "Code and data", "Assessment",
            ])
            with r_tab:
                if rv.get("results"):
                    table = pd.DataFrame([{
                        "Finding": x.get("finding", ""),
                        "Value": " ".join(str(x.get(k)) for k in ("metric", "value") if x.get(k)),
                        "Condition": x.get("condition", ""),
                        "Section": x.get("section", ""),
                        "Quote found": "Yes" if x.get("verified") else "No, check it",
                        "Quote": x.get("quote", ""),
                    } for x in sorted(rv["results"], key=lambda x: not x.get("verified"))])
                    st.dataframe(table, hide_index=True, width="stretch",
                                 column_config={"Quote": st.column_config.TextColumn(width="large")})
                else:
                    st.caption("No results were extracted.")
            with m_tab:
                if rv.get("methods"):
                    render_items(rv["methods"], "item", ("detail",))
                else:
                    st.caption("No methods were extracted.")
            with l_tab:
                if rv.get("limitations"):
                    st.markdown("###### Limitations the authors state")
                    render_items(rv["limitations"], "item")
                if rv.get("future_work"):
                    st.markdown("###### Future work the authors state")
                    render_items(rv["future_work"], "item")
                if not rv.get("limitations") and not rv.get("future_work"):
                    st.caption("The paper's text didn't state limitations or future work in the passages reviewed.")
            with c_tab:
                repro = rv.get("reproducibility") or {}
                with st.container(horizontal=True, gap="small"):
                    for label in ("code", "data"):
                        value = str(repro.get(label, "unclear"))
                        st.badge(f"{label.capitalize()}: {value}", color="green" if value == "available" else "gray")
                if rv.get("links"):
                    st.markdown("Links found in the paper:")
                    for url in rv["links"]:
                        st.markdown(f"- [{url}]({url})")
                else:
                    st.caption("No code or data links were found in the text.")
                if rv.get("resources"):
                    render_items(rv["resources"], "name", ("kind",))
            with a_tab:
                st.caption("This is the review model's assessment, based only on the quote-checked extractions above.")
                for label, key in (("Strengths", "strengths"), ("Weaknesses", "weaknesses"), ("Open questions", "open_questions")):
                    if rv.get(key):
                        st.markdown(f"###### {label}")
                        st.markdown("\n".join(f"- {x}" for x in rv[key]))

# ---------- research gaps ----------
with t_gaps:
    shortlisted = [p for p in papers if p["id"] in shortlist]
    scope = (shortlisted or papers)[:ANALYSIS_CAP]
    n_read = len(scope)
    read_full = ss.read_mode == "full"
    in_full = [p for p in scope if p["id"] in reviews]
    no_open = [p for p in scope if p["id"] not in reviews and p["id"] in ss.ft_errors]
    to_read = [p for p in scope if p["id"] not in reviews and p["id"] not in ss.ft_errors] if read_full else []
    run_it = run_now = False
    with st.container(border=True, key="panel_analyze"):
        if shortlisted:
            note = f"The analysis covers your **{papers_count(len(shortlisted), 'shortlisted paper')}**"
            note += f" (the first {ANALYSIS_CAP})." if len(shortlisted) > ANALYSIS_CAP else "."
        else:
            note = (f"Nothing is shortlisted, so the analysis covers the **{papers_count(n_read, 'most recent paper')}**. "
                    "Shortlist papers in the Papers tab to focus it.")
        st.markdown(note)
        parts = []
        if read_full:
            if in_full:
                parts.append(f"{len(in_full)} already read in full")
            if to_read:
                parts.append(f"{len(to_read)} will be read in full first")
            if no_open:
                one = len(no_open) == 1
                parts.append(f"{len(no_open)} {'has' if one else 'have'} no open full text, so {'its' if one else 'their'} "
                             "abstract is used until you upload the PDF (Full-text reviews tab)")
            st.markdown("**Reading full papers:** " + "; ".join(parts) + ".")
            if to_read and provider == "Local Ollama":
                st.caption(f"Reading takes a few minutes per paper with a local model, so {len(to_read)} papers may "
                           "take a while. Finished papers are kept even if you stop.")
        else:
            st.markdown("**Reading abstracts only** (change this in the sidebar)"
                        + (f"; {len(in_full)} papers already reviewed in full are still used in full." if in_full else "."))
        st.caption("Gaps, resources and roadmap steps all come from this analysis, so they follow the same reading depth. "
                   "Every gap is a candidate until you check it against the wider literature.")
        with st.container(horizontal=True):
            if to_read:
                run_it = st.button(f"Read {papers_count(len(to_read))} in full, then analyze", type="primary",
                                   icon=":material/menu_book:")
                run_now = st.button("Analyze now with what's read so far", type="tertiary",
                                    help="Uses full text where it's already read and abstracts for the rest.")
            else:
                run_it = st.button("Run analysis again" if analysis else f"Analyze {papers_count(n_read)}",
                                   type="secondary" if analysis else "primary", icon=":material/psychology:")

    if run_it or run_now:
        ok = True
        if run_it and to_read:
            counts = run_batch(to_read)
            if counts["failed"] and not counts["ok"]:
                ok = False
                st.error("Reading the papers failed, so the analysis didn't run. Check the model, then try again."
                         + (OLLAMA_HINT if provider == "Local Ollama" else ""))
        if ok:
            ok = False
            current = {pid: rv for pid, rv in ss.reviews.items() if pid in index}
            n_full_now = sum(1 for p in scope if p["id"] in current)
            with st.status(f"Analyzing {papers_count(n_read)}: {n_full_now} in full, {n_read - n_full_now} by abstract",
                           expanded=True) as status:
                try:
                    st.write("Using full-text findings, stated limitations and future work where papers were read in "
                             "full, and abstracts for the rest; then drafting gaps, resources and steps")
                    old = ss.get("analysis") or {}
                    fresh_analysis = analyze(scope, question, openai_key, model, provider=provider, base_url=base_url,
                                             reviews=current)
                    fresh_analysis.update({"read_mode": ss.read_mode, "ran_at": now_str()})
                    if old.get("pursued"):
                        # Keep the plan, progress and lab work tied to the gaps you're pursuing.
                        fresh_analysis["draft_steps"] = fresh_analysis.get("steps", [])
                        for key in ("steps", "pursued", "plan_resources", "revisions"):
                            if old.get(key):
                                fresh_analysis[key] = old[key]
                        fresh_analysis["plan_kept_from"] = old.get("ran_at") or "an earlier analysis"
                    ss.analysis = fresh_analysis
                    ss.analysis_id += 1
                    log_event("analysis", f'Ran the gap analysis on {papers_count(n_read)} ({n_full_now} in full): '
                                          f'{len(ss.analysis.get("gaps", []))} gaps, {len(ss.analysis.get("steps", []))} steps')
                    if not ss.analysis.get("pursued"):
                        clear_keys("step_done_")
                    ss.flash = ("Analysis ready. Pick a gap to pursue and the roadmap and resources will follow it.",
                                ":material/check_circle:")
                    ok = True
                except Exception as exc:
                    status.update(label="Analysis failed", state="error", expanded=True)
                    st.error(f"{exc}" + (OLLAMA_HINT if provider == "Local Ollama" else ""))
        if ok:
            st.rerun()

    said = {"limitations": [], "future_work": [], "results": []}
    for p in scope:
        for key in said:
            for x in (reviews.get(p["id"]) or {}).get(key, []):
                if not x.get("verified"):
                    continue
                statement = x.get("item") or " ".join(str(x.get(k)) for k in ("finding", "metric", "value", "condition") if x.get(k))
                said[key].append({"Paper": p["id"], "Statement": statement, "Section": x.get("section", ""),
                                  "Quote": x.get("quote", "")})
    if any(said.values()):
        with st.container(border=True, key="panel_said"):
            st.markdown("#### What the papers say")
            st.caption(f"Gathered from the {papers_count(len([p for p in scope if p['id'] in reviews]))} read in full. "
                       "The gaps below build on these, especially the limitations and future work the authors state.")
            lim_tab, fut_tab, res_tab = st.tabs([f"Limitations they state ({len(said['limitations'])})",
                                                 f"Future work they call for ({len(said['future_work'])})",
                                                 f"Key results ({len(said['results'])})"])
            for container, key in ((lim_tab, "limitations"), (fut_tab, "future_work"), (res_tab, "results")):
                with container:
                    if said[key]:
                        st.dataframe(pd.DataFrame(said[key]), hide_index=True, width="stretch",
                                     column_config={"Statement": st.column_config.TextColumn(width="medium"),
                                                    "Quote": st.column_config.TextColumn(width="large")})
                    else:
                        st.caption("None stated in the passages read.")

    if not analysis:
        st.info("Run the analysis above to get gap candidates, each linked to the papers that support it. "
                "Then pick one to pursue: the roadmap, flowchart and resources will be built around it.")
    else:
        st.caption(analysis.get("coverage_note", ""))
        newer = [p["id"] for p in scope if p["id"] in reviews and p["id"] not in (analysis.get("full_text_ids") or [])]
        if newer:
            one = len(newer) == 1
            st.info(f"{papers_count(len(newer))} in this analysis's scope {'was' if one else 'were'} read in full after "
                    f"it ran. Run the analysis again to use {'its' if one else 'their'} full text.", icon=":material/update:")
        pursued = [i for i in range(1, len(gaps) + 1) if is_pursued(gaps[i - 1])]
        if gaps and not pursued:
            st.markdown("**Pick a gap to pursue.** The roadmap, flowchart and resources will then be built around it; "
                        "pursue more than one and each gets its own branch.")
        if pursued:
            css = "".join(f".st-key-panel_gap_{i} {{ box-shadow: inset 6px 0 0 {HIGHLIGHT}; }}" for i in pursued)
            st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
        statuses = list(dict.fromkeys(str(g.get("status", "hypothesis")) for g in gaps))
        visible = statuses
        if len(statuses) > 1:
            visible = st.pills("Show gaps marked as", statuses, selection_mode="multi", default=statuses,
                               key=f"gap_filter_{ss.analysis_id}", format_func=str.capitalize) or []
        if not gaps:
            st.warning("The model returned no gap candidates. Try shortlisting more papers with abstracts, or another model.")
        elif not visible:
            st.info("Pick at least one status above to see gaps.")
        for i, gap in enumerate(gaps, 1):
            status_text = str(gap.get("status", "hypothesis"))
            if status_text not in visible:
                continue
            with st.container(border=True, key=f"panel_gap_{i}"):
                st.markdown(f'#### {gap.get("title", "Gap candidate")}')
                cited = [pid for pid in gap.get("evidence_ids", []) if pid in index]
                n_cited_full = sum(1 for pid in cited if pid in reviews)
                with st.container(horizontal=True, gap="small"):
                    st.badge(status_text.capitalize(), color=status_color(status_text))
                    st.badge(f'{gap.get("confidence", "Unrated")} confidence'.capitalize(),
                             color=confidence_color(gap.get("confidence", "")))
                    if cited and n_cited_full == len(cited):
                        st.badge("Based on full text", icon=":material/menu_book:", color="green")
                    elif n_cited_full:
                        st.badge(f"{n_cited_full} of {len(cited)} cited papers read in full", icon=":material/menu_book:",
                                 color="orange")
                    elif cited:
                        st.badge("Based on abstracts", icon=":material/description:", color="gray")
                st.write(gap.get("explanation", ""))
                for q in gap.get("evidence_quotes", []):
                    st.markdown(f'<blockquote class="rp-quote">“{esc(str(q))}”</blockquote>', unsafe_allow_html=True)
                    origin = next((f"Quoted from the full text of {pid}" for pid in cited if pid in reviews and any(
                        quote_match(q, x.get("quote", "")) for k in ("limitations", "future_work", "results", "methods")
                        for x in reviews[pid].get(k, []) if x.get("verified"))), None) or next(
                        (f"Quoted from the abstract of {pid}" for pid in cited if quote_match(q, index[pid]["abstract"])), None)
                    if origin:
                        st.caption(origin)
                evidence_row(gap.get("evidence_ids", []), f"gap{i}")
                if is_pursued(gap):
                    n_steps = sum(1 for x in (analysis.get("steps") or []) if x.get("gap") == gap.get("title"))
                    with st.container(horizontal=True, vertical_alignment="center"):
                        st.badge(f"Pursuing: {papers_count(n_steps, 'step')} in the roadmap", icon=":material/flag:",
                                 color="yellow")
                        st.button("Stop pursuing", type="tertiary", key=f"stop_{i}", on_click=stop_pursuing,
                                  args=(gap.get("title"),), help="Removes this gap's branch from the roadmap.")
                elif st.button("Pursue this gap", icon=":material/flag:", key=f"pursue_btn_{i}",
                               help="Builds the roadmap, flowchart and resources around closing this gap."):
                    if pursue_gap(gap):
                        ss.flash = (f'Plan ready for "{gap.get("title")}". See 4. Roadmap and 5. Resources.',
                                    ":material/flag:")
                        st.rerun()

# ---------- resources ----------
with t_resources:
    for g in pursued_gaps():
        needed = (analysis.get("plan_resources") or {}).get(g["title"]) or []
        st.markdown(f'#### For the gap you\'re pursuing: {g["title"]}')
        if not needed:
            st.caption("The plan for this gap didn't list specific resources.")
        cols = st.columns(2, gap="medium")
        for j, res in enumerate(needed):
            with cols[j % 2], st.container(border=True, key=f'panel_gres_{abs(hash(g["title"]))}_{j}'):
                st.markdown(f'**{res.get("name", "Untitled")}**')
                st.caption(str(res.get("category") or "Resource"))
                st.write(res.get("why", ""))
                st.badge(str(res.get("status", "proposed")).capitalize(), color=status_color(res.get("status")))
                evidence_row(res.get("evidence_ids", []), f'gres{abs(hash(g["title"]))}_{j}')
    if pursued_gaps() and analysis:
        st.markdown("#### From the overall analysis")
    if not analysis:
        st.info("Run the analysis in Research gaps to list the materials, software, datasets and code the papers mention.")
    elif not resources:
        st.info("The analysis found no resources stated in these abstracts.")
    else:
        categories = list(dict.fromkeys(str(r.get("category") or "Other") for r in resources))
        picked = st.pills("Category", categories, selection_mode="multi", key=f"res_filter_{ss.analysis_id}")
        st.caption("Pick categories to narrow the list. With none picked, everything is shown.")
        for ci, cat in enumerate(picked or categories):
            items = [r for r in resources if str(r.get("category") or "Other") == cat]
            st.markdown(f"##### {cat} ({len(items)})")
            cols = st.columns(2, gap="medium")
            for j, res in enumerate(items):
                with cols[j % 2], st.container(border=True, key=f"panel_res_{ci}_{j}"):
                    st.markdown(f'**{res.get("name", "Untitled")}**')
                    st.write(res.get("why", ""))
                    st.badge(str(res.get("status", "unknown")).capitalize(), color=status_color(res.get("status")))
                    evidence_row(res.get("evidence_ids", []), f"res{ci}_{j}")

    linked = [(p, reviews[p["id"]]["links"]) for p in papers if p["id"] in reviews and reviews[p["id"]].get("links")]
    if linked:
        st.markdown("#### Code and data links found in the papers' full text")
        st.caption("Taken directly from the papers you read in full, so these are the authors' own links.")
        for p, links in linked:
            st.markdown(f'**{p["id"]}**: {p["title"]}  \n' + "  \n".join(f"[{u}]({u})" for u in links))

    st.divider()
    st.markdown("#### Find code and data")
    st.caption("GitHub results are live search hits. They aren't verified as any paper's official code.")
    with st.container(horizontal=True, vertical_alignment="bottom"):
        repo_default = (pursued_gaps()[0]["title"] if pursued_gaps() else question)[:110]
        repo_query = st.text_input("Repository keywords", value=repo_default,
                                   key=f"repo_query_{ss.nonce}_{len(pursued_gaps())}")
        find_repos = st.button("Search GitHub", icon=":material/code:")
    if find_repos:
        try:
            ss.repos = github_search(repo_query, github_token)
        except Exception as exc:
            st.error(f"GitHub search failed: {exc}. If you've hit the rate limit, add a GitHub token in the sidebar.")
    repos = ss.get("repos")
    if repos is not None:
        if not repos:
            st.info("No repositories matched. Try fewer or broader keywords.")
        cols = st.columns(2, gap="medium")
        for j, repo in enumerate(repos):
            with cols[j % 2], st.container(border=True, key=f"panel_repo_{j}"):
                st.markdown(f'**[{repo["name"]}]({repo["url"]})**')
                st.write(repo["description"] or "No description.")
                st.caption(f'{repo["stars"]:,} stars')
    q = quote(question)
    with st.container(horizontal=True):
        st.link_button("Search Zenodo", f"https://zenodo.org/search?q={q}", icon=":material/open_in_new:")
        st.link_button("Search Hugging Face datasets", f"https://huggingface.co/datasets?search={q}",
                       icon=":material/open_in_new:")
    st.caption("Prices, stock and instrument specifications aren't claimed without external checks.")

# ---------- roadmap ----------
with t_roadmap:
    if not steps:
        st.info("Run the analysis in Research gaps to draft a step-by-step roadmap you can tick off.")
    else:
        done_count = sum(done_flags)
        goals = pursued_gaps()
        if goals:
            with st.container(border=True, key="panel_goals"):
                st.markdown("#### Pursuing " + ("this gap" if len(goals) == 1 else f"{len(goals)} gaps"))
                for g in goals:
                    st.markdown(f'**{g["title"]}**' + (f'  \nClosed when: {g["success_criteria"]}' if g.get("success_criteria") else ""))
                    st.button("Stop pursuing", type="tertiary", key=f'stop_rm_{g["title"]}', on_click=stop_pursuing,
                              args=(g["title"],))
                if analysis.get("plan_kept_from"):
                    st.caption("The gaps were re-analyzed after this plan was made. The plan, progress and lab notes were "
                               "kept; pursue a new gap in 3. Research gaps to add it.")
        else:
            st.info("This is a draft plan for your whole question. Pick a gap to pursue in 3. Research gaps and the "
                    "roadmap, flowchart and resources will be rebuilt around closing it.", icon=":material/flag:")
        st.progress(done_count / len(steps), text=f"{done_count} of {len(steps)} steps done")

        shown = next((e for e in ss.lab_log if e["id"] == ss.get("show_diagnosis") and e.get("diagnosis")), None)
        if shown:
            with st.container(border=True, key="panel_diagnosis"):
                st.markdown(f'#### Diagnosis for step {shown["step_index"]}: {shown["step_title"]}')
                st.caption(f'From your lab note {shown["id"]}: "{shown["observed"][:200]}"')
                render_diagnosis(shown, f"rm{shown['id']}")
                st.button("Hide diagnosis", type="tertiary", key="hide_diag",
                          on_click=lambda: ss.pop("show_diagnosis", None))

        pending = ss.get("pending_revision")
        if pending:
            with st.container(border=True, key="panel_revision"):
                k = pending["from_step"]
                st.markdown(f"#### Proposed plan from step {k}")
                st.caption(f'Suggested after your lab note {pending["entry_id"]}. '
                           + (f"Steps 1 to {k - 1} stay as they are. " if k > 1 else "")
                           + "You can undo this later.")
                for j, new_step in enumerate(pending["steps"], k):
                    st.markdown(f'**{j}. {new_step.get("title", "Task")}**  \n{new_step.get("details", "")}')
                    evidence_row(new_step.get("evidence_ids", []), f"rev{j}")
                with st.container(horizontal=True):
                    st.button("Use this plan", type="primary", icon=":material/check:", on_click=apply_revision)
                    st.button("Keep the current plan", type="tertiary", on_click=dismiss_revision)

        # Log a result from the lab, then optionally pick up from there.
        with st.container(border=True, key="panel_log"):
            st.markdown("#### Log a result from the lab")
            if not ss.get("project_dir"):
                st.markdown("Save this work as a project first, so your lab log is kept and you can come back to it.")
                with st.container(horizontal=True, vertical_alignment="bottom"):
                    st.text_input("Project name", key="new_project_name_inline",
                                  placeholder=(question or "")[:50])
                    st.button("Save as project", icon=":material/save:", type="primary",
                              on_click=create_project, args=("new_project_name_inline",), key="save_project_inline")
            else:
                st.caption("Tell ResearchPilot what you did and what happened. If something went wrong, it picks up "
                           "from that step with your papers, full-text reviews and earlier attempts as context.")
                with st.form("lab_log_form", clear_on_submit=True, border=False):
                    step_i = st.selectbox("Step", list(range(1, len(steps) + 1)), index=next_step_index() - 1,
                                          format_func=lambda i: f'{i}. {steps[i - 1].get("title", "Task")}')
                    c1, c2 = st.columns(2)
                    did = c1.text_area("What did you do?", height=110,
                                       placeholder="Samples, settings, parameters, what you changed")
                    observed = c2.text_area("What happened?", height=110,
                                            placeholder="Results, numbers, errors, what looked wrong")
                    outcome = st.segmented_control("Outcome", OUTCOMES, default="Problem", key="log_outcome")
                    files = st.file_uploader("Attach data, images or notes (optional)", accept_multiple_files=True,
                                             key="log_files")
                    with st.container(horizontal=True):
                        save_fix = st.form_submit_button("Save and troubleshoot", type="primary",
                                                         icon=":material/build:")
                        save_only = st.form_submit_button("Save to lab log", icon=":material/save:")
                if save_fix or save_only:
                    if not (did.strip() or observed.strip()):
                        st.error("Describe what you did or what happened, then save again.")
                    else:
                        entry = add_log_entry(step_i, did, observed, outcome, files)
                        if save_fix and entry["outcome"] != "Worked":
                            if run_troubleshoot(entry):
                                ss.flash = ("Diagnosis ready. See the likely causes and the proposed plan below.",
                                            ":material/build:")
                                st.rerun()
                        else:
                            ss.flash = (f'Saved to the lab log for step {step_i}.', ":material/save:")
                            st.rerun()

        done_css = "".join(
            f".st-key-panel_step_{i} {{ box-shadow: inset 6px 0 0 {HIGHLIGHT}; }}"
            for i, flag in enumerate(done_flags, 1) if flag
        )
        if done_css:
            st.markdown(f"<style>{done_css}</style>", unsafe_allow_html=True)
        def render_steps():
            group = None
            for i, step in enumerate(steps, 1):
                if goals and step.get("gap") != group:
                    group = step.get("gap")
                    st.markdown("##### Shared setup" if group == SHARED else f"##### Gap: {group}")
                with st.container(border=True, key=f"panel_step_{i}"):
                    st.checkbox(f'**{i}. {step.get("title", "Task")}**', key=f"step_done_{i}",
                                on_change=step_toggled, args=(i, step.get("title", "Task")))
                    st.write(step.get("details", ""))
                    with st.container(horizontal=True, gap="small"):
                        st.badge(str(step.get("status", "proposed")).capitalize(), color=status_color(step.get("status")))
                        if step.get("added_after"):
                            st.badge(f'Added after lab note {step["added_after"]}', icon=":material/build:", color="violet")
                    evidence_row(step.get("evidence_ids", []), f"step{i}")
                    notes = [e for e in ss.lab_log if e["step_index"] == i and e["step_title"] == step.get("title", "Task")]
                    if notes:
                        last = notes[-1]
                        st.caption(f'{papers_count(len(notes), "lab note")}. Latest, {last["created"][:10]} '
                                   f'({last["outcome"].lower()}): "{last["observed"][:140].rstrip()}" '
                                   "Full notes are in the Lab notebook tab.")

        def render_flow():
            with st.container(border=True, key="panel_flow"):
                st.graphviz_chart(roadmap_dot(steps, done_flags, goals), width="content" if goals else "stretch")
                st.caption(("Shared setup comes first, then one branch per gap you're pursuing, ending in what would "
                            "show the gap is closed. " if goals else "")
                           + "Solid boxes cite retrieved papers; dashed boxes are proposals with no supporting paper. "
                           "Highlighted boxes are done.")
                st.download_button("Download flowchart (.mmd)", mermaid_source(steps, pursued_gaps()), "research_flowchart.mmd",
                                   mime="text/plain", icon=":material/download:", on_click="ignore")
            revisions = analysis.get("revisions") or []
            if revisions:
                last_rev = revisions[-1]
                st.caption(f'The roadmap was revised {papers_count(len(revisions), "time")}; most recently from '
                           f'step {last_rev["from_step"]} after lab note {last_rev["entry_id"]}.')
                st.button("Undo the last revision", icon=":material/undo:", type="tertiary", on_click=undo_revision)

        if goals:  # branches need room: flowchart across the page, steps below it
            render_flow()
            render_steps()
        else:
            left, right = st.columns([3, 2], gap="large")
            with left:
                render_steps()
            with right:
                render_flow()

# ---------- lab notebook ----------
with t_notebook:
    if not ss.get("project_dir"):
        st.info("Save this work as a project (in the sidebar) to keep a lab notebook. You can then log results "
                "against roadmap steps, close the app, and pick up later where you stopped.")
    if ss.lab_log:
        st.markdown("#### Lab notes")
        latest = ss.lab_log[-1]["id"]
        for e in reversed(ss.lab_log):
            with st.container(border=True, key=f"panel_note_{e['id']}"):
                st.markdown(f'**{e["id"]}. Step {e["step_index"]}: {e["step_title"]}**')
                with st.container(horizontal=True, gap="small"):
                    st.badge(e["outcome"], color=status_color(e["outcome"]))
                    if e.get("gap"):
                        st.badge(f'For gap: {e["gap"][:60]}', icon=":material/flag:", color="yellow")
                    st.badge(f'{e["created"][:10]} {e["created"][11:16]}', icon=":material/schedule:", color="gray")
                if e.get("did"):
                    st.markdown(f'**What was done:** {e["did"]}')
                if e.get("observed"):
                    st.markdown(f'**What happened:** {e["observed"]}')
                if e.get("attachments"):
                    st.caption("Attached: " + ", ".join(a["name"] for a in e["attachments"])
                               + f'. Saved in {Path(ss.get("project_dir") or "").name}/files.')
                if e.get("diagnosis"):
                    with st.expander("Diagnosis", expanded=e["id"] == latest, icon=":material/build:"):
                        render_diagnosis(e, f"nb{e['id']}")
                elif e["outcome"] != "Worked" and ss.get("project_dir"):
                    if st.button("Troubleshoot this", icon=":material/build:", key=f"ts_{e['id']}"):
                        if run_troubleshoot(e):
                            ss.flash = ("Diagnosis ready. The proposed plan is in the Roadmap tab.", ":material/build:")
                            st.rerun()
    elif ss.get("project_dir"):
        st.markdown("No lab notes yet. Log your first result in the Roadmap tab.")
    if ss.timeline:
        st.markdown("#### Project timeline")
        icons = {"search": "search", "review": "menu_book", "analysis": "lightbulb", "plan": "route",
                 "lab": "edit_note", "diagnosis": "build", "project": "folder"}
        st.markdown("\n".join(
            f'- :material/{icons.get(ev["kind"], "circle")}: {ev["time"][:10]} {ev["time"][11:16]}: {ev["text"]}'
            for ev in reversed(ss.timeline[-200:])))

# ---------- export ----------
with t_export:
    extra = []
    if pursued_gaps():
        extra += ["", "## Gaps being pursued"]
        for g in pursued_gaps():
            extra += [f'### {g["title"]}', g.get("explanation", ""), f'- Closed when: {g.get("success_criteria") or "not stated"}']
            needed = (analysis.get("plan_resources") or {}).get(g["title"]) or []
            if needed:
                extra.append("- Resources: " + "; ".join(f'{r.get("name")} ({r.get("status", "proposed")})' for r in needed))
    if steps:
        extra += ["", "## Roadmap progress" + ("" if pursued_gaps() else " (draft plan, not tied to a gap)")]
        group = None
        for i, s_ in enumerate(steps, 1):
            if pursued_gaps() and s_.get("gap") != group:
                group = s_.get("gap")
                extra.append(f'**{"Shared setup" if group == SHARED else "Gap: " + str(group)}**')
            extra.append(f'- [{"x" if done_flags[i - 1] else " "}] {i}. {s_.get("title", "Task")}')
    if shortlist:
        extra += ["", "## Shortlisted papers"] + [f'- [{p["id"]}] {p["title"]}' for p in papers if p["id"] in shortlist]
    if reviews:
        extra += ["", "## Full-text reviews", "Items are quoted from each paper; only quotes found in the paper's text are listed."]
        extra += ["\n" + review_markdown(reviews[p["id"]]) for p in papers if p["id"] in reviews]
    if ss.lab_log:
        extra += ["", "## Lab notebook"]
        for e in ss.lab_log:
            extra += ["", f'### {e["id"]}: step {e["step_index"]}, {e["step_title"]} ({e["outcome"]}, {e["created"][:10]})',
                      f'- What was done: {e.get("did") or "not recorded"}', f'- What happened: {e.get("observed") or "not recorded"}']
            d = e.get("diagnosis") or {}
            if d.get("summary"):
                extra.append(f'- Diagnosis: {d["summary"]}')
            for c in d.get("likely_causes", []):
                ids = ", ".join(f"[{i}]" for i in c.get("evidence_ids", [])) or "no supporting paper"
                extra.append(f'  - Possible cause ({c.get("status", "hypothesis")}): {c.get("cause", "")}; evidence: {ids}')
    revisions = (analysis or {}).get("revisions") or []
    if revisions:
        extra += ["", "## Roadmap revisions"] + [
            f'- {r["time"][:10]}: revised from step {r["from_step"]} after lab note {r["entry_id"]}' for r in revisions]
    base = markdown_report(question, papers, analysis or None)
    head, sep, tail = base.rpartition("\n**Caution:**")
    report = (head + "\n".join(extra) + "\n" + sep + tail) if sep else base + "\n".join(extra)

    st.markdown("#### Download your research packet")
    st.caption("The report includes your shortlist, the gaps you marked to pursue, roadmap progress, and your lab notebook. "
               "The project file opens in ResearchPilot on any computer, with everything needed to pick up again.")
    with st.container(horizontal=True):
        st.download_button("Report (.md)", report, "researchpilot_report.md", mime="text/markdown",
                           type="primary", icon=":material/description:", on_click="ignore")
        st.download_button("Paper metadata (.json)", json.dumps(papers, indent=2, default=str),
                           "researchpilot_papers.json", mime="application/json", icon=":material/data_object:",
                           on_click="ignore")
        bib = pd.DataFrame([{k: p[k] for k in ("id", "title", "year", "authors", "doi", "url", "abstract")} for p in papers])
        if reviews:
            st.download_button("Full-text reviews (.json)", json.dumps(reviews, indent=2, default=str),
                               "researchpilot_reviews.json", mime="application/json", icon=":material/menu_book:",
                               on_click="ignore")
        st.download_button("Bibliography (.csv)", bib.to_csv(index=False), "researchpilot_papers.csv",
                           mime="text/csv", icon=":material/table:", on_click="ignore")
        if steps:
            st.download_button("Flowchart (.mmd)", mermaid_source(steps, pursued_gaps()), "research_flowchart.mmd",
                               mime="text/plain", icon=":material/account_tree:", on_click="ignore", key="mmd_export")
        st.download_button("Project file (.json)", json.dumps({**project_snapshot(), "name": ss.get("project_name") or question[:50]},
                                                              indent=1, default=str),
                           "researchpilot_project.json", mime="application/json", icon=":material/folder_zip:",
                           on_click="ignore", key="project_export")
    with st.container(height=480, border=True, key="panel_preview"):
        st.markdown(report)
    st.caption("Metadata-only records don't establish exact experimental methods. The app doesn't buy anything, "
               "run experiments, or claim that a gap is novel.")

# ---------- autosave ----------
if ss.get("project_dir"):
    _snapshot = json.dumps(project_snapshot(), sort_keys=True, default=str)
    if hash(_snapshot) != ss.get("_saved_hash"):
        try:
            ss.saved_at = save_project(ss.project_dir, json.loads(_snapshot))
            ss._saved_hash = hash(_snapshot)
        except OSError as exc:
            st.toast(f"Couldn't save the project: {exc}", icon=":material/error:")
