# ResearchPilot: From Literature to Laboratory

A small, local Streamlit hackathon prototype for scientific paper discovery through Valency Bond, a publication timeline, and evidence-limited planning.

## Install on macOS using Conda

```bash
conda create -n researchpilot python=3.11 -y
conda activate researchpilot
cd ~/Downloads/researchpilot
python -m pip install -r requirements.txt
python -m pip check
python -m streamlit run app.py
```

If you already have a clean `researchpilot` environment, do not recreate it; simply install requirements.

## Credentials

- Copy the entire Valency API token from the Valency app's **Settings > API Token**. In the Streamlit sidebar paste the raw `tok_...` token, **not** `Bearer tok_...` and not the server URL.
- Endpoint defaults to `https://mcp.valency.io/`; opening it in a browser without an Authorization header normally produces `Missing Authorization header`.
- Click **Test Valency connection** to initialize Streamable HTTP MCP, list tools, and call `list_sources` if offered. Then click **Search Valency Bond**.
- **No OpenAI key is needed with Local Ollama**, the default analysis provider. An OpenAI API key is only needed if you switch the provider to OpenAI API; paid API access is separate from ChatGPT subscriptions.
- A GitHub token is optional; rate limits apply without one.
- An email for Unpaywall is optional. With it, the full-text review can also find legal open-access copies of journal papers (set it in the sidebar or as `UNPAYWALL_EMAIL` in `.env`).
- Do not screenshot or publish your token. Rotate any token exposed publicly.

## How the steps connect

The tabs are numbered because each one feeds the next, and a **Next** line above them always says what to do now.

1. **Papers**: Valency Bond finds papers; you shortlist the ones that matter.
2. **Full-text reviews**: papers are read in full (automatically, with Reading depth set to Full papers).
3. **Research gaps**: **What the papers say** gathers the limitations, future work and key results from the papers read in full; the gaps build on them. **Pursue this gap** builds the plan around it, reading its papers in full first if needed.
4. **Roadmap**: until you pursue a gap it's a draft for your whole question. Once you do, it becomes shared setup steps plus a branch for each gap you're pursuing, and each branch ends in what would show the gap is closed. Pursue another gap and it gets its own branch; **Stop pursuing** removes only that branch. Steps you've ticked stay ticked.
5. **Resources**: what the gaps you're pursuing need comes first, then the overall analysis, then code and data links from the papers' own text. GitHub search starts from the gap you're pursuing.
6. **Lab notebook**: results logged against roadmap steps (and the gap they served); troubleshooting revises only that branch of the plan.

Re-running the gap analysis keeps the plan, progress and lab notes for the gaps you're pursuing. **Export** and the project file follow the same pipeline.

## Features

- Scientific literature query with date bounds through the live Bond MCP server, with example questions to start from
- Clickable publication timeline: click a year's bar to filter the papers (Shift-click to add years)
- Paper explorer: filter, sort, tick a row to read its abstract in the side panel, and shortlist papers
- **Full-paper review**: for papers you choose, fetch the full text and review it section by section (methods, results, limitations, future work, code and data links), with every item quoted and the quote checked against the paper
- AI gap analysis that reads only your shortlisted papers, using full-text reviews where available (or the most recent ones if none are shortlisted), constrained by retrieved metadata/abstracts
- Cited evidence opens in a popover with the paper's abstract, so you can check a claim without leaving the page
- Projects you can close and reopen, a lab notebook, and troubleshooting that picks up from the step where an experiment failed
- Pursue a gap and the roadmap, branching flowchart and resources are built around it; tick off steps as you go (solid boxes cite papers; dashed boxes are unsupported proposals)
- GitHub live repository discovery (not proof of author affiliation)
- Markdown report (including your shortlist, pursued gaps and roadmap progress), JSON metadata, CSV bibliography, Mermaid flowchart

## Reading depth: one setting for the whole app

In the sidebar, **Reading depth** is either **Full papers** (the default) or **Abstracts only**. It's saved with your project, so you set it once.

With **Full papers**, every step that reads literature reads it in full:

- **Research gaps** (and the resources and roadmap steps that come from the same analysis): any paper in the analysis that hasn't been read yet is fetched and reviewed in full first (**Read N papers in full, then analyze**). The analysis then works from each paper's stated limitations, future work, quantitative results and methods, not just its abstract. Each gap is labelled **Based on full text**, **Based on abstracts**, or how many of its papers were read in full, and every quote says whether it comes from the full text or the abstract.
- If papers are read in full after an analysis ran, the Research gaps tab tells you and offers to run it again.
- **Resources** also lists the code and data links found in the papers' own text.
- **Troubleshooting** reads the papers the failed step relies on, and up to three newly found papers, in full before diagnosing.
- Papers with no open full text use their abstract until you upload the PDF; the app always says which ones.

**Abstracts only** is faster: papers are read by abstract, and full-text reviews you've already done are still used.

## Full-paper review

Valency Bond finds papers and returns their metadata and abstracts. Full text is handled in bulk in the **Full-text reviews** tab:

- Choose **Shortlisted** or **All papers**.
- **Find full text** (quick, no AI) downloads what's openly available and shows which papers will need a PDF.
- **Review** reads every paper in the batch with one progress bar. Each paper is saved as soon as it's done, so stopping part-way keeps the finished ones. Switching tabs while it runs is fine; clicking other buttons stops the batch.
- **Add PDFs in bulk**: drop all your PDFs at once. Each is matched to its paper by title, DOI or arXiv ID; fix any match in the table, then review them together.

You can still review a single paper from its detail panel (**Review full paper**). The text comes from, in order:

1. a PDF you upload (for example, downloaded through your library's access),
2. the paper's arXiv HTML version, then its arXiv PDF,
3. a legal open-access copy found through Unpaywall (only if you've added your email).

If none is found, the app says what it tried and offers a PDF upload. Text is fetched only for the papers you pick, and stays on your computer.

The review skips the introduction, related work and references, splits the methods, results, discussion, limitations and conclusion sections into passages, and extracts items from each with a word-for-word quote. Each quote is checked against the paper's text. A quote counts if it appears word for word, or closely (at least 85% of its words in order, which tolerates a model's small copying slips) with every number exactly as in the paper, so a changed value never passes. Items whose quote isn't found are flagged "check it" and aren't used in the gap analysis. A final step writes a summary, relevance, strengths, weaknesses and a reproducibility check. The gap analysis then uses these quote-checked items for reviewed papers and abstracts for the rest.

Speed: with local Ollama, expect a few minutes per paper (passages are kept small so they fit the model's context; Qwen3's slow "thinking" mode is switched off for reviews). The OpenAI provider is much faster. For better local results, raise Ollama's context length in the Ollama app's settings, or start it with `OLLAMA_CONTEXT_LENGTH=32768 ollama serve`.

Respect each paper's license: uploaded PDFs and downloaded text are only processed locally, and the exported reviews contain short quotes, not full text.

## Projects, lab notebook and picking up where you stopped

Research rarely works the first time. ResearchPilot keeps a project so you can come back after an experiment and continue from that point, instead of starting over.

1. **Save as project** (sidebar, or the Roadmap tab). Everything is kept: question, papers, shortlist, full-text reviews, gaps, roadmap and progress. From then on it saves automatically to `projects/<name>/` next to `app.py`. Nothing leaves your computer.
2. **Run the experiment**, then in the Roadmap tab **Log a result from the lab**: pick the step, say what you did and what happened, choose Worked, Partly worked or Problem, and attach data, images or notes if you like. "Worked" ticks the step off.
3. **Save and troubleshoot** picks up from that step. It gives the AI only the context that matters:
   - the step, and earlier attempts at that point in the plan (with what was suggested then, so it doesn't repeat itself),
   - quote-checked limitations, results and methods from your full-text reviews that match the problem,
   - your attached data files (the first part of CSV, TXT and similar files),
   - new papers about this specific problem, found with a targeted Valency Bond search and added to the project.

   It returns likely causes (each marked literature-backed or hypothesis, with clickable evidence), quick checks to tell the causes apart, and a **proposed plan from that step onward**. Click **Use this plan** to apply it; earlier steps stay as they are, and **Undo the last revision** restores the old plan.
4. **Close the project** and come back any time: open it from the sidebar and a *Welcome back* panel shows where you stopped, your last lab note and the next step.

The **Lab notebook** tab lists every lab note with its diagnosis, plus a timeline of the whole project. **Export** adds the lab notebook and roadmap revisions to the report, and offers a **project file (.json)** you can open in ResearchPilot on another computer (sidebar: *Or open a project file*).

Diagnoses are suggestions to test, not instructions. Check anything involving hazardous materials or equipment with your lab's safety procedures.

## Limits and verification

Bond advertises its MCP tools dynamically; this project introspects their published input schemas rather than depending on one fixed version. If Bond changes its argument names, **Test Valency connection** will show the tools and parameter names so the integration can be adjusted. Passing dates is best-effort and is independently checked by year after retrieval. Paper extraction is dependent on the tool response format. The AI component reads full text only for papers you review in full (see Full-paper review); all other papers are read by title and abstract. It does not certify that a research gap is novel, provide pricing, or imply third-party GitHub repositories are official author code. Claims with no support must remain labeled as suggestions.

## Troubleshooting

- `Missing Authorization header` in a browser: normal for a URL opened without token headers.
- `Tool object has no attribute inputSchema`: old code; this release uses `input_schema` with compatibility fallback.
- `unexpected keyword argument` from a Streamlit function: your Streamlit is older than 1.60. Run `python -m pip install -U -r requirements.txt`.
- The app looks plain (default fonts and colors): you started Streamlit from another folder, so it didn't find `.streamlit/config.toml`. `cd` into the project folder first.
- `ExceptionGroup`: the Streamlit error box now shows the nested exception; share the text **without tokens**.
- To inspect SDK properties: `python -c "from mcp.types import Tool; print(Tool.model_fields.keys())"`

## Included files

- `app.py`: Streamlit UI
- `bond.py`: Valency MCP connection, introspection, source test, and search
- `research.py`: normalization, constrained analysis, GitHub lookup, exports
- `fulltext.py`: finds and downloads full text (arXiv, Unpaywall, uploaded PDFs) and splits it into sections
- `review.py`: section-by-section full-paper review
- `quotes.py`: checks that quotes really come from the paper (exact or close match; numbers must match)
- `project.py`: saving, opening and autosaving projects; lab-log attachments
- `troubleshoot.py`: picks up from a step where an experiment hit a problem
- `projects/`: created when you save your first project (your local data; keep it out of version control)
- `requirements.txt`: Python dependencies (Streamlit 1.60 or newer)
- `.streamlit/config.toml`: the app's theme (colors and fonts). Keep this folder next to `app.py`, and start the app from that folder so Streamlit finds it
- `.env.example`: optional local environment variables (never commit secrets)


## Local LLM (no paid model API)

Install [Ollama](https://ollama.com/download/mac) and run its desktop app, then:

```bash
ollama pull qwen3:8b
ollama list
conda activate researchpilot
cd ~/Downloads/researchpilot
python -m pip install -r requirements.txt
python -m streamlit run app.py
```

In the sidebar select **Local Ollama**, set model to `qwen3:8b`, leave API base URL `http://localhost:11434/v1`, and enter only your Valency Bond token. Smaller-memory Macs can use `ollama pull qwen3:4b` and model `qwen3:4b`. Local inference uses your own CPU/GPU and consumes electricity but has no per-request model API fee. The agent uses quote-checked full-text reviews where you have them, extracts observations from the remaining abstracts in batches of up to five papers, then synthesizes candidates and validates the cited paper identifiers. It does not prove research novelty or accuracy.

## Year metadata
The year is taken from Bond's explicit metadata fields first, then arXiv IDs / URL date path where available, and optionally Crossref DOI metadata. arXiv IDs encode submission year, not necessarily journal publication year. Other URL years are unverified. Year unknown means no reliable year could be parsed.
