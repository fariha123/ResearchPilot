"""Find, download and split the full text of papers that Valency Bond discovers.

Valency Bond returns metadata and abstracts. For papers you choose to review in full, this module
gets the text from, in order: a PDF you upload (for example through your library's access),
the arXiv HTML version, the arXiv PDF, or a legal open-access copy found through Unpaywall.
The text is split into sections so the review can focus on methods, results and limitations.
"""
import re
import time
import unicodedata
from urllib.parse import quote

import requests

USER_AGENT = {"User-Agent": "ResearchPilot/2.0 (open-source research assistant; full-text review)"}
ARXIV_NEW = r"(\d{4}\.\d{4,5})(?:v\d+)?"
ARXIV_OLD = r"([a-z\-]+(?:\.[A-Z]{2})?/\d{7})(?:v\d+)?"
CODE_LINK = re.compile(
    r"https?://(?:www\.)?(?:github\.com|gitlab\.com|bitbucket\.org|zenodo\.org|figshare\.com|"
    r"huggingface\.co|osf\.io|materialsproject\.org|nomad-lab\.eu|doi\.org/10\.5281)[^\s)\]}>,;\"']*",
    re.I,
)
MIN_TEXT_CHARS = 2000  # less than this usually means a scanned PDF or an error page

# Order matters: the first matching pattern decides a section's kind.
SECTION_KINDS = [
    ("references", r"\breferences\b|bibliograph|acknowledg|funding|author contributions|competing interest|"
                   r"conflicts? of interest|declaration"),
    ("availability", r"availability|reproducibility|code and data"),
    ("limitations", r"limitation|threats to validity|caveat|failure (?:case|mode)"),
    ("conclusion", r"conclusion|concluding|summary|outlook|future work|perspective"),
    ("abstract", r"^abstract$"),
    ("introduction", r"introduction|^background$|motivation"),
    ("related", r"related work|prior work|previous work|literature review|state of the art"),
    ("results", r"result|evaluation|^experiments$|performance|benchmark|case stud|validation"),
    ("discussion", r"discussion"),
    ("methods", r"method|approach|experimental|materials|model|framework|data|simulation|computational|"
                r"implementation|training|architecture|setup|procedure|synthesis|characteri[sz]ation"),
]
REFERENCE_STOP = re.compile(r"^(?:references|bibliography|literature cited)$", re.I)


class FullTextUnavailable(Exception):
    """Raised when no full text could be found; the message says what was tried."""


def classify(title):
    t = re.sub(r"\s+", " ", str(title or "")).strip().lower()
    for kind, pattern in SECTION_KINDS:
        if re.search(pattern, t):
            return kind
    return "body"  # topical headings like "Phase identification network" are core content


def _norm_space(text):
    return re.sub(r"\s+", " ", text or "").strip()


# ---------- identifiers ----------
def find_arxiv_id(paper):
    raw = paper.get("raw") or {}
    candidates = [raw.get(k) for k in ("arxiv_id", "arxiv", "arXiv", "paper_id", "id", "external_id", "identifier")]
    candidates += [paper.get("url"), raw.get("url"), raw.get("pdf_url"), raw.get("arxiv_url"), paper.get("doi"), raw.get("doi")]
    for value in candidates:
        if not value:
            continue
        s = str(value).strip()
        m = (re.search(r"arxiv\.org/(?:abs|pdf|html)/" + ARXIV_NEW, s, re.I)
             or re.search(r"10\.48550/arxiv\." + ARXIV_NEW, s, re.I)
             or re.fullmatch(r"(?:arxiv:)?" + ARXIV_NEW, s, re.I))
        if m:
            return m.group(1)
        m = re.search(r"arxiv\.org/(?:abs|pdf)/" + ARXIV_OLD, s, re.I) or re.fullmatch(r"(?:arxiv:)?" + ARXIV_OLD, s, re.I)
        if m:
            return m.group(1)
    return None


# ---------- parsing ----------
def parse_arxiv_html(html_text):
    """Split an arXiv (LaTeXML) HTML paper into sections. Returns (sections, links)."""
    from bs4 import BeautifulSoup

    soup = BeautifulSoup(html_text, "html.parser")
    article = soup.select_one("article.ltx_document") or soup.select_one("article") or soup.body or soup
    for math in article.find_all("math"):
        math.replace_with(" " + (math.get("alttext") or "") + " ")
    links = sorted({a["href"].rstrip(".") for a in article.find_all("a", href=True) if CODE_LINK.match(a["href"])})
    for junk in article.select(".ltx_bibliography, .ltx_page_footer, .ltx_page_header, script, style, nav, button"):
        junk.decompose()

    sections = []
    abstract = article.select_one(".ltx_abstract")
    if abstract:
        text = re.sub(r"^\s*abstract[.:]?\s*", "", _norm_space(abstract.get_text(" ")), flags=re.I)
        sections.append({"title": "Abstract", "kind": "abstract", "text": text})
        abstract.decompose()

    tops = [s for s in article.find_all("section")
            if {"ltx_section", "ltx_appendix"} & set(s.get("class", []))]
    if not tops:  # fall back to h2 headings for non-LaTeXML pages
        tops = [h.parent for h in article.find_all("h2")]
    for sec in tops:
        heading = sec.find(re.compile(r"^h[1-6]$"))
        title = "Section"
        if heading:
            for tag in heading.select(".ltx_tag"):
                tag.decompose()
            title = _norm_space(heading.get_text(" ")) or title
            heading.decompose()
        text = _norm_space(sec.get_text(" "))
        if not text:
            continue
        kind = "appendix" if "ltx_appendix" in sec.get("class", []) else classify(title)
        sections.append({"title": title, "kind": kind, "text": text})
    return sections, links


def _roman(s):
    values = {"I": 1, "V": 5, "X": 10}
    total = 0
    for i, ch in enumerate(s):
        v = values[ch]
        total += -v if i + 1 < len(s) and values[s[i + 1]] > v else v
    return total


NUMBERED = re.compile(r"^\s*(\d{1,2}|[IVX]{1,5})\.?\s+([A-Z][A-Za-z0-9 ,:&()/\-]{2,80})$")
NAMED = re.compile(
    r"^\s*(abstract|introduction|background|related work|methods?|materials and methods|methodology|"
    r"experimental(?: section| details| methods)?|results(?: and discussion)?|discussion|conclusions?|"
    r"summary|limitations|outlook|data availability|code availability|data and code availability|"
    r"acknowledge?ments?|references|bibliography)\s*:?\s*$", re.I)


def parse_pdf(data):
    """Extract text from a PDF and split it at top-level headings. Returns (sections, links)."""
    import pymupdf

    doc = pymupdf.open(stream=data, filetype="pdf")
    raw = "\n".join(page.get_text("text") for page in doc)
    links = set()
    for page in doc:
        for link in page.get_links():
            uri = link.get("uri") or ""
            if CODE_LINK.match(uri):
                links.add(uri.rstrip("."))
    doc.close()
    text = unicodedata.normalize("NFKC", raw)
    text = re.sub(r"(\w)-\n(\w)", r"\1\2", text)  # rejoin hyphenated line breaks
    links |= {m.group(0).rstrip(".") for m in CODE_LINK.finditer(text)}

    sections, current, body, expected = [], {"title": "Front matter", "kind": "front"}, [], None
    for line in text.split("\n"):
        stripped = line.strip()
        title = None
        m = NUMBERED.match(stripped)
        if m and not stripped.endswith(".") and len(m.group(2).split()) <= 10:
            num = int(m.group(1)) if m.group(1).isdigit() else _roman(m.group(1))
            if (expected is None and num in (1, 2)) or num == expected:
                title, expected = m.group(2).strip(), num + 1
        elif NAMED.match(stripped):
            title = stripped.rstrip(":").strip()
        if title is None:
            body.append(stripped)
            continue
        if body:
            sections.append({**current, "text": _norm_space(" ".join(body))})
        if REFERENCE_STOP.match(title):
            body = []
            current = None
            break
        current, body = {"title": title.title() if title.isupper() else title, "kind": classify(title)}, []
    if current is not None and body:
        sections.append({**current, "text": _norm_space(" ".join(body))})

    headed = [s for s in sections if s["kind"] != "front"]
    if len(headed) < 2:  # no usable headings: keep the whole text, minus the reference list
        whole = re.split(r"\n\s*(?:References|REFERENCES|Bibliography)\s*\n", text)[0]
        sections = [{"title": "Full text", "kind": "body", "text": _norm_space(whole)}]
    return [s for s in sections if s["text"]], sorted(links)


# ---------- matching uploaded PDFs to papers ----------
def _words(text):
    return re.sub(r"[^a-z0-9]+", " ", unicodedata.normalize("NFKC", str(text or "")).lower()).strip()


def match_pdf(data, filename, papers):
    """Guess which paper an uploaded PDF is. Returns (paper_id or None, how it was matched)."""
    import pymupdf

    try:
        doc = pymupdf.open(stream=data, filetype="pdf")
        pages = [doc[i].get_text("text") for i in range(min(2, doc.page_count))]
        meta_title = (doc.metadata or {}).get("title") or ""
        doc.close()
    except Exception:
        return None, "couldn't read this PDF"
    first = unicodedata.normalize("NFKC", pages[0] if pages else "")
    first_words = " " + _words(first) + " "
    candidates = [p for p in papers if _words(p.get("title"))]

    # 1. The paper's full title printed on the first page (or in the PDF's metadata) is the strongest signal.
    for p in candidates:
        title = _words(p["title"])
        if len(title) > 15 and (f" {title} " in first_words or title == _words(meta_title)):
            return p["id"], "title"
    # 2. Its own DOI or arXiv ID on the first page.
    for doi in re.findall(r"10\.\d{4,9}/[^\s\"<>]+", first):
        doi = doi.rstrip(".,;)]").lower()
        for p in candidates:
            if p.get("doi") and p["doi"].lower() == doi:
                return p["id"], "DOI"
    for m in re.finditer(r"arxiv[:\s]*(\d{4}\.\d{4,5})", first, re.I):
        for p in candidates:
            if find_arxiv_id(p) == m.group(1):
                return p["id"], "arXiv ID"
    # 3. Most of the title's words on the first two pages, in the metadata or in the file name.
    haystack = set(_words(" ".join(pages) + " " + meta_title + " " + filename).split())
    best_score, best_id = 0.0, None
    for p in candidates:
        words = [w for w in _words(p["title"]).split() if len(w) > 2]
        if not words:
            continue
        score = sum(w in haystack for w in words) / len(words)
        if score > best_score:
            best_score, best_id = score, p["id"]
    if best_score >= 0.8:
        return best_id, "title words"
    return None, "no match"


# ---------- fetching ----------
_last_arxiv_call = [0.0]


def _arxiv_get(url, timeout=40):
    # arXiv asks automated clients to leave about three seconds between requests.
    wait = 3.0 - (time.time() - _last_arxiv_call[0])
    if wait > 0:
        time.sleep(wait)
    _last_arxiv_call[0] = time.time()
    return requests.get(url, headers=USER_AGENT, timeout=timeout)


def _package(source, url, sections, links, license_=None):
    chars = sum(len(s["text"]) for s in sections)
    if chars < MIN_TEXT_CHARS:
        raise FullTextUnavailable(f"{source} had almost no extractable text (it may be a scanned PDF)")
    return {"source": source, "url": url, "license": license_, "sections": sections,
            "links": links, "chars": chars}


def get_full_text(paper, unpaywall_email="", uploaded_pdf=None):
    """Return {"source", "url", "license", "sections", "links", "chars"} or raise FullTextUnavailable."""
    if uploaded_pdf:
        sections, links = parse_pdf(uploaded_pdf)
        return _package("Uploaded PDF", None, sections, links)

    tried = []
    arxiv_id = find_arxiv_id(paper)
    if arxiv_id:
        url = f"https://arxiv.org/html/{arxiv_id}"
        try:
            r = _arxiv_get(url)
            if r.ok and "ltx_document" in r.text:
                sections, links = parse_arxiv_html(r.text)
                return _package("arXiv HTML", r.url, sections, links, "See the arXiv record for the license")
            tried.append("arXiv has no HTML version")
        except FullTextUnavailable as exc:
            tried.append(str(exc))
        except requests.RequestException as exc:
            tried.append(f"arXiv HTML request failed ({type(exc).__name__})")
        url = f"https://arxiv.org/pdf/{arxiv_id}"
        try:
            r = _arxiv_get(url, timeout=60)
            if r.ok and r.content[:4] == b"%PDF":
                sections, links = parse_pdf(r.content)
                return _package("arXiv PDF", r.url, sections, links, "See the arXiv record for the license")
            tried.append(f"arXiv PDF returned HTTP {r.status_code}")
        except FullTextUnavailable as exc:
            tried.append(str(exc))
        except requests.RequestException as exc:
            tried.append(f"arXiv PDF request failed ({type(exc).__name__})")
    else:
        tried.append("no arXiv ID in the Bond record")

    doi = (paper.get("doi") or "").strip()
    if doi and unpaywall_email:
        try:
            r = requests.get(f"https://api.unpaywall.org/v2/{quote(doi, safe='/')}",
                             params={"email": unpaywall_email}, headers=USER_AGENT, timeout=20)
            info = r.json() if r.ok else {}
            locations = [info.get("best_oa_location") or {}] + (info.get("oa_locations") or [])
            pdf_urls = list(dict.fromkeys(loc.get("url_for_pdf") for loc in locations if loc and loc.get("url_for_pdf")))
            if not pdf_urls:
                tried.append("Unpaywall found no open-access PDF")
            for pdf_url in pdf_urls[:3]:
                try:
                    pr = requests.get(pdf_url, headers=USER_AGENT, timeout=60)
                    if pr.ok and pr.content[:4] == b"%PDF":
                        sections, links = parse_pdf(pr.content)
                        lic = next((loc.get("license") for loc in locations if loc and loc.get("url_for_pdf") == pdf_url), None)
                        return _package("Open-access PDF (Unpaywall)", pdf_url, sections, links, lic)
                    tried.append("an open-access link didn't return a PDF")
                except (requests.RequestException, FullTextUnavailable) as exc:
                    tried.append(str(exc) if isinstance(exc, FullTextUnavailable) else "an open-access download failed")
        except (requests.RequestException, ValueError):
            tried.append("Unpaywall request failed")
    elif doi:
        tried.append("Unpaywall skipped (add your email in the sidebar to search for open-access copies)")
    raise FullTextUnavailable("; ".join(tried))
