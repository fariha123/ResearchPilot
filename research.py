"""Normalize search results, verified external resources, and paper-grounded analysis."""
import json, os, re
from datetime import datetime, date
from urllib.parse import quote
import requests


def flatten_response(value):
    """Handle different MCP envelopes without treating raw prose as fabricated papers."""
    if isinstance(value, list):
        acc=[]
        for x in value: acc.extend(flatten_response(x))
        return acc
    if isinstance(value, dict):
        if any(k in value for k in ("title", "paper_title")):
            return [value]
        for k in ("papers", "results", "items", "works", "data", "records", "matches"):
            if k in value:
                rows=flatten_response(value[k])
                if rows: return rows
        if "raw" in value:
            try: return flatten_response(json.loads(value["raw"]))
            except (ValueError, TypeError): return []
    return []

def _year_from_date(value):
    if isinstance(value, dict):
        for key in ("date-parts", "date_parts", "published", "date", "value", "raw"):
            if key in value:
                val = value[key]
                if key in ("date-parts", "date_parts") and isinstance(val, list) and val:
                    part = val[0]
                    if isinstance(part, list) and part:
                        return _year_from_date(part[0])
                else:
                    y = _year_from_date(val)
                    if y: return y
        return None
    if isinstance(value, list):
        for v in value:
            y=_year_from_date(v)
            if y:return y
        return None
    text = str(value or "")
    match = re.search(r"(?<!\d)(19\d{2}|20\d{2})(?!\d)", text)
    if match: return int(match.group(1))
    if text.isdigit() and len(text) == 4 and 1900 <= int(text) <= date.today().year+1:
        return int(text)
    return None

def _year_from_url(url):
    from urllib.parse import urlparse
    u=urlparse(url)
    path=u.path
    # arXiv IDs YYMM.NNNNN encode submission month, not necessarily journal publication year.
    if u.netloc.lower().endswith("arxiv.org"):
        m=re.search(r"(?:/abs/|/pdf/)(\d{2})(0[1-9]|1[0-2])\.\d{4,5}",path)
        if m:
            yy=int(m.group(1)); return (2000+yy if yy<=date.today().year%100+1 else 1900+yy),"arXiv identifier (submission year)"
    for pat in (r"/(19\d{2}|20\d{2})/",r"/(19\d{2}|20\d{2})-(?:0[1-9]|1[0-2])"):
        m=re.search(pat,path)
        if m:return int(m.group(1)),"URL path (unverified)"
    return None,None

def _doi_metadata(doi, timeout=8):
    if not doi: return {}
    try:
        response=requests.get('https://api.crossref.org/works/'+quote(doi,safe='/'),timeout=timeout,headers={'User-Agent':'ResearchPilot/1.1 (research metadata lookup)'})
        if response.ok:
            msg=response.json().get('message',{})
            return {'year': _year_from_date(msg.get('published') or msg.get('issued') or msg.get('created')), 'date_source':'Crossref DOI metadata'}
    except (requests.RequestException, ValueError): pass
    return {}

def normalize(response_sets, start, end, enrich_doi=False):
    seen=set(); papers=[]
    for tool, payload in response_sets:
        for p in flatten_response(payload):
            title=str(p.get("title") or p.get("paper_title") or "").strip()
            if not title: continue
            doi=str(p.get("doi") or "").removeprefix("https://doi.org/").strip()
            url=p.get("url") or p.get("paper_url") or p.get("link") or p.get("arxiv_url") or ("https://doi.org/"+doi if doi else "")
            if isinstance(url, dict): url=url.get("uri", "")
            if not url: url="https://scholar.google.com/scholar?q="+quote(title)
            url=str(url)
            year=None; source=None
            for key in ('year','publication_year','published_year','publicationYear','published', 'published_date','publication_date','date','publicationDate','published_at','publication','issued','created'):
                year=_year_from_date(p.get(key))
                if year: source='Bond: '+key; break
            if not year:
                year,source=_year_from_url(url)
            if not year and enrich_doi:
                meta=_doi_metadata(doi)
                year=meta.get('year'); source=meta.get('date_source')
            if year and not (int(start[:4]) <= year <= int(end[:4])):continue
            key=(doi.lower() or title.lower())
            if key in seen:continue
            seen.add(key)
            authors=p.get("authors") or []
            if isinstance(authors,list): authors=", ".join(a if isinstance(a,str) else str(a.get("name", "")) for a in authors[:8])
            papers.append({"id":f"P{len(papers)+1:03d}","title":title,"year":year,"year_source":source or "not provided", "authors":str(authors),
                           "abstract":str(p.get("abstract") or p.get("summary") or ""),
                           "doi":doi,"url":url,"source_tool":tool,"raw":p})
    return sorted(papers,key=lambda p:(p["year"] or 0,p["title"]),reverse=True)

SYSTEM="""You are a rigorous scientific evidence analyst. Only cite supplied paper IDs [P001]. Treat the supplied abstracts and full-text extractions as the maximum available evidence: papers with evidence_basis 'full text' were reviewed in full (their extractions carry section names), the rest only by abstract. Do NOT infer methods, chemicals, model hardware, findings, future work, GitHub links, or prices not directly stated. Prefer full-text evidence when it is available: for papers read in full, ground gaps in their stated limitations, stated future work and quantitative findings, and take evidence_quotes from their verified_full_text_quotes. Mark gaps as hypotheses based on limited coverage. Do not invent source IDs. Distinguish 'reported', 'hypothesis', and 'unknown'. Return strictly JSON with keys gaps, resources, steps. gaps: list of {title, explanation, evidence_ids, evidence_quotes, confidence, status}; resources: list of {name, category, why, evidence_ids, status}; steps: list of {title, details, evidence_ids, status}. Cite evidence_ids only when actually supported. Steps can be recommendations/status=proposed and evidence_ids empty. Max 6 gaps, 20 resources, 10 steps. Include a caution that absence from a limited paper corpus does not establish novelty."""

def _model_client(provider, api_key, base_url):
    from openai import OpenAI
    if provider == 'Local Ollama':
        return OpenAI(base_url=(base_url or 'http://localhost:11434/v1').rstrip('/')+'/', api_key='ollama', timeout=240)
    if not api_key: raise ValueError('OpenAI key required for OpenAI analysis. Choose Local Ollama for no paid model API.')
    return OpenAI(api_key=api_key, timeout=180)

def _parse_json(text):
    text=(text or '').strip()
    if text.startswith('```'):
        text=re.sub(r'^```(?:json)?\s*|\s*```$', '',text, flags=re.I).strip()
    try:return json.loads(text)
    except json.JSONDecodeError:
        start=text.find('{'); end=text.rfind('}')
        if start>=0 and end>start:return json.loads(text[start:end+1])
        raise ValueError('LLM did not return parseable JSON. Try another local model or rerun.')

def _ask(client, model, messages):
    response=client.chat.completions.create(model=model,temperature=0,response_format={'type':'json_object'},messages=messages)
    return _parse_json(response.choices[0].message.content)

def _contains(quote, text):
    from quotes import quote_match
    return quote_match(quote, text) is not None

def _review_observation(rv):
    """Turn a full-text review (see review.py) into an observation. Only items whose quote was found in the
    paper (exactly or closely) are used; stated limitations and future work come first because gaps grow from them."""
    def pick(key, field, n):
        return [f"{x.get(field) or ''} ({x.get('section')})" for x in rv.get(key,[]) if x.get('verified') and x.get(field)][:n]
    results=[' '.join(str(x.get(k)) for k in ('finding','metric','value','condition') if x.get(k))+f" ({x.get('section')})"
             for x in rv.get('results',[]) if x.get('verified')][:12]
    return {'paper_id':rv['paper_id'],'evidence_basis':'full text','summary':rv.get('summary',''),
            'contribution':rv.get('contribution',''),
            'stated_limitations':pick('limitations','item',10),'stated_future_work':pick('future_work','item',8),
            'findings':results,'methods':pick('methods','item',10),'reported_resources':pick('resources','name',10),
            'code_and_data_links':list(rv.get('links') or [])[:5],
            'reviewer_assessment':{'weaknesses':list(rv.get('weaknesses') or [])[:5],
                                   'open_questions':list(rv.get('open_questions') or [])[:5]}}

def analyze(papers, question, api_key='', model='qwen3:8b', provider='Local Ollama', base_url='http://localhost:11434/v1', reviews=None):
    # Two-stage evidence harness limits hallucinations and fits local model context windows.
    # Papers with a full-text review contribute their verified extractions; the rest are read by abstract.
    client=_model_client(provider,api_key,base_url)
    reviews=reviews or {}
    usable=[{k:p[k] for k in ('id','title','year','abstract','doi','url')} for p in papers]
    selected=usable[:30]
    reviewed=[p for p in selected if p['id'] in reviews]
    abstract_only=[p for p in selected if p['id'] not in reviews]
    observations=[_review_observation(reviews[p['id']]) for p in reviewed]
    batches=[abstract_only[i:i+5] for i in range(0,len(abstract_only),5)]
    for batch in batches:
        result=_ask(client,model,[
          {'role':'system','content':'Extract ONLY information explicitly supported by supplied titles and abstracts. Never infer materials, results or software not stated. Return JSON object with key observations: list of {paper_id, finding, limitation, methods, reported_resources}. Each entry should be brief. No invented quotes.'},
          {'role':'user','content':json.dumps({'question':question,'papers':batch},ensure_ascii=False)[:18000]}
        ])
        ids={p['id'] for p in batch}
        for x in result.get('observations',[]):
            if isinstance(x,dict) and x.get('paper_id') in ids: observations.append(x)
    result=_ask(client,model,[
      {'role':'system','content':SYSTEM+' Do not generate evidence_quotes unless exact substrings of supplied abstracts. No invented DOIs, URL, or codes. Steps may be proposed and lack evidence IDs. Include at most 5 gap candidates, 15 resources, 8 steps.'},
      {'role':'user','content':json.dumps({'question':question,'observations':observations,'paper_reference':[
          {**{k:p[k] for k in ('id','title','abstract')},'evidence_basis':'full text' if p['id'] in reviews else 'abstract only',
           **({'verified_full_text_quotes':[x.get('quote') for key in ('results','methods','limitations','future_work')
                                            for x in reviews[p['id']].get(key,[]) if x.get('verified')][:20]} if p['id'] in reviews else {})}
          for p in selected]},ensure_ascii=False)[:70000]}
    ])
    valid={p['id']:p for p in usable}
    evidence_text={pid:p['abstract']+' '+' '.join(str(x.get('quote','')) for key in ('results','methods','limitations','future_work','resources')
                                                  for x in reviews.get(pid,{}).get(key,[]) if x.get('verified'))
                   for pid,p in valid.items()}
    for category in ('gaps','resources','steps'):
        items=result.get(category,[])
        result[category]=items if isinstance(items,list) else []
        for item in result[category]:
            if not isinstance(item,dict):continue
            ids=item.get('evidence_ids',[])
            item['evidence_ids']=[pid for pid in ids if pid in valid] if isinstance(ids,list) else []
            quotes=item.get('evidence_quotes',[])
            if isinstance(quotes,list):
                item['evidence_quotes']=[q for q in quotes if isinstance(q,str) and any(_contains(q,evidence_text[pid]) for pid in item['evidence_ids'])]
            if not item['evidence_ids'] and item.get('status')=='reported': item['status']='unverified'
    result['coverage_note']=(f'Analyzed {len(selected)} papers: {len(reviewed)} from full-text reviews and '
                             f'{len(abstract_only)} from abstracts only.')
    result['full_text_ids']=[p['id'] for p in reviewed]
    return result

PLAN_PROMPT="""You plan research to close ONE research gap. Use only the supplied evidence: the gap, its cited papers
(with full-text extractions where they were read in full: methods, stated limitations, stated future work, findings),
and the general resources. Return a JSON object with keys:
- shared_steps: setup steps any plan in this project needs (data, instruments, baselines). Give these ONLY when
  existing_shared_steps is empty; otherwise return [].
- steps: 3-7 steps specific to closing this gap, in order; each {title, details, evidence_ids}. Titles under 8 words.
- resources: what this gap needs (materials, software, datasets, instruments, code); each
  {name, category, why, evidence_ids, status ("reported" when a cited paper states it, otherwise "proposed")}
- success_criteria: one or two sentences naming the measurable result that would show the gap is closed.
Cite only supplied paper ids. Do not repeat existing_shared_steps. Build on what the cited papers already did and on
the limitations and future work they state."""

def _no_think(model):
    return ' /no_think' if 'qwen3' in str(model).lower() else ''

def plan_for_gap(gap, question, papers, reviews=None, existing_shared=None, resources=None, api_key='', model='qwen3:8b',
                 provider='Local Ollama', base_url='http://localhost:11434/v1'):
    """A focused plan for one gap: shared setup steps (first gap only), gap steps, resources and success criteria."""
    client=_model_client(provider,api_key,base_url)
    reviews=reviews or {}
    cited=[p for p in papers if p['id'] in (gap.get('evidence_ids') or [])]
    context=[]
    for p in cited:
        item={'id':p['id'],'title':p['title'],'abstract':(p.get('abstract') or '')[:1500],
              'evidence_basis':'full text' if p['id'] in reviews else 'abstract only'}
        if p['id'] in reviews:
            obs=_review_observation(reviews[p['id']])
            item.update({k:obs[k] for k in ('summary','stated_limitations','stated_future_work','findings','methods',
                                            'reported_resources','code_and_data_links')})
        context.append(item)
    payload={'research_question':question,
             'gap':{k:gap.get(k) for k in ('title','explanation','evidence_quotes')},
             'cited_papers':context,
             'existing_shared_steps':[s.get('title') for s in (existing_shared or [])],
             'general_resources':[{k:r.get(k) for k in ('name','category','why')} for r in (resources or []) if isinstance(r,dict)][:15]}
    result=_ask(client,model,[{'role':'system','content':PLAN_PROMPT},
                              {'role':'user','content':json.dumps(payload,ensure_ascii=False)[:30000]+_no_think(model)}])
    if not isinstance(result,dict): result={}
    valid={p['id'] for p in papers}
    for key in ('shared_steps','steps','resources'):
        items=result.get(key)
        result[key]=[x for x in items if isinstance(x,dict) and (x.get('title') or x.get('name'))] if isinstance(items,list) else []
        for x in result[key]:
            ids=x.get('evidence_ids')
            x['evidence_ids']=[i for i in ids if i in valid] if isinstance(ids,list) else []
            if key=='resources':
                x['status']='reported' if x['evidence_ids'] and x.get('status')=='reported' else 'proposed'
            else:
                x['status']='proposed'
    if existing_shared:
        result['shared_steps']=[]
    result['success_criteria']=str(result.get('success_criteria') or '')
    return result

def github_search(q, token="", limit=5):
    """Verified live repository URLs, not claims of paper-author affiliation."""
    h={"Accept":"application/vnd.github+json", "User-Agent":"ResearchPilot-MVP"}
    if token: h["Authorization"]="Bearer "+token
    r=requests.get("https://api.github.com/search/repositories",params={"q":q,"per_page":limit},headers=h,timeout=15)
    r.raise_for_status()
    return [{"name":x["full_name"],"url":x["html_url"],"description":x.get("description") or "", "stars":x["stargazers_count"]} for x in r.json().get("items",[])]

def markdown_report(question,papers,analysis):
    lines=["# ResearchPilot evidence report", "",f"**Question:** {question}","", f"**Papers retrieved:** {len(papers)}", "", "## Paper bibliography"]
    for p in papers: lines.append(f'- [{p["id"]}] {p["title"]} ({p["year"] or "unknown year"}). {p["url"]}')
    for section,arr in (("Research gaps", "gaps"),("Resources", "resources"),("Step-by-step plan", "steps")):
        lines.extend(["",f"## {section}"])
        for it in (analysis or {}).get(arr,[]):
            title=it.get("title") or it.get("name") or "Item"
            explanation=it.get("explanation") or it.get("why") or it.get("details") or ""
            ids=", ".join("["+x+"]" for x in it.get("evidence_ids",[])) or "No directly supporting paper"
            lines.extend([f"### {title}", f"{explanation}",f"- Evidence: {ids}; status: {it.get('status','unspecified')}"])
    lines.extend(["", "**Caution:** This is a selective literature search. Unretrieved work may exist. Resource availability, licenses and reproducibility require independent checks."])
    return "\n".join(lines)
