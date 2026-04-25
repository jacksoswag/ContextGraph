import requests, trafilatura #type:ignore
import re, random, os, json
from dotenv import load_dotenv # type: ignore
from constants import (
    DEFAULT_UNKNOWN_YEAR,
    MIN_EXTRACTABLE_BLOCK_CHARS,
    NON_SERPER_ENGINE_SAMPLE_COUNT,
    NON_SERPER_FOLLOWUP_BATCH_SIZE,
    NON_SERPER_MIN_PER_ENGINE_LIMIT,
    SCRAPE_DEFAULT_LIMIT,
    SERPER_MAX_RESULTS,
    SERPER_FULL_PAGE_RESULTS,
    SERPER_MIN_REQUESTED_RESULTS,
    SERPER_OVERFETCH_MULTIPLIER,
    SERPER_PAGE_TIMEOUT,
    SERPER_RESULT_TIMEOUT,
    SERPER_SHARE,
    SNIPPET_ONLY_BLOCK_CHARS,
)

load_dotenv()

YEAR_RE = re.compile(r"\b(19|20)\d{2}\b")
_SERPER_API_KEY_WARNING_EMITTED = False

def _extract_page_text(url, headers=None):
    try:
        response = requests.get(url, headers=headers, timeout=SERPER_PAGE_TIMEOUT)
        response.raise_for_status()
        content_type = str(response.headers.get("content-type", "")).lower()
        if content_type and all(kind not in content_type for kind in ("html", "xml", "text")):
            return ""
        html = response.text
        if not html:
            return ""
        return trafilatura.extract(html) or ""
    except Exception:
        return ""

def _extract_year(*candidates):
    for candidate in candidates:
        text = str(candidate or "")
        match = YEAR_RE.search(text)
        if match:
            try:
                return int(match.group(0))
            except ValueError:
                continue
    return DEFAULT_UNKNOWN_YEAR

def _normalize_scrape_text(*parts):
    text = "\n\n".join(
        " ".join(str(part or "").split()).strip()
        for part in parts
        if str(part or "").strip()
    ).strip()
    return text

def _serper_result_block(result, headers=None, fetch_page=True):
    title = result.get("title", "")
    snippet = result.get("snippet", "")
    link = result.get("link", "https://google.com")
    if not str(link).startswith("http"):
        return None

    page_text = _extract_page_text(link, headers=headers) if fetch_page else ""
    text = _normalize_scrape_text(title, snippet, page_text)
    if len(text) < MIN_EXTRACTABLE_BLOCK_CHARS:
        text = _normalize_scrape_text(title, snippet)
    if len(text) < SNIPPET_ONLY_BLOCK_CHARS:
        return None

    year = _extract_year(result.get("date"))
    return (text, "google_serper", link, year)

def _serper(q, h, limit=12): # Google Search via Serper.dev
    global _SERPER_API_KEY_WARNING_EMITTED
    o = []
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key:
        if not _SERPER_API_KEY_WARNING_EMITTED:
            print("[SCRAPE] Warning: SERPER_API_KEY is missing. Serper results will be skipped.")
            _SERPER_API_KEY_WARNING_EMITTED = True
        return o
    
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": q, "num": limit})
        headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
        response = requests.post(url, headers=headers, data=payload, timeout=SERPER_RESULT_TIMEOUT)
        response.raise_for_status()
        organic_results = response.json().get("organic", [])
        if not organic_results:
            return o

        page_headers = dict(h)
        page_headers.setdefault("Accept", "text/html,application/xhtml+xml,application/xml;q=0.9,text/plain;q=0.8,*/*;q=0.7")
        for idx, res in enumerate(organic_results[:limit]):
            block = _serper_result_block(
                res,
                headers=page_headers,
                fetch_page=idx < SERPER_FULL_PAGE_RESULTS,
            )
            if block:
                o.append(block)
    except Exception as e:
        print(f"[SCRAPE] Serper request failed for '{q}': {e}")
    return o

def _loc(q, h, limit=4):
    """Library of Congress Digital Collections"""
    o = []
    try:
        r = requests.get(
            "https://www.loc.gov/search/",
            params={"q": q, "fo": "json", "c": limit},
            headers=h, timeout=3
        ).json()
        for res in r.get("results", []):
            desc = " ".join(res.get("description", []))
            url = res.get("url", "https://www.loc.gov")
            date = res.get("date", "")
            year = int(date[:4]) if (isinstance(date, str) and date[:4].isdigit()) else DEFAULT_UNKNOWN_YEAR
            if len(desc) > 50: o.append((desc, "loc", url, year))
    except: pass
    return o

def _doaj(q, h, limit=4):
    """Directory of Open Access Journals"""
    o = []
    try:
        r = requests.get(
            f"https://doaj.org/api/v2/search/articles/{q}",
            params={"pageSize": limit},
            headers=h, timeout=3
        ).json()
        for res in r.get("results", []):
            bib = res.get("bibjson", {})
            abs_text = bib.get("abstract")
            url = bib.get("link", [{}])[0].get("url", "https://doaj.org")
            year = bib.get("year", DEFAULT_UNKNOWN_YEAR)
            if abs_text: o.append((abs_text, "doaj", url, year))
    except: pass
    return o

def _wiki(q, h, limit=4):
    o = []
    try:
        r = requests.get("https://en.wikipedia.org/w/api.php",params={"action":"query","list":"search","srsearch":q,"format":"json","srlimit":limit},headers=h,timeout=2).json()
        for s in r.get("query",{}).get("search",[]):
            pid = s['pageid']
            title = s.get('title','').replace(' ','_')
            url = f"https://en.wikipedia.org/wiki/{title}"
            c = requests.get("https://en.wikipedia.org/w/api.php",params={"action":"query","prop":"extracts","explaintext":True,"pageids":pid,"format":"json"},headers=h,timeout=2).json()
            e = c['query']['pages'][str(pid)].get('extract','')
            if e: o.append((e,"wikipedia",url))
    except: pass
    return o

def _arxiv(q, h, limit=4):
    o = []
    try:
        r = requests.get(f"https://export.arxiv.org/api/query?search_query=all:{q.replace(' ','+')}&max_results={limit}",headers=h,timeout=2)
        ids = re.findall(r'<id>(.*?)</id>',r.text)
        summaries = re.findall(r'<summary>(.*?)</summary>',r.text,re.DOTALL)
        dates = re.findall(r'<published>(.*?)</published>', r.text) # <published>2023-01-01T00:00:00Z</published>
        for i, a in enumerate(summaries):
            url = ids[i+1] if i+1 < len(ids) else "https://arxiv.org"
            year = DEFAULT_UNKNOWN_YEAR
            if i < len(dates):
                try: year = int(dates[i][:4])
                except: pass
            o.append((a,"arxiv",url,year))
    except: pass
    return o

def _ss(q, h, limit=4):
    o = []
    try:
        r = requests.get("https://api.semanticscholar.org/graph/v1/paper/search",params={"query":q,"limit":limit,"fields":"title,abstract,url,year"},headers=h,timeout=2).json()
        for p in r.get("data",[]): 
            a = p.get("abstract")
            url = p.get("url","https://semanticscholar.org")
            year = p.get("year", DEFAULT_UNKNOWN_YEAR)
            if a: o.append((a,"semantic_scholar",url,year))
    except: pass
    return o

def _pm(q, h, limit=4):
    o = []
    try:
        r = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi",params={"db":"pubmed","term":q,"retmax":limit,"retmode":"json"},headers=h,timeout=2).json()
        ids = r.get("esearchresult",{}).get("idlist",[])
        if ids:
            r2 = requests.get("https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi",params={"db":"pubmed","id":",".join(ids),"rettype":"abstract","retmode":"text"},headers=h,timeout=2)
            if r2.text:
                url = f"https://pubmed.ncbi.nlm.nih.gov/{ids[0]}/"
                o.append((r2.text,"pubmed",url,DEFAULT_UNKNOWN_YEAR))
    except: pass
    return o

def _oa(q, h, limit=4):
    o = []
    try:
        r = requests.get("https://api.openalex.org/works",params={"search":q,"per_page":limit,"select":"id,title,abstract_inverted_index,publication_year","mailto":"sks@example.com"},headers=h,timeout=2).json()
        for w in r.get("results",[]):
            inv = w.get("abstract_inverted_index")
            url = w.get("id","https://openalex.org")
            year = w.get("publication_year", DEFAULT_UNKNOWN_YEAR)
            if inv:
                words = [""]*(max(max(p) for p in inv.values())+1)
                for word, positions in inv.items():
                    for p in positions:
                        if p < len(words): words[p] = word
                a = " ".join(w2 for w2 in words if w2)
                if len(a) > 50: o.append((a,"openalex",url,year))
    except: pass
    return o


def _the_conversation(q, h, limit=4): # Search The Conversation (Creative Commons, Scholarly News)
    o = []
    try:
        r = requests.get(f"https://theconversation.com/us/search?q={q.replace(' ','+')}", headers=h, timeout=2)
        urls = re.findall(r'href="(https://theconversation.com/.*?)"', r.text)[:limit]
        for u in urls:
            html = trafilatura.fetch_url(u)
            if html:
                txt = trafilatura.extract(html)
                if txt: o.append((txt, "the_conversation", u))
    except: pass
    return o

def _propublica(q, h, limit=4): # Search ProPublica (Investigative Journalism)
    o = []
    try:
        r = requests.get(f"https://www.propublica.org/search?q={q.replace(' ','+')}", headers=h, timeout=2)
        urls = re.findall(r'href="(https://www.propublica.org/article/.*?)"', r.text)[:limit]
        for u in urls:
            html = trafilatura.fetch_url(u)
            if html:
                txt = trafilatura.extract(html)
                if txt: o.append((txt, "propublica", u))
    except: pass
    return o


def scrape_one(query, cache, limit=SCRAPE_DEFAULT_LIMIT):
    if query in cache: return cache[query]
    h = {
        'User-Agent': 'SKS-Scholarly-Research-Bot/2.5 (High-Fidelity Intelligence Core)',
        'Accept': 'application/json, text/plain, */*'
    }
    serper_blocks = []
    other_blocks = []

    serper_limit = max(1, int(round(limit * SERPER_SHARE)))
    other_limit = max(0, limit - serper_limit)

    non_serper_engines = [
        _wiki,
        _arxiv,
        _ss,
        _pm,
        _oa,
        _the_conversation,
        _propublica,
        _loc,
        _doaj,
    ]

    randomized_non_serper = list(non_serper_engines)
    random.shuffle(randomized_non_serper)

    requested_serper_results = min(
        SERPER_MAX_RESULTS,
        max(SERPER_MIN_REQUESTED_RESULTS, int(round(serper_limit * SERPER_OVERFETCH_MULTIPLIER))),
    )
    try:
        serper_blocks.extend(_serper(query, h, requested_serper_results))
    except Exception:
        pass

    # Keep scraping process-parallel but engine-sequential inside each worker.
    # This avoids concurrent native HTML parsing in trafilatura, which has been
    # unstable on macOS in our worker setup.
    initial_sample_count = min(NON_SERPER_ENGINE_SAMPLE_COUNT, len(randomized_non_serper))
    pending_non_serper = randomized_non_serper[:initial_sample_count]
    remaining_non_serper = randomized_non_serper[initial_sample_count:]

    def run_non_serper_batch(engine_batch, remaining_budget):
        if remaining_budget <= 0 or not engine_batch:
            return
        per_engine_limit = max(NON_SERPER_MIN_PER_ENGINE_LIMIT, (remaining_budget + len(engine_batch) - 1) // len(engine_batch))
        for fn in engine_batch:
            try:
                fetched = fn(query, h, per_engine_limit)
            except Exception:
                fetched = []
            other_blocks.extend(fetched)

    run_non_serper_batch(pending_non_serper, other_limit)

    while len(other_blocks) < other_limit and remaining_non_serper:
        batch_size = min(NON_SERPER_FOLLOWUP_BATCH_SIZE, len(remaining_non_serper))
        next_batch = remaining_non_serper[:batch_size]
        remaining_non_serper = remaining_non_serper[batch_size:]
        run_non_serper_batch(next_batch, other_limit - len(other_blocks))

    random.shuffle(serper_blocks)
    random.shuffle(other_blocks)

    result = []
    result.extend(serper_blocks[:serper_limit])
    result.extend(other_blocks[:other_limit])

    # If one side returns fewer blocks than requested, backfill from whatever
    # remains so we still maximize total usable context for extraction.
    if len(result) < limit:
        remaining_serper = serper_blocks[min(len(serper_blocks), serper_limit):]
        remaining_other = other_blocks[min(len(other_blocks), other_limit):]
        backfill_pool = remaining_serper + remaining_other
        random.shuffle(backfill_pool)
        result.extend(backfill_pool[:max(0, limit - len(result))])

    result = result[:limit]
    
    if result: cache[query] = result # Save to cache
    return result

def scrape_worker_loop(scrape_queue, connection_queue, stop_event, cache, sync_counter=None):
    print(f"[SCRAPE] Logic-Extraction Worker Online.")

    while not stop_event.is_set():
        try:
            task = scrape_queue.get(timeout=1.0)
            query = task.get("query", "")
        except: continue
            
        result_payload = {"query": query, "blocks": []}
        try:
            print(f"[SCRAPE] Starting query: '{query}'")
            blocks = scrape_one(query, cache, limit=SCRAPE_DEFAULT_LIMIT) # Fetch
            print(f"[SCRAPE] Retrieved {len(blocks)} blocks for query: '{query}'")
            distill_blocks = [] # Package data for distillation
            for fb in blocks:
                year = fb[3] if len(fb) > 3 else DEFAULT_UNKNOWN_YEAR
                year_tag = f"|year={year}" if year else ""
                distill_blocks.append({"content": fb[0], "tag": f"{fb[1]}{year_tag}|{fb[2]}"})
            result_payload["blocks"] = distill_blocks
        except Exception as e:
            print(f"[SCRAPE] Error processing '{query}': {e}")
            result_payload["error"] = str(e)
        finally:
            connection_queue.put(result_payload)
            # Signal task completion to the main process
            if sync_counter is not None:
                try:
                    with sync_counter.get_lock():
                        sync_counter.value -= 1
                except: pass

    print("[SCRAPE] worker exiting")
