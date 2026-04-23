import requests, trafilatura #type:ignore
import re, random, os, json
from concurrent.futures import ThreadPoolExecutor, as_completed
from d_logic_extractor import find_connections

def _serper(q, h, limit=4): # Google Search via Serper.dev
    o = []
    api_key = os.getenv("SERPER_API_KEY")
    if not api_key: return o
    
    try:
        url = "https://google.serper.dev/search"
        payload = json.dumps({"q": q, "num": limit})
        headers = {'X-API-KEY': api_key, 'Content-Type': 'application/json'}
        r = requests.post(url, headers=headers, data=payload, timeout=5).json()
        for res in r.get("organic", []):
            snippet = res.get("snippet", "")
            link = res.get("link", "https://google.com")
            # Try to grab year from the result if present, else default to 2024
            year = 2024
            if len(snippet) > 20:
                o.append((snippet, "google_serper", link, year))
    except: pass
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
            date = res.get("date", "2024")
            year = int(date[:4]) if (isinstance(date, str) and date[:4].isdigit()) else 2024
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
            year = bib.get("year", 2024)
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
            year = 2024
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
            year = p.get("year", 2024)
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
                o.append((r2.text,"pubmed",url,2024)) # PubMed fetch is raw text, defaulting to 2024
    except: pass
    return o

def _oa(q, h, limit=4):
    o = []
    try:
        r = requests.get("https://api.openalex.org/works",params={"search":q,"per_page":limit,"select":"id,title,abstract_inverted_index,publication_year","mailto":"sks@example.com"},headers=h,timeout=2).json()
        for w in r.get("results",[]):
            inv = w.get("abstract_inverted_index")
            url = w.get("id","https://openalex.org")
            year = w.get("publication_year", 2024)
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


def scrape_one(query, cache, limit=40):
    if query in cache: return cache[query]
    h = {
        'User-Agent': 'SKS-Scholarly-Research-Bot/2.5 (High-Fidelity Intelligence Core)',
        'Accept': 'application/json, text/plain, */*'
    }
    blocks = []
    engine_limit = max(4, limit // 10) 
    engines = [
        (_wiki, query, h, engine_limit), 
        (_arxiv, query, h, engine_limit), 
        (_ss, query, h, engine_limit), 
        (_pm, query, h, engine_limit), 
        (_oa, query, h, engine_limit), 
        (_the_conversation, query, h, engine_limit), 
        (_propublica, query, h, engine_limit),
        (_serper, query, h, engine_limit),
        (_loc, query, h, engine_limit),
        (_doaj, query, h, engine_limit)
    ]

    with ThreadPoolExecutor(20) as ex:
        futures = [ex.submit(fn, q, headers, lim) for fn, q, headers, lim in engines]
        for f in as_completed(futures): # for each future
            try: blocks.extend(f.result())
            except: pass
    random.shuffle(blocks)
    result = blocks[:limit]
    
    if result: cache[query] = result # Save to cache
    return result

def scrape_worker_loop(scrape_queue, connection_queue, stop_event, cache, sync_counter=None):
    print(f"[SCRAPE] Logic-Extraction Worker Online.")

    while not stop_event.is_set():
        try:
            task = scrape_queue.get(timeout=1.0)
            if task.get("cmd") == "wave_complete":
                connection_queue.put(task)
                continue
            query = task.get("query", "")
        except: continue
            
        result_payload = {"query": query, "connections": []}
        try:
            print(f"[SCRAPE] Starting query: '{query}'")
            blocks = scrape_one(query, cache, limit=80) # Fetch
            print(f"[SCRAPE] Retrieved {len(blocks)} blocks for query: '{query}'")
            distill_blocks = [] # Package data for distillation
            for fb in blocks: distill_blocks.append({"content": fb[0], "tag": f"{fb[1]}|{fb[2]}"})
            
            connections = find_connections(distill_blocks) # Extract Connections
            print(f"[SCRAPE] Extracted {len(connections)} connections for query: '{query}'")
            result_payload["connections"] = connections
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
