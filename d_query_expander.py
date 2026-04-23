import ollama, re # type: ignore

def expand_query_llm(query): # Uses Ollama 1B to generate 8 related research queries.
    expanded = []
    try:
        # Prompt Strict 1B model
        p1 = f"You are a researcher. Generate 3 short keyword queries (3-5 words each) related to '{query}'. RULES: NO intro. NO numbers. NO bullet points. JUST the queries.\n{query}:"
        r1 = ollama.generate(model='llama3.2:1b', prompt=p1, options={"temperature": 0.0})
        p2 = f"You are a researcher. Generate 4 short keyword queries (1-2 words each) related to '{query}'. RULES: NO intro. NO numbers. NO bullet points. JUST the queries.\n{query}:"
        r2 = ollama.generate(model='llama3.2:1b', prompt=p2, options={"temperature": 0.0})
        raw = r1['response'] + "\n" + r2['response'] # Combine LLM Output
        for line in raw.split('\n'):
            clean = re.sub(r'^[0-9]+[\.\-\)\s]+', '', line.strip()) # Strip leading numbers (1., 2., 1 -)
            clean = re.sub(r'^[\-\*\•\s]+', '', clean) # Strip bullet points
            if not clean or ":" in clean or "here are" in clean.lower() or "research queries" in clean.lower(): # Filter out conversational junk
                continue
            expanded.append(clean)
    except Exception as e:
        print(f"LLM Error: {e}")
    return expanded

def expand(query): # Main entry point: Subsets + LLM Expansion.
    print(f"Expanding root query: '{query}'")
    llm_queries = expand_query_llm(query)
    all_queries = [query] + llm_queries
    return all_queries
