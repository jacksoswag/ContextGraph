from d_logic_extractor import find_connections
import json

test_blocks = [
    {
        "content": "Einstein was a legendary physicist [1]. He discovered the theory of relativity in (1905).",
        "tag": "ArXiv",
        "url": "https://arxiv.org/example"
    },
    {
        "content": "All humans are mortal. Socrates is a human, so he is mortal.",
        "tag": "Wiki",
        "url": "https://wikipedia.org/socrates"
    },
    {
        "content": "Innovation causes economic growth, but it does not guarantee happiness.",
        "tag": "News",
        "url": "https://news.com/tech"
    }
]

print("--- STARTING PIPELINE TEST ---")
results = find_connections(test_blocks)

for i, conn in enumerate(results):
    print(f"\n[Connection {i+1}]")
    print(f"  Subject:   {conn.get('subject')}")
    print(f"  Predicate: {conn.get('predicate')}")
    print(f"  Verb:      {conn.get('verb', 'N/A')}")
    print(f"  Source:    {conn.get('source')}")
    print(f"  Truth:     {conn.get('truth')}")

print("\n--- TEST COMPLETE ---")
