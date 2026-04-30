import time
from queue import Empty

from constants import EXTRACTION_BLOCK_LIMIT, MAX_CONNECTIONS_PER_QUERY
from d_logic_extractor import find_connections


def extraction_worker_loop(extract_queue, asu_queue, stop_event):
    while not stop_event.is_set():
        try:
            task = extract_queue.get(timeout=0.2)
        except Empty:
            continue

        task = dict(task or {})
        query = task.get("query", "unknown")
        research_pass = task.get("research_pass", "primary")
        blocks = list(task.get("blocks", []) or [])
        connections = list(task.get("connections", []) or [])

        try:
            if blocks and not connections and not task.get("error"):
                extraction_blocks = blocks[:EXTRACTION_BLOCK_LIMIT]
                print(
                    f"[EXTRACT:{research_pass}] Extracting {len(extraction_blocks)}/{len(blocks)} "
                    f"blocks for query: '{query}'"
                )
                connections = find_connections(
                    extraction_blocks,
                    query=query,
                    connection_limit=MAX_CONNECTIONS_PER_QUERY,
                )
                print(
                    f"[EXTRACT:{research_pass}] Extracted {len(connections)} connections "
                    f"(cap {MAX_CONNECTIONS_PER_QUERY}) for query: '{query}'"
                )
            task["connections"] = connections
            task["extracted"] = True
        except Exception as exc:
            task["connections"] = []
            task["extracted"] = True
            task["error"] = str(exc)
        finally:
            asu_queue.put(task)
            time.sleep(0.01)
