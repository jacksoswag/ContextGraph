import os
import sys
import json
import struct
import numpy as np # type: ignore
from multiprocessing import shared_memory
from sentence_transformers import SentenceTransformer # type: ignore

# This script attaches to shared-memory segments created and owned by main.py.
# If Python's resource tracker registers them here, it may unlink them on exit,
# which breaks the later physics handoff.
from multiprocessing import resource_tracker
def remove_shm_from_resource_tracker():
    def fix_register(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.register(name, rtype)
    resource_tracker.register = fix_register

    def fix_unregister(name, rtype):
        if rtype == "shared_memory":
            return
        return resource_tracker._resource_tracker.unregister(name, rtype)
    resource_tracker.unregister = fix_unregister

remove_shm_from_resource_tracker()

def group_ASUs():
    print("[GROUPING] Initializing Semantic Information Grouping...")
    if not os.path.exists("agents_pre_group.json"): # Load Agent Names from the Mind's export
        print("[GROUPING] ERROR: No agent data found. Exporting from Mind...")
        return
    with open("agents_pre_group.json", "r") as f:
        agent_data = json.load(f) # List of {"index": i, "name": "..."}
    if not agent_data:
        print("[GROUPING] No agents to group.")
        return
    model = SentenceTransformer('all-MiniLM-L6-v2') # Generate Embeddings (384-dimensional)
    names = [a["name"] for a in agent_data]
    print(f"[GROUPING] Embedding {len(names)} agents...")
    embeddings = model.encode(names)
    # Clustering (Row-by-Row to save memory)
    print("[GROUPING] Computing semantic clusters (Row-by-Row)...")
    groups = []
    visited = set()
    # Pre-normalize embeddings for faster dot product similarity
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    norm_embeddings = embeddings / (norms + 1e-10)
    for i in range(len(names)):
        if i in visited: continue
        current_group = [i]
        visited.add(i)
        # Calculate similarity for JUST this agent against all others
        target_vec = norm_embeddings[i]
        similarities = np.dot(norm_embeddings, target_vec)
        # Find matches >= 0.95
        matches = np.where((similarities >= 0.90) & (np.arange(len(names)) > i))[0]
        for idx in matches:
            if idx not in visited:
                current_group.append(idx)
                visited.add(idx)
        groups.append(current_group)
    print(f"[GROUPING] Found {len(groups)} unique concept clusters from {len(names)} agents.")
    index_map = {}
    new_agents = []
    for group in groups:
        group_names = [agent_data[idx]["name"] for idx in group]
        winner_name = max(group_names, key=len)
        winner_idx = agent_data[group[0]]["index"]
        for member_idx_in_list in group:
            real_old_idx = agent_data[member_idx_in_list]["index"]
            index_map[real_old_idx] = winner_idx
            
        new_agents.append({"index": winner_idx, "name": winner_name})

    # 5. Graph Surgery: Update SHM Connections
    shm_name = os.getenv("SHM_SKS_CONNECTIONS")
    if not shm_name:
        print("[GROUPING] ERROR: SHM_SKS_CONNECTIONS env not set.")
        return

    shm = shared_memory.SharedMemory(name=shm_name)
    count = struct.unpack_from("i", shm.buf, 0)[0]
    
    new_connections = []
    seen_conns = set()
    
    print(f"[GROUPING] Rewriting {count} logical connections...")
    
    for i in range(count):
        off = 4 + (i * 16)
        idxA, rel_type, flags, idxB = struct.unpack_from("iiii", shm.buf, off)

        # Map to new representatives
        newA = index_map.get(idxA, idxA)
        newB = index_map.get(idxB, idxB)

        # Skip self-loops and duplicates
        if newA == newB: continue
        conn_key = (newA, newB, rel_type, flags)
        if conn_key in seen_conns: continue

        seen_conns.add(conn_key)
        new_connections.append((newA, rel_type, flags, newB))

    # Clear and rewrite
    shm.buf[4:] = b"\x00" * (shm.size - 4)
    for i, (a, t, n, b) in enumerate(new_connections):
        off = 4 + (i * 16)
        struct.pack_into("iiii", shm.buf, off, a, t, n, b)
    
    struct.pack_into("i", shm.buf, 0, len(new_connections))
    
    # 6. Export Mapping for Mind Sync
    with open("agent_mapping.json", "w") as f:
        json.dump({
            "mapping": index_map,
            "new_agents": new_agents
        }, f)
        
    shm.close()
    print(f"[GROUPING] Success. Graph optimized to {len(new_connections)} connections.")

if __name__ == "__main__":
    group_ASUs()
