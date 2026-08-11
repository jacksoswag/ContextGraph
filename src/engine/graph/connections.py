import struct
import numpy as np  # type: ignore
from engine.common.constants import (CONNECTION_RECORD_SIZE, CONNECTION_UTILITY_OFFSET, MAX_CONNECTIONS, PREDICATE_QUANT_EXACT_OFFSET, PREDICATE_TENSE_EXACT_OFFSET, PREDICATE_TRUTH_EXACT_OFFSET, SUBJECT_QUANT_EXACT_OFFSET, SUBJECT_TENSE_EXACT_OFFSET, SUBJECT_TRUTH_EXACT_OFFSET,); from engine.extract.noise_cleanup import clean_agent_name, clean_clause_text, is_usable_agent_text
from engine.extract.word_info_map import literal_from_index; from engine.agents.connection import Connector, ConnectionEndpoint; from engine.common.shm import display_source, format_specifics, merge_specifics
# Merges subject, predicate, and relation specifics into one metadata payload.
def _merged_specifics_payload(existing=None, subject_specifics=None, predicate_specifics=None, connection_specifics=None,):
    payload = dict(existing or {}); payload["subject_specifics"] = merge_specifics(payload.get("subject_specifics"), subject_specifics,); payload["predicate_specifics"] = merge_specifics(payload.get("predicate_specifics"), predicate_specifics,); payload["connection_specifics"] = merge_specifics(payload.get("connection_specifics"), connection_specifics,)
    return payload
# Returns whether exact metadata codes can coexist on a merged connection.
def _compatible_code(existing_code, incoming_code, unknown_code=-1):
    existing = int(existing_code if existing_code is not None else unknown_code); incoming = int(incoming_code if incoming_code is not None else unknown_code)
    return existing == unknown_code or incoming == unknown_code or existing == incoming
# Keeps the existing exact code unless the incoming code is more specific.
def _prefer_specific_code(existing_code, incoming_code, unknown_code=-1):
    existing = int(existing_code if existing_code is not None else unknown_code); incoming = int(incoming_code if incoming_code is not None else unknown_code)
    if existing == unknown_code and incoming != unknown_code:
        return incoming
    return existing
# Returns whether endpoint modifier ids can coexist on a merged connection.
def _modifier_codes_compatible(existing_modifier_ids, incoming_modifier_ids):
    existing = tuple(int(value) for value in (existing_modifier_ids or ())); incoming = tuple(int(value) for value in (incoming_modifier_ids or ()))
    return not existing or not incoming or existing == incoming
# Keeps existing modifier ids unless incoming modifiers add specificity.
def _prefer_specific_modifiers(existing_modifier_ids, incoming_modifier_ids):
    existing = tuple(int(value) for value in (existing_modifier_ids or ())); incoming = tuple(int(value) for value in (incoming_modifier_ids or ()))
    if not existing and incoming:
        return incoming
    return existing
# Returns endpoint state for a connection key from metadata or shared memory.
def _state_for_key(graph, key):
    state = graph.connection_states.get(key, {}); subject_sp = state.get("subject_sp"); predicate_sp = state.get("predicate_sp")
    if isinstance(subject_sp, ConnectionEndpoint) and isinstance(predicate_sp, ConnectionEndpoint):
        return subject_sp, predicate_sp
    s_idx, o_idx, _rel_type = key[:3]; offset = graph.connection_offsets.get(key)
    if offset is not None:
        try:
            return (ConnectionEndpoint(quantifier=struct.unpack_from("<i", graph.shm_connections.buf, offset + SUBJECT_QUANT_EXACT_OFFSET,)[0], tense=struct.unpack_from("<i", graph.shm_connections.buf, offset + SUBJECT_TENSE_EXACT_OFFSET,)[0], truth=struct.unpack_from("<i", graph.shm_connections.buf, offset + SUBJECT_TRUTH_EXACT_OFFSET,)[0], ASU_idx=s_idx,), ConnectionEndpoint(quantifier=struct.unpack_from("<i", graph.shm_connections.buf, offset + PREDICATE_QUANT_EXACT_OFFSET,)[0], tense=struct.unpack_from("<i", graph.shm_connections.buf, offset + PREDICATE_TENSE_EXACT_OFFSET,)[0], truth=struct.unpack_from("<i", graph.shm_connections.buf, offset + PREDICATE_TRUTH_EXACT_OFFSET,)[0], ASU_idx=o_idx,),)
        except struct.error:
            pass
    return (ConnectionEndpoint(quantifier=-1, tense=-1, truth=-1, ASU_idx=s_idx), ConnectionEndpoint(quantifier=-1, tense=-1, truth=-1, ASU_idx=o_idx),)
# Computes state specificity for connection graph storage.
def _state_specificity(sp):
    if not isinstance(sp, ConnectionEndpoint):
        return 0
    return sum(1 for value in (sp.quantifier, sp.tense, sp.truth) if int(value) >= 0) + (1 if tuple(sp.modifier_idx or ()) else 0)
# Builds exact connection signature for connection graph storage.
def _exact_connection_signature(s_idx, o_idx, rel_type, subject_sp, predicate_sp):
    return (int(s_idx), int(o_idx), int(rel_type), int(getattr(subject_sp, "quantifier", -1)), int(getattr(subject_sp, "tense", -1)), int(getattr(subject_sp, "truth", -1)), tuple(int(value) for value in getattr(subject_sp, "modifier_idx", ()) or ()), int(getattr(predicate_sp, "quantifier", -1)), int(getattr(predicate_sp, "tense", -1)), int(getattr(predicate_sp, "truth", -1)), tuple(int(value) for value in getattr(predicate_sp, "modifier_idx", ()) or ()),)
# Returns connection bucket key for connection graph storage.
def _connection_bucket_key(s_idx, o_idx, rel_type):
    return (int(s_idx), int(o_idx), int(rel_type))
# Registers a connection key in exact and bucket indexes.
def _register_connection_key(graph, key, offset=None):
    graph.seen_connections.add(key); graph.connection_buckets.setdefault(_connection_bucket_key(*key[:3]), set()).add(key)
    if offset is not None:
        graph.connection_offsets[key] = offset
# Attaches a Connector wrapper to its subject and predicate agents.
def _attach_connector(connector):
    seen = set()
    for agent in (connector.source_agent, connector.target):
        if agent is None or agent.index in seen:
            continue
        seen.add(agent.index); agent.connectors.append(connector)
# Removes a connection key from exact, bucket, and offset indexes.
def _unregister_connection_key(graph, key):
    graph.seen_connections.discard(key); graph.connection_offsets.pop(key, None); bucket_key = _connection_bucket_key(*key[:3]); bucket = graph.connection_buckets.get(bucket_key)
    if bucket is not None:
        bucket.discard(key)
        if not bucket:
            graph.connection_buckets.pop(bucket_key, None)
# Merges compatible states while preserving stronger data.
def _merge_compatible_states(existing_subject, existing_predicate, incoming_subject, incoming_predicate):
    if not (_compatible_code(existing_subject.truth, incoming_subject.truth, -1) and _compatible_code(existing_predicate.truth, incoming_predicate.truth, -1) and _compatible_code(existing_subject.quantifier, incoming_subject.quantifier, -1) and _compatible_code(existing_predicate.quantifier, incoming_predicate.quantifier, -1) and _compatible_code(existing_subject.tense, incoming_subject.tense, -1) and _compatible_code(existing_predicate.tense, incoming_predicate.tense, -1) and _modifier_codes_compatible(existing_subject.modifier_idx, incoming_subject.modifier_idx) and _modifier_codes_compatible(existing_predicate.modifier_idx, incoming_predicate.modifier_idx)):
        return None
    merged_subject = ConnectionEndpoint(quantifier=_prefer_specific_code(existing_subject.quantifier, incoming_subject.quantifier, -1), tense=_prefer_specific_code(existing_subject.tense, incoming_subject.tense, -1), truth=_prefer_specific_code(existing_subject.truth, incoming_subject.truth, -1), ASU_idx=getattr(existing_subject, "ASU_idx", -1), modifier_idx=_prefer_specific_modifiers(existing_subject.modifier_idx, incoming_subject.modifier_idx),)
    merged_predicate = ConnectionEndpoint(quantifier=_prefer_specific_code(existing_predicate.quantifier, incoming_predicate.quantifier, -1), tense=_prefer_specific_code(existing_predicate.tense, incoming_predicate.tense, -1), truth=_prefer_specific_code(existing_predicate.truth, incoming_predicate.truth, -1), ASU_idx=getattr(existing_predicate, "ASU_idx", -1), modifier_idx=_prefer_specific_modifiers(existing_predicate.modifier_idx, incoming_predicate.modifier_idx),)
    return merged_subject, merged_predicate
# Finds an existing connection key that can absorb incoming endpoint metadata.
def _find_compatible_connection_key(graph, s_idx, o_idx, rel_type, subject_sp, predicate_sp):
    bucket = graph.connection_buckets.get(_connection_bucket_key(s_idx, o_idx, rel_type), set()); best_key = None; best_subject = None; best_predicate = None; best_specificity = -1
    for key in bucket:
        existing_subject, existing_predicate = _state_for_key(graph, key); merged_state = _merge_compatible_states(existing_subject, existing_predicate, subject_sp, predicate_sp,)
        if merged_state is None:
            continue
        merged_subject, merged_predicate = merged_state; specificity = _state_specificity(existing_subject) + _state_specificity(existing_predicate)
        if specificity > best_specificity:
            best_key = key; best_subject = merged_subject; best_predicate = merged_predicate; best_specificity = specificity
    return best_key, best_subject, best_predicate
# Reads connection utility from storage or shared memory.
def _read_connection_utility(graph, offset):
    try:
        return float(struct.unpack_from("<f", graph.shm_connections.buf, offset + CONNECTION_UTILITY_OFFSET,)[0])
    except struct.error:
        return 0.0
# Writes connection utility to storage or shared memory.
def _write_connection_utility(graph, offset, utility):
    struct.pack_into("<f", graph.shm_connections.buf, offset + CONNECTION_UTILITY_OFFSET, max(0.0, min(1.0, float(utility))),)
# Writes one fixed-size native connection record into shared memory.
def _pack_connection_record(graph, offset, s_idx, rel_type, o_idx, utility, subject_sp, predicate_sp):
    struct.pack_into("<iiifiiiiii", graph.shm_connections.buf, offset, int(s_idx), int(rel_type), int(o_idx), float(max(0.0, min(1.0, utility))), int(getattr(subject_sp, "quantifier", -1)), int(getattr(subject_sp, "tense", -1)), int(getattr(subject_sp, "truth", -1)), int(getattr(predicate_sp, "quantifier", -1)), int(getattr(predicate_sp, "tense", -1)), int(getattr(predicate_sp, "truth", -1)),)
# Rewrites a packed connection record after endpoint metadata becomes more specific.
def _rewrite_connection_record(graph, key, subject_sp, predicate_sp):
    offset = graph.connection_offsets.get(key)
    if offset is None:
        return key
    s_idx, o_idx, rel_type = key[:3]; utility = _read_connection_utility(graph, offset); _pack_connection_record(graph, offset, s_idx, rel_type, o_idx, utility, subject_sp, predicate_sp); new_key = _exact_connection_signature(s_idx, o_idx, rel_type, subject_sp, predicate_sp)
    if new_key == key:
        return key
    source = graph.connection_sources.pop(key, None); state = graph.connection_states.pop(key, None); specifics = graph.connection_specifics.pop(key, None); previous_agents = graph.connection_previous_agents.pop(key, None); evidence_text = graph.connection_texts.pop(key, None); _unregister_connection_key(graph, key)
    _register_connection_key(graph, new_key, offset=offset)
    if source is not None:
        graph.connection_sources[new_key] = source
    if state is not None:
        graph.connection_states[new_key] = state
    if specifics is not None:
        graph.connection_specifics[new_key] = specifics
    if previous_agents is not None:
        graph.connection_previous_agents[new_key] = previous_agents
    if evidence_text is not None:
        graph.connection_texts[new_key] = evidence_text
    return new_key
# Normalizes previous agent ids for connection graph storage.
def _normalize_previous_agent_ids(previous_agent_ids=None):
    if previous_agent_ids is None:
        return ()
    if isinstance(previous_agent_ids, (int, np.integer)):
        values = [int(previous_agent_ids)]
    else:
        values = list(previous_agent_ids or [])
    normalized = []; seen = set()
    for value in values:
        try:
            agent_id = int(value)
        except (TypeError, ValueError):
            continue
        if agent_id < 0 or agent_id in seen:
            continue
        seen.add(agent_id); normalized.append(agent_id)
    return tuple(sorted(normalized))
# Merges previous agent ids while preserving stronger data.
def _merge_previous_agent_ids(existing=None, incoming=None):
    return _normalize_previous_agent_ids(list(_normalize_previous_agent_ids(existing)) + list(_normalize_previous_agent_ids(incoming)))
# Merges connection metadata while preserving stronger data.
def _merge_connection_metadata(graph, key, source="unknown", specifics_payload=None, previous_agent_ids=None, evidence_text="", replace_source=False,):
    if source and source != "unknown":
        existing_source = graph.connection_sources.get(key)
        if replace_source or existing_source in (None, "", "unknown"):
            graph.connection_sources[key] = source
    if specifics_payload is not None:
        graph.connection_specifics[key] = _merged_specifics_payload(graph.connection_specifics.get(key), subject_specifics=specifics_payload.get("subject_specifics"), predicate_specifics=specifics_payload.get("predicate_specifics"), connection_specifics=specifics_payload.get("connection_specifics"),)
    merged_previous_agents = _merge_previous_agent_ids(graph.connection_previous_agents.get(key), previous_agent_ids,)
    if merged_previous_agents:
        graph.connection_previous_agents[key] = merged_previous_agents
    clean_text = " ".join(str(evidence_text or "").strip().split())
    if clean_text:
        existing_text = " ".join(str(graph.connection_texts.get(key, "") or "").strip().split())
        if replace_source or not existing_text:
            graph.connection_texts[key] = clean_text
# Stores source, specifics, state, and evidence metadata for one connection key.
def _store_connection_metadata(graph, key, source="unknown", specifics_payload=None, previous_agent_ids=None, evidence_text="",):
    graph.connection_sources[key] = source; graph.connection_specifics[key] = dict(specifics_payload or {}); graph.connection_previous_agents[key] = _normalize_previous_agent_ids(previous_agent_ids); graph.connection_texts[key] = " ".join(str(evidence_text or "").strip().split())
# Builds a default endpoint object from an agent name when extraction metadata is absent.
def _endpoint_for_name(endpoint, expected_name):
    if not isinstance(endpoint, ConnectionEndpoint):
        return None
    endpoint_name = clean_agent_name(endpoint.asu_value())
    if endpoint_name.lower() == expected_name.lower():
        return endpoint
    return ConnectionEndpoint(endpoint.quantifier, endpoint.tense, endpoint.truth, expected_name, modifier_idx=endpoint.modifier_idx,)
# Resolves upstream agent ids for a connection from explicit ids or endpoint names.
def _resolved_previous_agent_ids(graph, previous_agent_name=None, previous_agent_ids=None):
    resolved = list(_normalize_previous_agent_ids(previous_agent_ids)); previous_agent_name = str(previous_agent_name or "").strip().lower()
    if previous_agent_name:
        previous_agent = graph.agents.get(previous_agent_name)
        if previous_agent is not None:
            resolved.append(previous_agent.index)
    return _normalize_previous_agent_ids(resolved)
# Stores citable endpoint context for later thought evidence.
def _register_endpoint_context(subject_sp, predicate_sp, evidence_text, source):
    ConnectionEndpoint.register_context(subject_sp.ASU_idx, evidence_text, source); ConnectionEndpoint.register_context(predicate_sp.ASU_idx, evidence_text, source)
# Resolves relation text or id into a literal relation id.
def _resolve_relation_id(rel_type=None):
    if rel_type is not None:
        return int(rel_type)
    return -1
# Adds or merges one connection into graph state and the shared-memory connection buffer.
def add_connection(graph, s_name, o_name, rel_type=None, source="unknown", subject_sp=None, predicate_sp=None, subject_specifics=None, predicate_specifics=None, connection_specifics=None, evidence_text="", previous_agent_name=None, previous_agent_ids=None, create_agents=True,):
    s_name = clean_agent_name(s_name); o_name = clean_agent_name(o_name); evidence_text = clean_clause_text(evidence_text, subject=s_name, predicate=o_name)
    if not is_usable_agent_text(s_name) or not is_usable_agent_text(o_name):
        return
    if s_name.lower() == o_name.lower():
        return
    subject_sp = _endpoint_for_name(subject_sp, s_name) or ConnectionEndpoint(quantifier=-1, tense=-1, truth=-1, ASU_idx=s_name,); predicate_sp = _endpoint_for_name(predicate_sp, o_name) or ConnectionEndpoint(quantifier=-1, tense=-1, truth=-1, ASU_idx=o_name,)
    if create_agents:
        s_agent, _ = graph._spawn_agent(s_name); o_agent, _ = graph._spawn_agent(o_name, near=s_agent.pos if s_agent else None)
    else:
        s_agent = graph.agents.get(s_name); o_agent = graph.agents.get(o_name)
    if not s_agent or not o_agent:
        return
    if graph.connection_count >= MAX_CONNECTIONS:
        return
    relation_id = _resolve_relation_id(rel_type=rel_type)
    if relation_id < 0:
        return
    resolved_previous_agent_ids = _resolved_previous_agent_ids(graph, previous_agent_name=previous_agent_name, previous_agent_ids=previous_agent_ids,); specifics_payload = _merged_specifics_payload(None, subject_specifics=subject_specifics, predicate_specifics=predicate_specifics, connection_specifics=connection_specifics,)
    utility = Connector.utility_for(subject_sp, relation_id, predicate_sp, subject_specifics=specifics_payload.get("subject_specifics"), predicate_specifics=specifics_payload.get("predicate_specifics"), connection_specifics=specifics_payload.get("connection_specifics"), evidence_text=evidence_text,)
    key = _exact_connection_signature(s_agent.index, o_agent.index, relation_id, subject_sp, predicate_sp)
    if key in graph.connection_offsets:
        offset = graph.connection_offsets.get(key)
        if offset is not None:
            _write_connection_utility(graph, offset, max(_read_connection_utility(graph, offset), utility))
        graph.connection_states[key] = {"subject_sp": subject_sp, "predicate_sp": predicate_sp}; _merge_connection_metadata(graph, key, source=source, specifics_payload=specifics_payload, previous_agent_ids=resolved_previous_agent_ids, evidence_text=evidence_text,); _register_endpoint_context(subject_sp, predicate_sp, evidence_text, source)
        return
    compatible_key, merged_subject_sp, merged_predicate_sp = _find_compatible_connection_key(graph, s_agent.index, o_agent.index, relation_id, subject_sp, predicate_sp,)
    if compatible_key is not None:
        subject_state = merged_subject_sp or subject_sp; predicate_state = merged_predicate_sp or predicate_sp; active_key = _rewrite_connection_record(graph, compatible_key, subject_state, predicate_state); offset = graph.connection_offsets.get(active_key)
        if offset is not None:
            _write_connection_utility(graph, offset, max(_read_connection_utility(graph, offset), utility))
        graph.connection_states[active_key] = {"subject_sp": subject_state, "predicate_sp": predicate_state}; _merge_connection_metadata(graph, active_key, source=source, specifics_payload=specifics_payload, previous_agent_ids=resolved_previous_agent_ids, evidence_text=evidence_text,)
        _register_endpoint_context(subject_sp, predicate_sp, evidence_text, source)
        return
    _store_connection_metadata(graph, key, source=source, specifics_payload=specifics_payload, previous_agent_ids=resolved_previous_agent_ids, evidence_text=evidence_text,); graph.connection_states[key] = {"subject_sp": subject_sp, "predicate_sp": predicate_sp}; _register_endpoint_context(subject_sp, predicate_sp, evidence_text, source)
    offset = 4 + (graph.connection_count * CONNECTION_RECORD_SIZE); _pack_connection_record(graph, offset, s_agent.index, relation_id, o_agent.index, utility, subject_sp, predicate_sp)
    connector = Connector(graph.shm_connections.buf, offset, graph.agents_by_idx, subject_sp=subject_sp, predicate_sp=predicate_sp, source=source, evidence_text=evidence_text, subject_specifics=specifics_payload.get("subject_specifics"), predicate_specifics=specifics_payload.get("predicate_specifics"), connection_specifics=specifics_payload.get("connection_specifics"), previous_agent_ids=resolved_previous_agent_ids,)
    _attach_connector(connector); graph.connection_count += 1; struct.pack_into("i", graph.shm_connections.buf, 0, graph.connection_count); _register_connection_key(graph, key, offset=offset); relation_text = literal_from_index(relation_id) or str(relation_id)
    previous_labels = ", ".join(graph.agents_by_idx[idx].ASU for idx in resolved_previous_agent_ids if idx in graph.agents_by_idx) or "-"; subject_modifier_text = " ".join(subject_sp.modifier_value()).strip() or "-"; predicate_modifier_text = " ".join(predicate_sp.modifier_value()).strip() or "-"
    graph._conn_log.append(f"[CONN] {s_name} -> {o_name} | Type: {relation_id} ({relation_text}) | " f"S(Q{subject_sp.quantifier},T{subject_sp.tense},TR{subject_sp.truth}) | " f"P(Q{predicate_sp.quantifier},T{predicate_sp.tense},TR{predicate_sp.truth}) | " f"Utility[{utility:.3f}] | " f"Modifiers: S[{subject_modifier_text}] P[{predicate_modifier_text}] | " f"Specifics: S[{format_specifics(specifics_payload.get('subject_specifics')) or '-'}] " f"P[{format_specifics(specifics_payload.get('predicate_specifics')) or '-'}] " f"R[{format_specifics(specifics_payload.get('connection_specifics')) or '-'}] | " f"Prev[{previous_labels}] | " f"Source: {display_source(source)}\n")
    if len(graph._conn_log) >= 100:
        graph.flush_conn_log()
