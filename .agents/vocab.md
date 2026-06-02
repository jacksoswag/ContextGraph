# Project Vocabulary & Terminology

**Purpose:** Stabilize language across reviews, refactors, tests, and reports. 
*Note: Terms describe working vocabulary, not proof of implementation correctness.*

## 1. General Terms
- **Behavioral Change:** Smallest user/system-visible behavior affected by a change (e.g., relation extraction, traversal ranking, realization wording, session state, memory format).
- **Native Behavior:** Outcomes produced by ordinary data flow and abstractions (NOT one-off branches/fixtures).
- **Hardcoding:** Bypassing general mechanisms to make a narrow input pass via branches, lookups, phrase lists, modes, or exceptions.
- **Compatibility Layer:** Isolated, named, and reported API/path translating legacy names into current architecture.

## 2. System Terms
- **Data:** Examples, seed records, persisted observations, fixtures, or external source material.
- **Traversal:** Movement through candidate structures, paths, memories, relations, or regions to gather support.
- **Extraction:** Converting input text/source material into structured internal data.
- **Truth Space:** The deterministic, verifiable, factual memory stored in the graph (e.g., SVO structures).
- **Discourse Space:** The conversational, rhetorical, and narrative paths used for communication, explicitly separated from Truth Space.
- **Thermodynamic Traversal:** Energy-based routing through the graph that prioritizes provenance and limits hallucination.

## 3. Preferred Language Mapping
Use neutral, durable names. **DO NOT** encode obsolete design claims. Remove old terms or restrict to compatibility boundaries.

| Prefer | Avoid | Context |
| :--- | :--- | :--- |
| `behavior` | prompt-specific labels | General action description |
| `evidence` | hidden truth | Verified information |
| `candidate` | final answer | Pre-selection states |
| `adapter` | compatibility logic | Boundary interfaces |
| `scaffold` | permanent architecture | Temporary/transitional paths |
| `shared mechanism` | special case | Reusable architectural patterns |

## 4. Architectural Review Checklist
Before finalizing changes, answer:
- [ ] Is this behavior modeled as reusable `data`, `traversal`, `scoring`, `extraction`, `runtime orchestration`, `realization`, or `storage`?
- [ ] Does an older path already do similar work?
- [ ] Does this code make the overall architecture simpler?
- [ ] Does the test pass due to intended behavior classes, rather than special cases?
- [ ] Are remaining fallbacks temporary, isolated, and documented?