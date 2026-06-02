# SKILL: Strategic & Novelty Assessment (Grounded Version)

## 1. Core Purpose
This skill evaluates AI retrieval systems, agent architectures, and AI product proposals for:

- technical novelty (real prior-art separation, not stylistic differences)
- enterprise viability under real constraints (latency, cost, compliance)
- degree of differentiation from modern RAG + agent ecosystems
- failure modes under scale, ambiguity, and incomplete knowledge

It explicitly avoids abstract purity tests and instead focuses on what would survive real-world deployment, procurement, and competitive replication.

---

## 2. The Novelty Anchor (Modern RAG Reality Baseline)

All systems must be compared against a **modern baseline stack**, not “simple vector RAG”.

### Commodity / Baseline Systems (2025 reality)
Modern production RAG already includes:
- multi-hop retrieval pipelines
- graph-augmented RAG (GraphRAG-style clustering/summarization)
- rerankers + cross-encoders
- agentic retrieval loops (ReAct-style or tool-based iteration)
- query rewriting + decomposition
- provenance tracking in some enterprise systems
- hybrid lexical + embedding search

### Advanced Systems (differentiation must exceed this baseline)
To be considered meaningfully novel, a system must introduce at least one of:

- new retrieval topology (not just ranking improvements)
- new state model over retrieval (beyond query rewriting)
- new verifiable grounding mechanism tied to structure, not prompts
- measurable improvement in:
  - hallucination reduction
  - evidence traceability
  - multi-hop reasoning accuracy under constraint
  - cost/latency tradeoffs at scale

### Assessment Rule
If a feature:
- depends primarily on LLM prompting behavior
- rebrands known IR techniques (BM25, ANN, reranking, graph traversal)
- or improves UX without structural retrieval changes

→ classify as **commodity augmentation, not architectural novelty**

---

## 3. Commercial Viability Filters

Evaluate proposals using real enterprise constraints.

### 1. Enterprise vs Consumer Fit

**Enterprise systems require:**
- auditability (traceable evidence paths)
- bounded hallucination risk (measurable, not absolute)
- predictable latency + cost curves
- deployment flexibility (on-prem / VPC / hybrid)

**Consumer systems require:**
- conversational flexibility
- open-world knowledge handling
- tolerance for probabilistic responses

**Key constraint:**
Strict grounding systems often improve enterprise value but degrade consumer usability.

---

### 2. Architectural Integrity
Assess whether the system cleanly separates:

- retrieval layer (facts, structure, evidence)
- reasoning layer (aggregation, inference, synthesis)
- generation layer (language rendering)

Flag as risk if:
- reasoning leaks into retrieval
- LLM acts as implicit memory store
- provenance is not preserved through transformations

---

### 3. Compute Efficiency Reality Check
Evaluate:
- dependency on frontier models vs small-model cascades
- cost scaling with graph size / context size
- agent loop inference overhead
- whether retrieval or LLM dominates compute cost

**Modern constraint:**
Enterprise buyers prioritize cost stability over peak capability.

---

## 4. IP & Risk Analysis

### Defensibility Standards
A feature is potentially defensible only if it includes:

- new data structure or graph topology with clear constraints
- measurable retrieval or reasoning advantage at scale
- reproducible algorithm not reducible to:
  - standard graph traversal
  - embedding similarity search
  - reranking pipelines
  - prompt-based reasoning

---

### Operational Risks
Consider:
- graph explosion / traversal cost blowups
- silent degradation under sparse or dense graphs
- feedback loops amplifying retrieval bias
- evaluation loops causing latency cascades

---

### Knowledge Contamination Risk
Check whether:
- LLM outputs are re-ingested as “facts”
- synthetic reasoning becomes pseudo-ground truth
- provenance is lost during aggregation steps

---

## 5. Required Output Format

### Executive Viability Verdict
Single sentence Go / No-Go based on:
- structural novelty
- enterprise viability
- scalability realism

---

### Novelty & Differentiation
Compare against:
- modern RAG systems
- GraphRAG-style architectures
- agentic retrieval systems

Focus on:
- what is structurally new
- what is reimplementation or repackaging

---

### Go-To-Market Impact
Evaluate:
- enterprise vs consumer fit
- procurement constraints
- cost structure realities
- deployment feasibility

---

### IP & Architectural Risk Matrix

| Risk Type | Description | Required Mitigation |
| :--- | :--- | :--- |
| **Technical** | Failure modes under scale (latency, graph explosion, loop instability) | Replace heuristics with bounded, measurable termination criteria |
| **Architectural** | Leakage between retrieval, reasoning, and generation layers | Enforce explicit separation + traceable transformation pipeline |
| **Market/IP** | Low defensibility due to prior art in IR, graph traversal, and agent systems | Focus on measurable system-level advantages (accuracy, cost, auditability) |