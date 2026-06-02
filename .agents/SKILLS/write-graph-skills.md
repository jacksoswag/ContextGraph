---
name: write-graph-skills
description: Author or convert SKILL documents optimized for ingestion into the knowledge graph. Use when creating new domain methodology SKILLs, converting existing .md skill files, or auditing SKILL documents for ingestion fidelity. The output documents are ingested by an NLP pipeline that extracts Subject-Verb-Object triples — all content MUST conform to that constraint.
---

# Writing Graph-Ingestible SKILLs

Graph SKILLs are methodology documents that get extracted into the knowledge graph as nodes and edges. When a user query matches the SKILL's domain, the retrieval engine surfaces the methodology as context for the solver, reducing iteration depth and improving answer quality.

## Critical Constraint: The Extraction Pipeline

The ingestion pipeline parses text using spaCy dependency parsing. It extracts edges by finding:
- A **subject** (`nsubj` dependency, must be NOUN or PROPN)
- A **verb** (ROOT, relcl, advcl, ccomp, xcomp)
- An **object** (`dobj` or `attr`, must be NOUN or PROPN)

### Sentences That Produce Graph Edges

Every sentence MUST have an explicit noun subject, a verb, and a noun object. These patterns work:

```
"The triage process involves checking index coverage."
  → triage process —[involve]→ index coverage ✓

"Database optimization requires analyzing execution plans."
  → database optimization —[require]→ execution plan ✓

"The system identifies query patterns through explain analysis."
  → system —[identify]→ query pattern ✓
```

### Sentences That Get SILENTLY DROPPED

These patterns produce zero graph content:

```
"Check the index coverage."
  → No subject → DROPPED ✗

"Index coverage is important."
  → "important" is ADJ, not NOUN → DROPPED ✗

"First, analyze the plan."
  → Imperative, no subject → DROPPED ✗

"There are many ways to optimize."
  → Expletive "there", no real subject → DROPPED ✗

"It is recommended to check indexes."
  → "it" is pronoun, not NOUN → DROPPED ✗
```

### The Core Rule

**NEVER write imperatives. NEVER write adjective-predicate sentences. ALWAYS use [Noun Subject] [Verb] [Noun Object] structure.**

Transform imperatives into declarative SVO:

| ✗ Imperative (dropped) | ✓ Declarative (extracted) |
|:---|:---|
| Check index coverage | The analyst checks index coverage |
| Identify slow queries | Performance triage identifies slow queries |
| Compare both approaches | The comparison examines both approaches |
| Look for N+1 patterns | The review detects N+1 patterns |
| Consider caching | The optimization strategy considers caching |
| Run the benchmark | The evaluation runs the benchmark |

## SKILL Document Structure

### Header Block

```markdown
# [Domain Name] Methodology

[One declarative sentence stating what this methodology covers and when retrieval should surface it. Use domain-specific nouns that overlap with likely user queries.]
```

The title and opening sentence are critical for BM25 retrieval seeding. Use the exact terms a user would type when asking about this domain.

### Methodology Body

Write each step as a declarative paragraph. Each paragraph should produce 2-5 graph edges.

```markdown
## [Step Name]

[Declarative sentences with explicit SVO structure. Each sentence becomes one or more graph edges. Use domain-specific nouns as subjects and objects.]
```

### Connecting Steps

Steps should reference each other by name so the graph creates edges between methodology nodes:

```markdown
The query analysis phase feeds results into the index audit phase.
The index audit produces recommendations for the optimization phase.
```

This creates edges: `query analysis —[feed]→ index audit —[produce]→ optimization phase`

### Evidence Requirements

State what kind of evidence satisfies each step:

```markdown
The query analysis step requires execution plan output from the database.
The index audit step requires the current schema definition.
```

### Failure Modes

State what happens when evidence is missing:

```markdown
Missing execution plans indicate the analyst needs database access credentials.
Absent schema definitions require a schema export from the production database.
```

## Retrieval Optimization

### Term Overlap With User Queries

The retrieval engine finds SKILL content via BM25 (keyword matching) and vector similarity (semantic matching). The SKILL document must contain the terms users actually type.

**Good:** "Database query performance troubleshooting examines slow queries."
- User types "my database queries are slow" → BM25 matches "database", "query", "slow"

**Bad:** "Systematic data-layer latency investigation protocols"
- User types "my database queries are slow" → no term overlap, SKILL is never retrieved

### Concept Density

Each sentence should introduce or connect domain concepts. Avoid filler:

**Good:** "The profiler measures query execution time against baseline thresholds."
- Produces: profiler —[measure]→ query execution time

**Bad:** "The next thing that the system does is to take a look at how long things take."
- Vague nouns ("thing", "look"), no useful graph edges

### Specificity vs. Generality

Write at the level of specificity where retrieval is most useful:

- **Too broad:** "The analysis examines the problem." (matches everything, helps nothing)
- **Too narrow:** "The PostgreSQL 16.2 B-tree index advisor checks pg_stat_user_indexes for idx_scan counts below 50." (matches only one exact scenario)
- **Right level:** "The index analysis checks usage statistics for underutilized indexes." (matches the domain, provides actionable guidance)

## Converting Existing SKILL Files

When converting an existing `.md` skill file (e.g., from `.agents/SKILLS/`), apply these transformations:

### Step 1: Strip Non-Extractable Content

Remove or convert:
- YAML frontmatter (not parsed by NLP pipeline)
- Markdown formatting artifacts (headers become paragraph context)
- Code blocks (spaCy does not parse code as natural language)
- Tables (row content is not parsed as sentences)
- Bullet lists with fragments (no SVO structure)

### Step 2: Convert Imperative Lists to Declarative Paragraphs

Source:
```markdown
## Review Checklist
- Runtime errors: Check for potential exceptions
- Performance: Look for N+1 queries
- Security: Scan for injection vulnerabilities
```

Converted:
```markdown
## Code Review Error Detection

The code review process checks the codebase for potential runtime errors including exceptions and null pointer issues. The performance audit detects N+1 query patterns and unbounded operations. The security scan identifies injection vulnerabilities and access control gaps.
```

### Step 3: Convert Code Examples to Declarative Descriptions

Source:
```python
# Bad: N+1 query
for user in users:
    print(user.profile.name)

# Good: Prefetch related
users = User.objects.prefetch_related('profile')
```

Converted:
```markdown
The N+1 pattern occurs when a loop executes a separate database query for each iteration. The prefetch solution uses a single batch query through prefetch_related to load all related objects. The ORM optimization replaces individual queries with a single joined fetch.
```

### Step 4: Ensure Every Paragraph Has Extractable Edges

Read each sentence and verify:
1. There is a noun subject (not a pronoun, not implied)
2. There is a verb
3. There is a noun object (not an adjective, not a clause fragment)

If any sentence fails, rewrite it.

### Step 5: Add Domain Keyword Anchors

Ensure the converted document contains the exact terms users would search for. If the original skill covers "code review" but the converted text only says "the audit process," add explicit anchors:

```markdown
Code review methodology follows a structured triage process. The review examines code changes for correctness and performance.
```

## Quality Checklist

Before finalizing a graph SKILL document, verify:

- [ ] Every sentence has an explicit noun subject
- [ ] Every sentence has a verb and a noun object
- [ ] No imperative sentences exist anywhere in the document
- [ ] No sentences use pronouns (he, she, it, they) as the subject
- [ ] No sentences use adjective predicates without noun objects ("X is important")
- [ ] Domain keywords match terms users would actually type
- [ ] Steps reference each other by name to create inter-step graph edges
- [ ] The opening sentence contains the highest-value retrieval keywords
- [ ] No code blocks, tables, or bullet fragments exist (all converted to prose)
- [ ] Each paragraph produces at least 2 extractable SVO edges

## Example: Full Converted SKILL

### Source (agent-style skill)

```markdown
---
name: code-review
description: Perform code reviews
---

# Code Review

## Checklist
- Look for runtime errors
- Check for N+1 queries
- Verify test coverage
- Flag security issues

## Feedback
- Be polite
- Provide suggestions
- Approve when minor issues remain
```

### Converted (graph-ingestible skill)

```markdown
# Code Review Methodology

Code review methodology provides a structured process for examining code changes. The review covers correctness, performance, security, and test coverage.

## Error Detection

The error detection phase examines code for runtime exceptions and null pointer issues. The reviewer identifies potential out-of-bounds access and unhandled error paths. Exception analysis traces error propagation through the call stack.

## Performance Analysis

The performance analysis detects N+1 query patterns in database access code. The reviewer identifies unbounded operations with quadratic complexity. The audit flags unnecessary memory allocations and redundant computations.

## Security Review

The security review scans code for injection vulnerabilities including SQL injection and cross-site scripting. The reviewer verifies access control enforcement on protected endpoints. The audit detects exposed secrets and insufficient input validation.

## Test Coverage Assessment

The test assessment verifies that business logic has functional test coverage. The reviewer confirms integration tests exist for component interactions. The assessment checks that critical user paths have end-to-end test coverage.

## Review Synthesis

The review synthesis combines findings from error detection, performance analysis, security review, and test assessment into actionable feedback. The reviewer provides specific code suggestions rather than abstract criticism. Minor issues receive approval with inline comments.
```

Each paragraph in the converted version produces 2-3 SVO graph edges. Each step references domain concepts that will match user queries. The methodology structure survives extraction as a connected subgraph.
