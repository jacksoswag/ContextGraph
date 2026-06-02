# SKILL: Behavioral Testing Standards

## 1. Core Philosophy
Tests must assert **observable behavior**, not implementation trivia. Our system (T-RAG) relies on complex, dynamically orchestrated reasoning loops. Brittle unit tests that mock out internal JIT methods or graph structures create massive architectural liabilities.

## 2. The Golden Rule
**DO NOT MOCK INTERNAL GRAPH STRUCTURES OR ROUTING LOGIC.** 
If a test requires you to mock a core abstraction (like how a path is scored or how energy diffuses), you are testing the implementation, not the behavior.

## 3. How to Test
- **Test at the Boundaries:** Feed raw context into the JSON Evaluators (like `TaskEvaluator`) and assert that the correct JSON routing decisions (e.g., `done`, `research`, `remaining`) are emitted.
- **Test End-to-End Traversal:** Seed a lightweight, real in-memory graph. Run the traversal engine and assert that the final extracted `evidence` or `candidate` output is correct.
- **Black-Box Mindset:** The test should not care *how* the engine arrives at the answer, only that the deterministic output matches the expected truth state.

## 4. When to Skip
If a behavior cannot be tested without heavy mocking (e.g., a massive LLM inference chain that cannot run locally in CI), **DO NOT write a brittle mock test**. Instead, document the behavior as unverified in the test suite and escalate to integration/HIL (Human-In-The-Loop) testing.
