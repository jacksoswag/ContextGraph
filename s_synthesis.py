import json
import os
import requests
from constants import LOGICAL_CONNECTORS

INHERENT_NEGATION_RELATIONS = {"without"}
DEFAULT_OLLAMA_CONNECT_TIMEOUT = 5
DEFAULT_OLLAMA_READ_TIMEOUT = 300

class KnowledgeSynthesizer: # Synthesizes arguments into a report using Ollama 3B
    def __init__(self, ollama_url="http://localhost:11434/api/generate", model="llama3.2:3b"):
        self.ollama_url = ollama_url
        self.model = model
        self.relation_index_path = "verb_index.json"
        self.relation_labels = list(LOGICAL_CONNECTORS)
        self.connect_timeout = float(os.getenv("OLLAMA_CONNECT_TIMEOUT", DEFAULT_OLLAMA_CONNECT_TIMEOUT))
        self.read_timeout = float(os.getenv("OLLAMA_READ_TIMEOUT", DEFAULT_OLLAMA_READ_TIMEOUT))

    def _load_relation_labels(self):
        labels = list(LOGICAL_CONNECTORS)
        if not os.path.exists(self.relation_index_path):
            self.relation_labels = labels
            return

        try:
            with open(self.relation_index_path, "r") as f:
                payload = json.load(f)
            
            # Load verbs from the index if available
            verbs = payload.get("verbs", [])
            seen = set(labels)
            for v in verbs:
                if v not in seen:
                    labels.append(v)
                    seen.add(v)
            self.relation_labels = labels
        except Exception as e:
            print(f"[SYNTHESIS] Warning: Could not load verb index: {e}")
            self.relation_labels = labels

    def _relation_label(self, step: str, truth: bool = True) -> str:
        step = step.strip()
        if step.isdigit():
            relation_id = int(step)
            if 0 <= relation_id < len(self.relation_labels):
                step = self.relation_labels[relation_id]
            else:
                return f"relation_{relation_id}"

        # Clean up specific logical connectors for the final report
        if step == "be": step = "is"
        if step == "of" or step == "subset of": step = "part of"
        if step in INHERENT_NEGATION_RELATIONS:
            return step

        if not truth:
            if step == "is": return "is not"
            if step == "part of": return "is not part of"
            if step.endswith("s"): return f"does not {step[:-1]}"
            return f"does not {step}"
        
        if step == "part of": return "is part of"
        return step

    def _source_suffix(self, source) -> str:
        source = str(source or "").strip()
        if not source or source == "unknown":
            return ""
        return f" ({source})"

    def _truncate_structured_chain(self, chain, max_steps=20):
        if not chain or not isinstance(chain[0], dict) or len(chain) <= max_steps + 1:
            return chain

        start_index = len(chain) - max_steps
        previous = chain[start_index - 1]
        if start_index > 1:
            start_node = previous.get("predicate") or previous.get("target")
        else:
            start_node = chain[0].get("node")
        return [{"node": start_node}] + chain[start_index:]

    def _format_argument(self, chain) -> str:
        if isinstance(chain, str):
            return chain.strip()
        if not chain:
            return ""

        if isinstance(chain[0], dict):
            chain = self._truncate_structured_chain(chain)
            start = str(chain[0].get("node", "")).strip()
            parts = [start] if start else []
            for step in chain[1:]:
                step_truth = bool(step.get("truth", True))
                relation_token = str(step.get("relation_label") or step.get("relation_id", "")).strip()
                relation = self._relation_label(relation_token, truth=step_truth)
                target = str(step.get("predicate", "") or step.get("target", "")).strip()
                if not target:
                    continue
                parts.append(f"{relation} {target}{self._source_suffix(step.get('source'))}".strip())
            return " ".join(part for part in parts if part)

        tokens = [str(token).strip() for token in chain if str(token).strip()]
        if not tokens:
            return ""
        if len(tokens) == 1:
            return tokens[0]

        parts = [tokens[0]]
        i = 1
        while i < len(tokens):
            relation_token = tokens[i]
            i += 1
            truth = True

            if i < len(tokens) and tokens[i].lower() == "not":
                truth = False
                i += 1

            target = tokens[i] if i < len(tokens) else ""
            if target:
                relation = self._relation_label(relation_token, truth=truth)
                parts.append(f"{relation} {target}".strip())
            i += 1
        return " ".join(parts)

    def synthesize(self, target_topic: str, research_goal: str, arguments: list[list[str]]) -> str: # Takes a list of thought chains and generates an explanation
        self._load_relation_labels()
        digest_lines = []
        for i, chain in enumerate(arguments):
            path_str = self._format_argument(chain)
            if path_str:
                digest_lines.append(f"{i+1}. {path_str}")
        digest = "\n".join(digest_lines)

        # Synthesis Prompt
        prompt = f"""
            Write a short research-style synthesis in formal academic prose.

            Research goal: "{research_goal}"
            Research target: "{target_topic}"

            Source-linked notes:
            {digest}

            Requirements:
            - Write as if composing a concise review-paper section, not a chat response.
            - Begin with a direct thesis that answers the research target.
            - Develop the explanation in coherent paragraphs with precise, declarative sentences.
            - Focus on mechanisms, relationships, implications, and points of tension or uncertainty when present.
            - Preserve inline source attributions that already appear in the notes.
            - If the notes are incomplete or conflicting, say so plainly in academic language.
            - Do not invent facts, links, causes, or claims that are not supported by the notes.
            - Do not mention notes, paths, chains, prompts, reasoning steps, or the writing process.
            - Do not use meta phrases such as "the chain of argument", "these notes suggest", "the evidence above", "this analysis", "we can infer", or "based on the provided information".
            - State the substantive claims directly.

            Return only the finished report text.
        """
        try: # Call Ollama
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": True,
                "options": {
                    "temperature": 0.0, # Lower temp for higher factual fidelity
                    "top_p": 0.1
                }
            }

            response = requests.post(
                self.ollama_url,
                json=payload,
                stream=True,
                timeout=(self.connect_timeout, self.read_timeout),
            )
            response.raise_for_status()

            chunks = []
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                data = json.loads(line)
                chunk = data.get("response", "")
                if chunk:
                    chunks.append(chunk)
                if data.get("done"):
                    break

            final_text = "".join(chunks).strip()
            return final_text or "Error: No response from model."

        except requests.exceptions.ConnectionError:
            return "Synthesis Failed: Could not connect to Ollama on localhost:11434. Start Ollama and retry."
        except requests.exceptions.ReadTimeout:
            return f"Synthesis Failed: Ollama did not finish within {int(self.read_timeout)} seconds. Increase OLLAMA_READ_TIMEOUT or use a smaller/faster model."
        except Exception as e:
            return f"Synthesis Failed: {str(e)}"
