"""
Node 3: MethodRanker

Takes Node 2 analyses and produces two layers of output:

  1. `methods` — the SPECIFIC/concrete method list. Candidates are batched
     (LLM output-token limits make a single one-shot call over 100+ verbose
     items unreliable), each batch is lightly deduped/classified/scored
     internally, then a second lightweight pass removes any duplicate
     technique that slipped across batch boundaries. This list stays large
     (close to the raw candidate count minus true duplicates) — it is NOT
     collapsed into a handful of generic buckets.

  2. `generalized_methods` — a small set of archetype clusters computed on
     top of the final specific list, purely for reporting/strategy (every
     specific method still stands on its own in `methods`).

Three LLM call types, all gpt-4o-mini: batch extraction, cross-batch dedup,
generalization. See prompts/ranker_*.txt.
"""
from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

from openai import OpenAI
from openai import APIError, RateLimitError

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nodes.feasibility_analyzer import PostAnalysis  # noqa: E402


PROMPTS_DIR = Path(__file__).resolve().parent.parent.parent / "prompts"
PROMPT_PATH = PROMPTS_DIR / "ranker_prompt.txt"
DEDUP_PROMPT_PATH = PROMPTS_DIR / "ranker_dedup_prompt.txt"
GENERALIZE_PROMPT_PATH = PROMPTS_DIR / "ranker_generalize_prompt.txt"


@dataclass
class RankedMethod:
    rank: int
    name: str
    method_type: str = ""        # generative | numeric | hybrid
    sub_category: str = ""        # post_processing | capture_pipeline | prompt_engineering | ...
    idea: str = ""
    mechanism: str = ""
    description: str = ""
    evidence_post_ids: List[str] = field(default_factory=list)
    evidence_quotes: List[str] = field(default_factory=list)
    scores: dict = field(default_factory=dict)   # {novelty, feasibility, expected_efficacy}
    priority: float = 0.0
    duplicates_v1_tool: Optional[str] = None
    implementation_hint: str = ""
    params_sketch: dict = field(default_factory=dict)
    merged_from: List[str] = field(default_factory=list)  # >1 entry iff a cross-batch dup was merged into this one


@dataclass
class GeneralizedMethod:
    rank: int
    name: str
    description: str = ""
    member_names: List[str] = field(default_factory=list)
    priority: float = 0.0


class MethodRanker:
    """Node 3. Batch-extract, cross-batch dedup, score, rank, and cluster candidate methods."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
        max_tokens: int = 8000,
        timeout: float = 300.0,
        client: Optional[OpenAI] = None,
        prompt_path: Path = PROMPT_PATH,
        dedup_prompt_path: Path = DEDUP_PROMPT_PATH,
        generalize_prompt_path: Path = GENERALIZE_PROMPT_PATH,
        batch_size: int = 15,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.client = client or OpenAI(timeout=timeout)
        self.batch_prompt = prompt_path.read_text(encoding="utf-8")
        self.dedup_prompt = dedup_prompt_path.read_text(encoding="utf-8")
        self.generalize_prompt = generalize_prompt_path.read_text(encoding="utf-8")
        self.batch_size = batch_size

    @staticmethod
    def _strip_json_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _build_records(self, analyses: List[PostAnalysis]) -> List[dict]:
        records = []
        idx = 0
        for a in analyses:
            if a.verdict == "discard":
                continue
            for m in a.candidate_methods:
                idx += 1
                records.append(
                    {
                        "cid": f"c{idx}",
                        "post_id": a.post_id,
                        "post_title": a.post_title,
                        "post_value": a.overall_value,
                        "name": m.name,
                        "idea": m.idea or m.description,
                        "mechanism": m.mechanism,
                        "method_type": m.method_type,
                        "evidence": m.quoted_evidence,
                        "source": m.source,
                    }
                )
        return records

    @staticmethod
    def _chunk(records: List[dict], size: int):
        for i in range(0, len(records), size):
            yield records[i : i + size]

    def _call_llm(self, system_prompt: str, user_payload: str, max_tokens: int) -> dict:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=max_tokens,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_payload},
                    ],
                )
                content = self._strip_json_fences(resp.choices[0].message.content or "")
                return json.loads(content)
            except (RateLimitError, APIError) as e:
                last_err = e
                wait = 2 ** attempt
                print(f"  LLM error ({type(e).__name__}); retry in {wait}s")
                time.sleep(wait)
            except json.JSONDecodeError as e:
                last_err = e
                print(f"  invalid JSON; retrying")
                time.sleep(1)
        raise RuntimeError(f"LLM call failed: {last_err}")

    # ── Phase 1: per-batch extraction (conservative intra-batch dedup) ────────

    def _extract_batch(self, batch: List[dict]) -> List[dict]:
        payload = json.dumps({"candidates": batch, "count": len(batch)}, ensure_ascii=False, indent=2)
        raw = self._call_llm(self.batch_prompt, payload, max_tokens=self.max_tokens)
        return raw.get("methods", [])

    # ── Phase 2: cross-batch dedup (literal-duplicate groups only) ───────────

    def _dedup_across_batches(self, methods: List[dict]) -> List[dict]:
        if len(methods) <= 1:
            return methods
        compact = [
            {"id": f"m{i}", "name": m.get("name", ""), "idea": m.get("idea", ""),
             "method_type": m.get("method_type", "")}
            for i, m in enumerate(methods)
        ]
        payload = json.dumps({"methods": compact}, ensure_ascii=False, indent=2)
        raw = self._call_llm(self.dedup_prompt, payload, max_tokens=2000)
        groups = raw.get("duplicate_groups", [])

        id_to_idx = {c["id"]: i for i, c in enumerate(compact)}
        parent = list(range(len(methods)))

        def find(x: int) -> int:
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x

        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra

        for g in groups:
            ids = [id_to_idx[i] for i in g if i in id_to_idx]
            for a, b in zip(ids, ids[1:]):
                union(a, b)

        clusters: dict[int, List[int]] = {}
        for i in range(len(methods)):
            clusters.setdefault(find(i), []).append(i)

        merged: List[dict] = []
        for members in clusters.values():
            if len(members) == 1:
                merged.append(methods[members[0]])
                continue
            members.sort(key=lambda i: -float(methods[i].get("priority", 0) or 0))
            rep = dict(methods[members[0]])
            post_ids, quotes, names = [], [], []
            for i in members:
                post_ids.extend(methods[i].get("evidence_post_ids", []))
                quotes.extend(methods[i].get("evidence_quotes", []))
                names.append(methods[i].get("name", ""))
            rep["evidence_post_ids"] = sorted(set(post_ids))
            rep["evidence_quotes"] = quotes[:5]
            rep["merged_from"] = names
            merged.append(rep)
        return merged

    # ── Phase 3: generalized/archetype clustering (additive, non-destructive) ─

    def _generalize(self, methods: List[dict]) -> List[dict]:
        if not methods:
            return []
        compact = [
            {"name": m.get("name", ""), "idea": m.get("idea", ""),
             "method_type": m.get("method_type", ""), "sub_category": m.get("sub_category", "")}
            for m in methods
        ]
        payload = json.dumps({"methods": compact}, ensure_ascii=False, indent=2)
        raw = self._call_llm(self.generalize_prompt, payload, max_tokens=4000)
        return raw.get("generalized_methods", [])

    def run(
        self,
        analyses: List[PostAnalysis],
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> List[dict]:
        """Rank and de-duplicate methods.

        Returns the specific methods as plain dicts (``asdict`` of
        :class:`RankedMethod`), i.e. exactly the ``methods`` list written to
        ``save_path``, so the in-memory path and ``--skip-rank`` path feed the
        synthesizer the same structure.
        """
        records = self._build_records(analyses)
        if verbose:
            print(f"[ranker] {len(records)} candidate methods")

        specific: List[RankedMethod] = []
        generalized: List[GeneralizedMethod] = []

        if records:
            batches = list(self._chunk(records, self.batch_size))
            all_methods: List[dict] = []
            for bi, batch in enumerate(batches, 1):
                if verbose:
                    print(f"[ranker] batch {bi}/{len(batches)} ({len(batch)} candidates)")
                out = self._extract_batch(batch)
                if verbose:
                    print(f"  → {len(out)} methods (intra-batch dedup)")
                all_methods.extend(out)

            if verbose:
                print(f"[ranker] {len(all_methods)} methods before cross-batch dedup")
            deduped = self._dedup_across_batches(all_methods)
            if verbose:
                n_merged = sum(1 for m in deduped if len(m.get("merged_from", [])) > 1)
                print(f"[ranker] {len(deduped)} methods after cross-batch dedup ({n_merged} were merges)")

            deduped.sort(key=lambda m: -float(m.get("priority", 0) or 0))
            for i, item in enumerate(deduped, 1):
                specific.append(
                    RankedMethod(
                        rank=i,
                        name=str(item.get("name", "")),
                        method_type=str(item.get("method_type", "")),
                        sub_category=str(item.get("sub_category", item.get("category", ""))),
                        idea=str(item.get("idea", "")),
                        mechanism=str(item.get("mechanism", "")),
                        description=str(item.get("description", "") or item.get("idea", "")),
                        evidence_post_ids=list(item.get("evidence_post_ids", [])),
                        evidence_quotes=list(item.get("evidence_quotes", [])),
                        scores=dict(item.get("scores", {})),
                        priority=float(item.get("priority", 0.0) or 0.0),
                        duplicates_v1_tool=item.get("duplicates_v1_tool") or None,
                        implementation_hint=str(item.get("implementation_hint", "")),
                        params_sketch=dict(item.get("params_sketch", {})),
                        merged_from=list(item.get("merged_from", [])),
                    )
                )

            if verbose:
                print("[ranker] generalizing into archetype clusters …")
            gen_raw = self._generalize(deduped)
            # Cluster priority is the maximum of its members' priorities.
            priority_by_name = {m.get("name", ""): float(m.get("priority", 0.0) or 0.0) for m in deduped}
            for item in gen_raw:
                members = list(item.get("member_names", []))
                member_prios = [priority_by_name[n] for n in members if n in priority_by_name]
                item["priority"] = max(member_prios) if member_prios else 0.0
            gen_raw.sort(key=lambda g: -float(g.get("priority", 0) or 0))
            for i, item in enumerate(gen_raw, 1):
                generalized.append(
                    GeneralizedMethod(
                        rank=i,
                        name=str(item.get("name", "")),
                        description=str(item.get("description", "")),
                        member_names=list(item.get("member_names", [])),
                        priority=float(item.get("priority", 0.0) or 0.0),
                    )
                )

        if verbose:
            print(f"[ranker] produced {len(specific)} specific methods, {len(generalized)} generalized clusters")
            for m in specific[:5]:
                print(f"  #{m.rank} [{m.method_type}/{m.sub_category}] {m.name}  prio={m.priority}")
            for g in generalized[:5]:
                print(f"  cluster #{g.rank} {g.name} ({len(g.member_names)} members) prio={g.priority}")

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            payload_out = {
                "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                "model": self.model,
                "source_post_ids": [a.post_id for a in analyses if a.verdict != "discard"],
                "methods": [asdict(m) for m in specific],
                "generalized_methods": [asdict(g) for g in generalized],
            }
            with open(save_path, "w", encoding="utf-8") as f:
                json.dump(payload_out, f, ensure_ascii=False, indent=2)
            if verbose:
                print(f"[ranker] wrote {save_path}")

        return [asdict(m) for m in specific]


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--analyses", required=True, help="path to analyses.jsonl from Node 2")
    parser.add_argument("--out", default="outputs/ranked_methods.json")
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--batch-size", type=int, default=15)
    args = parser.parse_args()

    analyses: List[PostAnalysis] = []
    with open(args.analyses, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cm = d.pop("candidate_methods", [])
            from nodes.feasibility_analyzer import CandidateMethod
            analyses.append(
                PostAnalysis(**d, candidate_methods=[CandidateMethod(**c) for c in cm])
            )

    ranker = MethodRanker(model=args.model, batch_size=args.batch_size)
    ranker.run(analyses, save_path=args.out)


if __name__ == "__main__":
    main()
