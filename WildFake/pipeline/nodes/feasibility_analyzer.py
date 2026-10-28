"""
Node 2: FeasibilityAnalyzer

For each Reddit post collected in Node 1, run an LLM call that:
  - Scores the post's value as a source of attack methods (3 axes + overall)
  - Extracts atomic candidate methods with quoted evidence

Uses gpt-4o-mini per repo convention. Outputs JSONL of PostAnalysis records.
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

# Allow `from nodes.reddit_collector import Post` when run as script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from nodes.reddit_collector import Post  # noqa: E402


PROMPT_PATH = Path(__file__).resolve().parent.parent.parent / "prompts" / "feasibility_prompt.txt"


@dataclass
class CandidateMethod:
    name: str
    quoted_evidence: str
    source: str
    idea: str = ""          # the concept (mechanism-level, not a PIL recipe)
    mechanism: str = ""     # why it makes the image look real / why a detector misses it
    method_type: str = ""   # generative | numeric | hybrid
    description: str = ""    # Falls back to `idea` if empty.


@dataclass
class PostAnalysis:
    post_id: str
    post_title: str
    verdict: str  # valuable | borderline | discard
    scores: dict  # {method_concreteness, implementability, relevance}
    overall_value: int
    rationale: str
    candidate_methods: List[CandidateMethod] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["candidate_methods"] = [asdict(m) for m in self.candidate_methods]
        return d


def _post_to_payload(p: Post, max_comment_chars: int = 1500) -> str:
    """Compact text view of a post for the LLM."""
    parts = [
        f"POST_ID: {p.id}",
        f"TITLE: {p.title}",
        f"SUBREDDIT: r/{p.subreddit}",
        f"NUM_COMMENTS: {p.num_comments}  SCORE: {p.score}",
        "BODY:",
        (p.selftext or "(empty)").strip(),
        "TOP_COMMENTS:",
    ]
    for c in p.top_comments:
        body = (c.body or "").strip()
        if len(body) > max_comment_chars:
            body = body[:max_comment_chars] + "...[truncated]"
        parts.append(f"--- u/{c.author} (score {c.score}):\n{body}")
    return "\n".join(parts)


class FeasibilityAnalyzer:
    """Node 2. Per-post LLM analysis of attack-method value."""

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.2,
        max_retries: int = 3,
        max_tokens: int = 1500,
        timeout: float = 180.0,
        client: Optional[OpenAI] = None,
        prompt_path: Path = PROMPT_PATH,
    ):
        self.model = model
        self.temperature = temperature
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.client = client or OpenAI(timeout=timeout)
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    @staticmethod
    def _strip_json_fences(s: str) -> str:
        s = s.strip()
        if s.startswith("```"):
            # ```json\n...\n```  or  ```\n...\n```
            lines = s.split("\n")
            if lines[0].startswith("```"):
                lines = lines[1:]
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            s = "\n".join(lines).strip()
        return s

    def _call_llm(self, user_payload: str) -> dict:
        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
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
                print(f"  LLM returned invalid JSON; retrying")
                time.sleep(1)
        raise RuntimeError(f"LLM call failed after {self.max_retries} retries: {last_err}")

    def analyze_one(self, post: Post) -> PostAnalysis:
        payload = _post_to_payload(post)
        raw = self._call_llm(payload)
        cands = [
            CandidateMethod(
                name=str(m.get("name", "")),
                quoted_evidence=str(m.get("quoted_evidence", "")),
                source=str(m.get("source", "")),
                idea=str(m.get("idea", "")),
                mechanism=str(m.get("mechanism", "")),
                method_type=str(m.get("method_type", "")),
                description=str(m.get("description", "") or m.get("idea", "")),
            )
            for m in raw.get("candidate_methods", [])
        ]
        return PostAnalysis(
            post_id=post.id,
            post_title=post.title,
            verdict=str(raw.get("verdict", "discard")),
            scores=raw.get("scores", {}) or {},
            overall_value=int(raw.get("overall_value", 0)),
            rationale=str(raw.get("rationale", "")),
            candidate_methods=cands,
        )

    def run(
        self,
        posts: List[Post],
        save_path: Optional[str] = None,
        verbose: bool = True,
    ) -> List[PostAnalysis]:
        analyses: List[PostAnalysis] = []
        for i, p in enumerate(posts, 1):
            if verbose:
                print(f"[analyzer] [{i}/{len(posts)}] {p.id} — {p.title[:60]}")
            try:
                a = self.analyze_one(p)
            except Exception as e:
                print(f"  failed: {e}; skipping")
                continue
            analyses.append(a)
            if verbose:
                print(
                    f"  verdict={a.verdict} value={a.overall_value} "
                    f"methods={len(a.candidate_methods)}"
                )

        if save_path:
            os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
            with open(save_path, "w", encoding="utf-8") as f:
                for a in analyses:
                    f.write(json.dumps(a.to_dict(), ensure_ascii=False) + "\n")
            if verbose:
                print(f"[analyzer] wrote {len(analyses)} analyses to {save_path}")

        return analyses


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--posts", required=True, help="path to posts.jsonl from Node 1")
    parser.add_argument("--out", default="outputs/analyses.jsonl")
    parser.add_argument("--model", default="gpt-4o-mini")
    args = parser.parse_args()

    posts: List[Post] = []
    with open(args.posts, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            comments = d.pop("top_comments", [])
            from nodes.reddit_collector import Comment
            posts.append(Post(**d, top_comments=[Comment(**c) for c in comments]))

    analyzer = FeasibilityAnalyzer(model=args.model)
    analyzer.run(posts, save_path=args.out)


if __name__ == "__main__":
    main()
