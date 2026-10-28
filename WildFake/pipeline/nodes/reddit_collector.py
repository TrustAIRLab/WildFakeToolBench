"""
Node 1: RedditCollector

Crawls Reddit for posts about making AI images pass detection. Two discourse
clusters are queried (technical "bypass detector" vocabulary + colloquial
"look less AI" vocabulary). Posts are filtered to ensure topic relevance and
discussion volume, then dedup'd and ranked.

Uses Reddit's official OAuth API (oauth.reddit.com) with an app-only
(client_credentials) token — read-only access to public data, no user login
needed. Requires REDDIT_CLIENT_ID + REDDIT_CLIENT_SECRET (see .env.example);
Reddit's unauthenticated www.reddit.com/*.json endpoint blocks datacenter IPs.
"""
from __future__ import annotations

import json
import os
import re
import time
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import List, Optional

USER_AGENT = "WildFake/0.1 (research, +local only)"
TOKEN_URL = "https://www.reddit.com/api/v1/access_token"
API_BASE = "https://oauth.reddit.com"

DEFAULT_SUBS = [
    # core
    "StableDiffusion", "deepfakes",
    # generation communities (rich in realism / prompt tricks)
    "comfyui", "StableDiffusionInfo", "midjourney", "aiArt", "MediaSynthesis",
    "civitai", "FluxAI", "sdforall",
    # photography crossover (capture-pipeline / film-emulation knowledge)
    "AnalogCommunity", "postprocessing",
]
DEFAULT_QUERIES = [
    # Cluster 1 — detection-framing (technical evasion vocabulary)
    "bypass deepfake detector",
    "evade AI detector image",
    "AIGC detector bypass",
    "how to prevent AI image detection",
    "aiornot fooled",
    # Cluster 2 — realism-framing (colloquial "look less AI" vocabulary)
    "make AI images look less AI",
    "AI images look realistic",
    "remove AI look",
    "make AI image look real",
    "get rid of the AI sheen",
    "AI image looks too perfect",
    # Cluster 3 — generative-side realism (prompt / model / img2img tricks)
    "prompt for realistic skin texture",
    "amateur photo prompt realistic",
    "img2img low denoise realism",
    "lora for photorealism realistic",
    "negative prompt remove AI look",
    "add film grain workflow comfyui",
    # Cluster 4 — capture-pipeline / analog laundering
    "photograph print of AI image",
    "phone camera filter AI image realistic",
    "scan film emulation AI image",
]
DEFAULT_RELEVANCE_KEYWORDS = [
    "bypass", "evade", "evading", "undetect", "undetected", "undetectable",
    "detector", "detection", "detect",
    "aiornot", "ai or not",
    "fool", "fooling", "trick",
    "disguise", "classifier", "forensic", "watermark",
    "look less", "look real", "look natural", "look human",
    "less ai", "not look ai", "avoid ai", "remove ai",
    "ai look", "generic ai", "ai-ity", "ai sheen", "too perfect", "plastic",
    "photoreal", "realistic", "amateur", "candid", "snapshot",
    "pass as", "pass for",
    # generative-side vocabulary
    "grain", "film", "texture", "skin", "denoise", "img2img", "lora",
    "negative prompt", "workflow", "checkpoint", "noise",
]


@dataclass
class Comment:
    author: str
    score: int
    body: str


@dataclass
class Post:
    id: str
    title: str
    subreddit: str
    author: str
    score: int
    num_comments: int
    created_utc: float
    permalink: str
    url: str
    selftext: str
    matched_queries: List[str] = field(default_factory=list)
    top_comments: List[Comment] = field(default_factory=list)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["top_comments"] = [asdict(c) for c in self.top_comments]
        return d

    def to_markdown(self) -> str:
        created = time.strftime("%Y-%m-%d", time.gmtime(self.created_utc))
        lines = [
            f"# {self.title}",
            "",
            f"- **Subreddit:** r/{self.subreddit}",
            f"- **Author:** u/{self.author}",
            f"- **Score:** {self.score} | **Comments:** {self.num_comments}",
            f"- **Date (UTC):** {created}",
            f"- **Reddit URL:** https://www.reddit.com{self.permalink}",
            f"- **External URL:** {self.url}",
            f"- **Matched queries:** {', '.join(sorted(self.matched_queries))}",
            "",
            "## Post body",
            "",
            self.selftext or "(no body / external link post)",
            "",
            "## Top comments",
            "",
        ]
        if not self.top_comments:
            lines.append("(no comments)")
        for c in self.top_comments:
            lines.append(f"### u/{c.author} — score {c.score}")
            lines.append("")
            lines.append(c.body or "")
            lines.append("")
        return "\n".join(lines)


def _slugify(s: str, max_len: int = 80) -> str:
    s = re.sub(r"[^\w\s-]", "", s).strip().lower()
    s = re.sub(r"[\s_]+", "-", s)
    return s[:max_len] or "untitled"


def _get_access_token(client_id: str, client_secret: str, timeout: int = 30) -> str:
    """App-only OAuth token (client_credentials grant) — read-only public data, no user login."""
    import base64

    auth = base64.b64encode(f"{client_id}:{client_secret}".encode()).decode()
    req = urllib.request.Request(
        TOKEN_URL,
        data=b"grant_type=client_credentials",
        headers={
            "User-Agent": USER_AGENT,
            "Authorization": f"Basic {auth}",
            "Content-Type": "application/x-www-form-urlencoded",
        },
        method="POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())["access_token"]


def _http_get_json(url: str, token: str, timeout: int = 30) -> dict:
    req = urllib.request.Request(
        url, headers={"User-Agent": USER_AGENT, "Authorization": f"Bearer {token}"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read())


class RedditCollector:
    """Node 1. Crawl Reddit and return a list of `Post` objects."""

    def __init__(
        self,
        subs: Optional[List[str]] = None,
        queries: Optional[List[str]] = None,
        relevance_keywords: Optional[List[str]] = None,
        per_query_limit: int = 50,
        final_limit: int = 10,
        min_comments: int = 10,
        top_comments: int = 5,
        sleep_between: float = 1.2,
        client_id: Optional[str] = None,
        client_secret: Optional[str] = None,
    ):
        self.subs = subs or DEFAULT_SUBS
        self.queries = queries or DEFAULT_QUERIES
        self.relevance_keywords = relevance_keywords or DEFAULT_RELEVANCE_KEYWORDS
        self.per_query_limit = per_query_limit
        self.final_limit = final_limit
        self.min_comments = min_comments
        self.top_comments = top_comments
        self.sleep_between = sleep_between

        client_id = client_id or os.environ.get("REDDIT_CLIENT_ID")
        client_secret = client_secret or os.environ.get("REDDIT_CLIENT_SECRET")
        if not client_id or not client_secret:
            raise RuntimeError(
                "REDDIT_CLIENT_ID / REDDIT_CLIENT_SECRET not set (see .env.example)"
            )
        self._token = _get_access_token(client_id, client_secret)

    def _search(self, query: str) -> List[dict]:
        sub_path = "+".join(self.subs)
        q = urllib.parse.quote(query)
        url = (
            f"{API_BASE}/r/{sub_path}/search"
            f"?q={q}&restrict_sr=on&limit={self.per_query_limit}&sort=relevance&t=all"
        )
        data = _http_get_json(url, self._token)
        return [c["data"] for c in data.get("data", {}).get("children", [])]

    def _fetch_top_comments(self, permalink: str) -> List[Comment]:
        url = f"{API_BASE}{permalink}?limit={self.top_comments}&sort=top&depth=1"
        try:
            data = _http_get_json(url, self._token)
        except Exception:
            return []
        out: List[Comment] = []
        try:
            for c in data[1]["data"]["children"][: self.top_comments]:
                if c["kind"] == "t1":
                    out.append(
                        Comment(
                            author=c["data"].get("author") or "[deleted]",
                            score=int(c["data"].get("score") or 0),
                            body=c["data"].get("body") or "",
                        )
                    )
        except Exception:
            pass
        return out

    def _score_relevance(self, d: dict) -> tuple[int, int]:
        title = (d.get("title") or "").lower()
        body = (d.get("selftext") or "").lower()
        in_title = sum(1 for kw in self.relevance_keywords if kw in title)
        in_body = sum(1 for kw in self.relevance_keywords if kw in body)
        return in_title, in_body

    def run(self, save_dir: Optional[str] = None, verbose: bool = True) -> List[Post]:
        # 1. Multi-query search, dedup by post id, track which queries hit each
        pool: dict[str, tuple[dict, set[str], int]] = {}
        for qi, q in enumerate(self.queries):
            if verbose:
                print(f"[collector] [{qi+1}/{len(self.queries)}] {q!r}")
            try:
                hits = self._search(q)
            except Exception as e:
                if verbose:
                    print(f"  error: {e}")
                continue
            for rank, d in enumerate(hits):
                pid = d.get("id")
                if not pid:
                    continue
                if pid in pool:
                    pool[pid][1].add(q)
                else:
                    pool[pid] = (d, {q}, rank)
            time.sleep(self.sleep_between)

        # 2. Filter: title must contain a relevance keyword AND >= min_comments
        scored = []
        for d, matched, first_rank in pool.values():
            in_title, in_body = self._score_relevance(d)
            if in_title == 0:
                continue
            if int(d.get("num_comments") or 0) < self.min_comments:
                continue
            scored.append((d, matched, first_rank, in_title, in_body))

        if verbose:
            print(
                f"[collector] kept {len(scored)} / {len(pool)} after filters "
                f"(title-keyword + >= {self.min_comments} comments)"
            )

        # 3. Rank: title-keyword hits > #queries matched > #comments > body hits > score
        scored.sort(
            key=lambda x: (
                -x[3], -len(x[1]), -int(x[0].get("num_comments") or 0),
                -x[4], -int(x[0].get("score") or 0), x[2],
            )
        )
        scored = scored[: self.final_limit]

        # 4. Hydrate with top comments + build Post objects
        posts: List[Post] = []
        for d, matched, _, _, _ in scored:
            comments = self._fetch_top_comments(d.get("permalink", ""))
            time.sleep(self.sleep_between)
            posts.append(
                Post(
                    id=d.get("id", ""),
                    title=d.get("title", ""),
                    subreddit=d.get("subreddit", ""),
                    author=d.get("author") or "[deleted]",
                    score=int(d.get("score") or 0),
                    num_comments=int(d.get("num_comments") or 0),
                    created_utc=float(d.get("created_utc") or 0),
                    permalink=d.get("permalink", ""),
                    url=d.get("url_overridden_by_dest") or d.get("url") or "",
                    selftext=d.get("selftext") or "",
                    matched_queries=sorted(matched),
                    top_comments=comments,
                )
            )

        if save_dir:
            self._save(posts, save_dir)
        return posts

    def _save(self, posts: List[Post], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        # JSONL: one line per post, machine-readable
        with open(os.path.join(save_dir, "posts.jsonl"), "w", encoding="utf-8") as f:
            for p in posts:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        # Markdown: human-readable per post + index
        md_dir = os.path.join(save_dir, "markdown")
        os.makedirs(md_dir, exist_ok=True)
        index = ["# Reddit collection", ""]
        for i, p in enumerate(posts, 1):
            fname = f"{i:02d}_{_slugify(p.title)}.md"
            with open(os.path.join(md_dir, fname), "w", encoding="utf-8") as f:
                f.write(p.to_markdown())
            index.append(
                f"{i}. [{p.title}](markdown/{fname}) — r/{p.subreddit}, "
                f"{p.num_comments} comments, score {p.score}"
            )
        with open(os.path.join(save_dir, "INDEX.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(index))


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="outputs/posts", help="output directory")
    parser.add_argument("--limit", type=int, default=10)
    parser.add_argument("--min-comments", type=int, default=10)
    args = parser.parse_args()

    collector = RedditCollector(final_limit=args.limit, min_comments=args.min_comments)
    posts = collector.run(save_dir=args.out)
    print(f"Saved {len(posts)} posts to {args.out}")


if __name__ == "__main__":
    main()
