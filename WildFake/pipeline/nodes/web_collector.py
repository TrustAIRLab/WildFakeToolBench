"""
Node 1c: WebCollector — normalise Step 1's search results into `Post` records.

Reads the source corpus that Step 1 (`sources.web_search.WebSearchAdapter`)
writes from the Tavily search API:

    outputs/web_harvest/sources.json   # [{url, title, cluster, content}, ...]

and converts it into the `Post` data model that Nodes 2 and 3 consume. Makes no
API calls and is fully deterministic.

Each source becomes a Post whose `selftext` holds the retrieved page text;
`subreddit` is set to "web" and `author` to the source domain.

Usage:
  python3 -m nodes.web_collector --sources outputs/web_harvest/sources.json \
                                 --out outputs/posts
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import urllib.parse
from pathlib import Path
from typing import List, Optional

from .reddit_collector import Comment, Post, _slugify


def _source_to_post(rec: dict) -> Post:
    url = rec.get("url", "")
    pid = "web_" + hashlib.md5(url.encode()).hexdigest()[:8]
    domain = urllib.parse.urlparse(url).netloc
    return Post(
        id=pid,
        title=rec.get("title", "") or domain,
        subreddit="web",
        author=domain,
        score=0,
        num_comments=0,
        created_utc=0.0,
        permalink=url,
        url=url,
        selftext=rec.get("content", ""),
        matched_queries=[rec.get("cluster", "")] if rec.get("cluster") else [],
        top_comments=[],
    )


class WebCollector:
    """Node 1c. Ingest the Step 1 source corpus into Post objects."""

    def __init__(self, sources_path: Path):
        self.sources_path = Path(sources_path)

    def run(self, save_dir: Optional[str] = None, verbose: bool = True) -> List[Post]:
        records = json.loads(self.sources_path.read_text(encoding="utf-8"))
        posts = [_source_to_post(r) for r in records if r.get("url")]
        if verbose:
            print(f"[web_collector] ingested {len(posts)} real sources from {self.sources_path}")
            clusters = {}
            for r in records:
                clusters[r.get("cluster", "?")] = clusters.get(r.get("cluster", "?"), 0) + 1
            for c, n in sorted(clusters.items()):
                print(f"    {c}: {n}")
        if save_dir:
            self._save(posts, save_dir)
        return posts

    def _save(self, posts: List[Post], save_dir: str) -> None:
        os.makedirs(save_dir, exist_ok=True)
        # Append/refresh a web_posts.jsonl alongside any reddit posts.jsonl
        with open(os.path.join(save_dir, "web_posts.jsonl"), "w", encoding="utf-8") as f:
            for p in posts:
                f.write(json.dumps(p.to_dict(), ensure_ascii=False) + "\n")
        md_dir = os.path.join(save_dir, "markdown")
        os.makedirs(md_dir, exist_ok=True)
        index = ["# Web collection", ""]
        for i, p in enumerate(posts, 1):
            fname = f"web_{i:02d}_{_slugify(p.title)}.md"
            with open(os.path.join(md_dir, fname), "w", encoding="utf-8") as f:
                f.write(p.to_markdown())
            index.append(f"{i}. [{p.title}]({p.url}) — {p.author}")
        with open(os.path.join(save_dir, "WEB_INDEX.md"), "w", encoding="utf-8") as f:
            f.write("\n".join(index))


def main():
    import argparse
    here = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", default=str(here / "outputs" / "web_harvest" / "sources.json"))
    parser.add_argument("--out", default=str(here / "outputs" / "posts"))
    args = parser.parse_args()
    posts = WebCollector(args.sources).run(save_dir=args.out)
    print(f"Saved {len(posts)} web posts to {args.out}")


if __name__ == "__main__":
    main()
