"""
WebSearchAdapter — searches via the Tavily API and returns fetched source content.

Requires TAVILY_API_KEY (see .env.example). Tavily crawls and extracts page
content server-side, so no local HTML scraping/parsing is needed.
Results are saved incrementally to sources.json so runs are idempotent.
The full, unfiltered Tavily response for every query is also appended to a
raw JSONL file (one line per query) for later offline analysis.

Usage:
    from pipeline.sources.web_search import WebSearchAdapter
    docs = WebSearchAdapter(out_path=Path("outputs/web_harvest/sources.json")).run()
"""
from __future__ import annotations

import json
import os
import time
import urllib.parse
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests

from .base import SourceDocument

TAVILY_SEARCH_URL = "https://api.tavily.com/search"

# ── Search queries covering the four main discourse clusters ─────────────────

SEARCH_QUERIES: list[tuple[str, str]] = [
    # (query, cluster)

    # detection_evasion
    ("how to make AI images pass deepfake detector post-processing", "detection_evasion"),
    ("bypass AI image detector techniques site:photography OR site:reddit", "detection_evasion"),
    ("AI generated image detection evasion image processing", "detection_evasion"),
    ("undetectable AI image tutorial 2024 2025", "detection_evasion"),

    # realism_framing
    ("make AI images look less artificial photography tips workflow", "realism_framing"),
    ("remove AI sheen look stable diffusion photography", "realism_framing"),
    ("make stable diffusion image look like real photograph", "realism_framing"),
    ("midjourney photorealistic tips post processing", "realism_framing"),
    ("how to make AI art look like real photo tutorial", "realism_framing"),

    # capture_pipeline_film
    ("add film grain analog look AI generated image", "capture_pipeline_film"),
    ("film emulation analog photography AI image Lightroom", "capture_pipeline_film"),
    ("Portra 400 film simulation digital photo realistic", "capture_pipeline_film"),
    ("halation light leak film overlay photography realistic", "capture_pipeline_film"),
    ("matte lifted blacks fade film look photography", "capture_pipeline_film"),

    # capture_pipeline_phone
    ("phone camera filter realistic AI image mobile ISP processing", "capture_pipeline_phone"),
    ("iPhone camera noise shadow processing realistic photo", "capture_pipeline_phone"),
    ("mobile camera processing pipeline noise compression artifacts", "capture_pipeline_phone"),

    # capture_pipeline_sensor
    ("camera sensor noise PRNU fingerprint photography forensics", "capture_pipeline_sensor"),
    ("sensor noise pattern real photograph vs AI image", "capture_pipeline_sensor"),
    ("Poisson Gaussian sensor noise model photography", "capture_pipeline_sensor"),

    # capture_pipeline_recapture
    ("print scan recapture AI image realistic photograph", "capture_pipeline_recapture"),
    ("photograph printed AI image realistic laundering", "capture_pipeline_recapture"),
    ("recaptured screen image moire pattern forensics", "capture_pipeline_recapture"),

    # compression
    ("JPEG compression artifacts camera realistic photo AI", "compression"),
    ("JPEG quantization table camera identification forensics", "compression"),
    ("JPEG ringing blocking artifacts authentic photograph", "compression"),

    # signal_texture
    ("high frequency texture noise floor real photograph vs AI", "signal_texture"),
    ("frequency separation skin pore texture overlay photography", "post_processing_texture"),
    ("fix plastic AI headshot restore skin pores Photoshop", "post_processing_texture"),
    ("skin texture frequency separation retouching realistic", "post_processing_texture"),

    # frequency
    ("unsharp mask high-pass filter photography realistic sharpness", "frequency"),
    ("detail enhancement clarity texture boost photography", "frequency"),
    ("dithering noise floor photography forensics", "frequency"),

    # optical
    ("chromatic aberration lens fringing photography realistic", "optical"),
    ("barrel distortion lens geometry photography realistic", "optical"),
    ("vignetting lens corner darkening photography authentic", "optical"),
    ("depth of field blur aperture photography realistic AI", "optical"),

    # post_processing_color / tone
    ("white balance temperature tint realistic photograph correction", "post_processing_color"),
    ("shadow highlight recovery photography tone realistic", "post_processing_color"),
    ("color grading S-curve split toning cinema photography", "post_processing_color"),
    ("desaturation HDR tone mapping photography realistic", "post_processing_color"),

    # compositional
    ("Poisson blending color harmonization composite photography", "compositional"),
    ("light harmonization compositing realistic shadows", "compositional"),

    # post_processing_resample
    ("resample upscale downscale chain image processing photography", "post_processing_resample"),
    ("ComfyUI upscale tile img2img relaundering workflow", "post_processing_resample"),

    # detection_evasion (spectral / adversarial)
    ("GAN fingerprint spectral artifacts detection AI image forensics", "detection_evasion_spectral"),
    ("adversarial perturbation fool deepfake detector FGSM imperceptible", "detection_evasion_adversarial"),
    ("remove invisible AI watermark SynthID stable signature", "detection_evasion_watermark"),
    ("EXIF metadata spoofing camera fingerprint injection forensics", "metadata_spoofing"),

]

# Platform-restricted queries. Run with
# `adapter.run(queries=X_TWITTER_QUERIES, include_domains=["x.com", "twitter.com"])`.
X_TWITTER_QUERIES: list[tuple[str, str]] = [
    ("AI generated image look realistic photography tips", "x_twitter"),
    ("photorealistic prompt trick stable diffusion midjourney", "x_twitter"),
    ("how I made this AI image look real workflow", "x_twitter"),
    ("undetectable AI image technique post-processing", "x_twitter"),
    ("remove AI look plastic skin realistic photo prompt", "x_twitter"),
    ("film grain analog look AI image workflow", "x_twitter"),
]

SKIP_DOMAINS = {
    "reddit.com", "youtube.com", "youtu.be",
    "facebook.com",
    "instagram.com", "tiktok.com", "linkedin.com",
    "amazon.com", "ebay.com", "etsy.com",
    "wikipedia.org",  # too generic
}


def _tavily_search(
    query: str,
    max_results: int = 6,
    max_chars: int = 8000,
    include_domains: list[str] | None = None,
) -> tuple[list[dict], dict]:
    """POST to the Tavily search API. Returns ([{url, title, content}], raw_response_json).

    `include_domains`, when given, is an explicit allowlist passed straight to Tavily
    and also overrides SKIP_DOMAINS for exactly those domains (e.g. reddit.com is in
    SKIP_DOMAINS for the general web crawl, but a caller doing a dedicated Reddit pass
    wants it back).
    """
    api_key = os.environ.get("TAVILY_API_KEY")
    if not api_key:
        raise RuntimeError("TAVILY_API_KEY not set (see .env.example)")

    payload = {
        "query": query,
        "max_results": max_results,
        "include_raw_content": "text",
    }
    if include_domains:
        payload["include_domains"] = include_domains

    try:
        resp = requests.post(
            TAVILY_SEARCH_URL,
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=30,
        )
        resp.raise_for_status()
    except Exception as e:
        print(f"    [tavily] search error: {e}")
        return [], {}

    raw = resp.json()
    allowed = {d.removeprefix("www.") for d in (include_domains or [])}
    results = []
    for item in raw.get("results", []):
        url = item.get("url", "")
        if not url.startswith("http"):
            continue
        domain = urllib.parse.urlparse(url).netloc.removeprefix("www.")
        if domain not in allowed and any(d in domain for d in SKIP_DOMAINS):
            continue
        content = (item.get("raw_content") or item.get("content") or "")[:max_chars]
        results.append({"url": url, "title": item.get("title", ""), "content": content})
    return results, raw


class WebSearchAdapter:
    """
    Auto-search Tavily and fetch pages for wild-knowledge collection.

    Results are saved incrementally to `out_path` (dedup by URL) so
    re-running only fetches pages not already present. The full raw
    Tavily response for every query (including thin/skipped hits) is
    appended to `raw_path` as JSONL, for offline analysis independent
    of the deduplicated sources.json.

    Usage::

        adapter = WebSearchAdapter(out_path=Path("outputs/web_harvest/sources.json"))
        docs = adapter.run(per_query=5)
    """

    def __init__(
        self,
        out_path: Path,
        sleep_search: float = 2.0,
        raw_path: Optional[Path] = None,
    ):
        self.out_path = Path(out_path)
        self.sleep_search = sleep_search
        self.raw_path = Path(raw_path) if raw_path else self.out_path.parent / "raw_tavily_responses.jsonl"

    def _load_existing(self) -> dict[str, dict]:
        if self.out_path.exists():
            try:
                records = json.loads(self.out_path.read_text(encoding="utf-8"))
                return {r["url"]: r for r in records if r.get("url")}
            except Exception:
                return {}
        return {}

    def _save(self, records: dict[str, dict]) -> None:
        self.out_path.parent.mkdir(parents=True, exist_ok=True)
        self.out_path.write_text(
            json.dumps(list(records.values()), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    def _append_raw(self, query: str, cluster: str, raw: dict) -> None:
        if not raw:
            return
        self.raw_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "query": query,
            "cluster": cluster,
            "collected_at": datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
            "response": raw,
        }
        with self.raw_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def run(
        self,
        queries: list[tuple[str, str]] = SEARCH_QUERIES,
        per_query: int = 5,
        verbose: bool = True,
        include_domains: list[str] | None = None,
    ) -> list[SourceDocument]:
        """
        Search each query, fetch new pages, and return all SourceDocuments.
        Already-fetched pages are returned from cache.
        """
        existing = self._load_existing()
        added = 0
        all_docs: list[SourceDocument] = []

        for query, cluster in queries:
            if verbose:
                print(f"[web_search] [{cluster}] {query[:65]}")
            hits, raw = _tavily_search(query, max_results=per_query, include_domains=include_domains)
            self._append_raw(query, cluster, raw)
            time.sleep(self.sleep_search)

            for hit in hits:
                url = hit["url"]
                if url in existing:
                    rec = existing[url]
                    all_docs.append(SourceDocument(
                        url=rec["url"],
                        title=rec.get("title", ""),
                        content=rec.get("content", ""),
                        platform="web",
                        discourse_cluster=rec.get("cluster", cluster),
                        collected_at=rec.get("collected_at", ""),
                    ))
                    continue

                content = hit["content"]
                if not content or len(content) < 300:
                    if verbose:
                        print(f"    skip (thin content): {url[:60]}")
                    continue

                now = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
                rec = {
                    "url": url,
                    "title": hit["title"],
                    "content": content,
                    "cluster": cluster,
                    "collected_at": now,
                    "source_query": query,
                }
                existing[url] = rec
                added += 1
                if verbose:
                    print(f"    + {url[:60]}")

                all_docs.append(SourceDocument(
                    url=url,
                    title=hit["title"],
                    content=content,
                    platform="web",
                    discourse_cluster=cluster,
                    collected_at=now,
                ))

            self._save(existing)

        if verbose:
            print(f"[web_search] done: +{added} new pages, {len(all_docs)} total docs")
        return all_docs


def main():
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Auto web-search and fetch wild-knowledge sources")
    parser.add_argument("--out", default="outputs/web_harvest/sources.json")
    parser.add_argument("--per-query", type=int, default=5,
                        help="Max results per search query (default 5)")
    parser.add_argument("--sleep-search", type=float, default=2.0)
    args = parser.parse_args()

    adapter = WebSearchAdapter(
        out_path=Path(args.out),
        sleep_search=args.sleep_search,
    )
    docs = adapter.run(per_query=args.per_query)
    print(f"\nTotal documents collected: {len(docs)}")


if __name__ == "__main__":
    main()
