"""
End-to-end pipeline: retrieve web discourse, mine manipulation methods, rank and
de-duplicate them, and synthesise each surviving method into a runnable tool.

Steps (all idempotent — reuse cached outputs unless --force):

  Step 1  WebSearchAdapter     → outputs/web_harvest/sources.json
  Step 2  WebCollector         → outputs/posts/web_posts.jsonl
  Step 3  FeasibilityAnalyzer  → outputs/analyses.jsonl
  Step 4  MethodRanker         → outputs/ranked_methods.json
  Step 5  ToolSynthesizer      → tools/<name>.py  (auto-promoted)
  Step 6  DatasetBuilder       → WildTool-Dataset/v<version>/wildtool_dataset.json

Usage::

  # Full run (web search → mine → rank → synthesize → dataset)
  python pipeline/pipeline.py

  # Skip slow steps when outputs already exist
  python pipeline/pipeline.py --skip-collect
  python pipeline/pipeline.py --skip-collect --skip-analyze --skip-rank

  # Just rebuild the dataset from existing tool files (+ outputs/ranked_methods.json if present)
  python pipeline/pipeline.py --only-dataset

  # Limit synthesis scope
  python pipeline/pipeline.py --skip-collect --skip-analyze --skip-rank --top 30 --min-feas 6
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
OUTPUTS = ROOT / "outputs"
sys.path.insert(0, str(ROOT))

load_dotenv(ROOT / ".env")

# ── Step output paths ─────────────────────────────────────────────────────────

SOURCES_JSON = OUTPUTS / "web_harvest" / "sources.json"
ANALYSES     = OUTPUTS / "analyses.jsonl"
RANKED       = OUTPUTS / "ranked_methods.json"


# ── Helpers ───────────────────────────────────────────────────────────────────

def _source_count(path: Path) -> int:
    if not path.exists():
        return 0
    try:
        return len(json.loads(path.read_text(encoding="utf-8")))
    except Exception:
        return 0


def _load_analyses(path: Path) -> list:
    from pipeline.nodes.feasibility_analyzer import CandidateMethod, PostAnalysis
    analyses = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            cm = d.pop("candidate_methods", [])
            analyses.append(
                PostAnalysis(**d, candidate_methods=[CandidateMethod(**c) for c in cm])
            )
    return analyses


def _load_ranked(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8")).get("methods", [])


# ── Pipeline ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace) -> None:
    print("=" * 65)
    print("WildFake Pipeline")
    print("=" * 65)

    # --only-dataset jumps straight to Step 6
    if args.only_dataset:
        print("\n[pipeline] --only-dataset: skipping to Step 6 (DatasetBuilder)")
        _step6(args)
        return

    # ── Step 1: Collect web sources ───────────────────────────────────────────
    if not args.skip_collect:
        print("\n── Step 1: Source collection ──────────────────────────────")
        before = _source_count(SOURCES_JSON)

        from pipeline.sources.web_search import WebSearchAdapter
        WebSearchAdapter(
            out_path=SOURCES_JSON,
            sleep_search=args.sleep_search,
        ).run(per_query=args.per_query)

        after = _source_count(SOURCES_JSON)
        print(f"\n[pipeline] sources: {before} → {after} (+{after - before} new)")
    else:
        print(f"\n[pipeline] --skip-collect: {_source_count(SOURCES_JSON)} existing sources")

    # ── Step 2: Convert sources → Post objects ────────────────────────────────
    print("\n── Step 2: WebCollector (sources → Post objects) ──────────")
    from pipeline.nodes.web_collector import WebCollector
    posts = WebCollector(SOURCES_JSON).run(
        save_dir=str(OUTPUTS / "posts"), verbose=True
    )
    print(f"[pipeline] {len(posts)} posts ready for analysis")

    # ── Step 3: FeasibilityAnalyzer ───────────────────────────────────────────
    if not args.skip_analyze:
        print("\n── Step 3: FeasibilityAnalyzer ────────────────────────────")
        from pipeline.nodes.feasibility_analyzer import FeasibilityAnalyzer
        analyses = FeasibilityAnalyzer(model=args.model).run(
            posts, save_path=str(ANALYSES)
        )
        print(f"[pipeline] {len(analyses)} analyses → {ANALYSES.relative_to(ROOT)}")
    else:
        if not ANALYSES.exists():
            sys.exit(f"[pipeline] ERROR: --skip-analyze but {ANALYSES} not found")
        analyses = _load_analyses(ANALYSES)
        print(f"[pipeline] --skip-analyze: loaded {len(analyses)} analyses")

    # ── Step 4: MethodRanker ─────────────────────────────────────────────────
    if not args.skip_rank:
        print("\n── Step 4: MethodRanker ───────────────────────────────────")
        from pipeline.nodes.method_ranker import MethodRanker
        ranked = MethodRanker(model=args.model).run(
            analyses, save_path=str(RANKED)
        )
        print(f"[pipeline] {len(ranked)} ranked methods → {RANKED.relative_to(ROOT)}")
    else:
        if not RANKED.exists():
            sys.exit(f"[pipeline] ERROR: --skip-rank but {RANKED} not found")
        ranked = _load_ranked(RANKED)
        print(f"[pipeline] --skip-rank: loaded {len(ranked)} ranked methods")

    # ── Step 5: ToolSynthesizer ───────────────────────────────────────────────
    if not args.skip_synth:
        print("\n── Step 5: ToolSynthesizer (codegen + auto-promote) ───────")
        from pipeline.distillation import ToolSynthesizer
        from pipeline.distillation.tool_synthesizer import select_methods

        methods = select_methods(
            ranked,
            top=args.top,
            min_feasibility=args.min_feas,
            skip_duplicates=not args.include_duplicates,
        )
        print(f"[pipeline] selected {len(methods)}/{len(ranked)} methods for synthesis")

        report = ToolSynthesizer(
            model=args.model,
            ssim_floor=args.ssim_floor,
            max_retries=args.max_retries,
        ).run(methods)
        promoted = report["promoted_this_run"]
        print(f"[pipeline] promoted this run: {len(promoted)} — {promoted}")
    else:
        print("\n[pipeline] --skip-synth: skipping tool synthesis")

    # ── Step 6: DatasetBuilder ────────────────────────────────────────────────
    _step6(args)


def _step6(args: argparse.Namespace) -> None:
    print("\n── Step 6: DatasetBuilder ─────────────────────────────────")
    from pipeline.dataset import DatasetBuilder
    dataset = DatasetBuilder().build(version=args.dataset_version)
    print(
        f"\n[pipeline] done. {dataset.counts['total']} entries "
        f"({dataset.counts.get('implemented', 0)} implemented)\n"
        f"           → WildTool-Dataset/v{args.dataset_version}/wildtool_dataset.json"
    )


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )

    # Skip flags
    parser.add_argument("--skip-collect",  action="store_true",
                        help="Reuse existing sources.json (skip web search)")
    parser.add_argument("--skip-analyze",  action="store_true",
                        help="Reuse existing analyses.jsonl")
    parser.add_argument("--skip-rank",     action="store_true",
                        help="Reuse existing ranked_methods.json")
    parser.add_argument("--skip-synth",    action="store_true",
                        help="Skip ToolSynthesizer")
    parser.add_argument("--only-dataset",  action="store_true",
                        help="Shorthand: skip all steps, only rebuild WildTool-Dataset")

    # Collection
    parser.add_argument("--per-query",    type=int,   default=5)
    parser.add_argument("--sleep-search", type=float, default=2.0)

    # Synthesis
    parser.add_argument("--top",          type=int,   default=50,
                        help="Max methods to synthesize (default 50)")
    parser.add_argument("--min-feas",     type=int,   default=5,
                        help="Min feasibility score to synthesize")
    parser.add_argument("--ssim-floor",   type=float, default=0.25,
                        help="Min SSIM for tool promotion (default 0.25)")
    parser.add_argument("--max-retries",  type=int,   default=3)
    parser.add_argument("--include-duplicates", action="store_true")

    # Model + dataset version
    parser.add_argument("--model",           default="gpt-4o-mini")
    parser.add_argument("--dataset-version", default="1.0")

    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
