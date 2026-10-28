"""
DatasetBuilder — assembles WildTool-Dataset from pipeline artifacts.

Joins three layers:
  1. Source evidence   (web_harvest/sources.json  +  posts.jsonl)
  2. Discovered methods (ranked_methods.json — from FeasibilityAnalyzer + MethodRanker)
  3. Implementation     (tools/*.py  +  MANIFEST.json)

Detector evaluation is not part of this artifact; the dataset records
provenance and implementation only.

Output:
  WildTool-Dataset/v<version>/wildtool_dataset.json   (machine-readable)
  WildTool-Dataset/v<version>/wildtool_dataset.md     (human-readable cards)

Usage::

    from pipeline.dataset import DatasetBuilder
    dataset = DatasetBuilder().build(version="1.0")
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

from ..schema import (
    DiscoveryRecord,
    ImplementationRecord,
    SourceRecord,
    WildToolDataset,
    WildToolEntry,
)

ROOT = Path(__file__).resolve().parent.parent.parent      # WildFake/
REPO_ROOT = ROOT.parent                                     # artifact root
OUTPUTS = ROOT / "outputs"
TOOLS_DIR = REPO_ROOT / "WildFakeTools" / "tools"

GENERATIVE_TOOLS = {
    "surface_naturalize", "background_realism", "iterative_denoise_pass",
}


def _norm(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", (name or "").lower()).strip("_")


def _file_hash(path: Path) -> str:
    try:
        return hashlib.sha256(path.read_bytes()).hexdigest()[:12]
    except Exception:
        return ""


# ── Layer loaders ─────────────────────────────────────────────────────────────

def _load_sources() -> dict[str, dict]:
    """Map post_id → source record, joined from stage-2 posts and stage-1 harvest.

    Each record has ``url``, ``title``, ``cluster`` and ``collected_at`` (any
    of which may be empty). Returns {} when the intermediate outputs are absent.
    """
    harvest_by_url: dict[str, dict] = {}
    web_harvest = OUTPUTS / "web_harvest" / "sources.json"
    if web_harvest.exists():
        for r in json.loads(web_harvest.read_text(encoding="utf-8")):
            if r.get("url"):
                harvest_by_url[r["url"]] = r

    by_post_id: dict[str, dict] = {}
    posts_file = OUTPUTS / "posts" / "web_posts.jsonl"
    if posts_file.exists():
        with posts_file.open(encoding="utf-8") as f:
            for line in f:
                try:
                    d = json.loads(line)
                except Exception:
                    continue
                pid = d.get("id") or ""
                url = d.get("permalink") or d.get("url") or ""
                if not pid:
                    continue
                h = harvest_by_url.get(url, {})
                queries = d.get("matched_queries") or []
                by_post_id[pid] = {
                    "url": url,
                    "title": d.get("title", "") or h.get("title", ""),
                    "cluster": h.get("cluster", "") or (queries[0] if queries else ""),
                    "collected_at": h.get("collected_at", ""),
                }
    return by_post_id


def _load_ranked_methods() -> list[dict]:
    """Merge ranked methods from the pipeline outputs directory."""
    seen: dict[str, dict] = {}
    for path in [
        OUTPUTS / "ranked_methods.json",
    ]:
        if not path.exists():
            continue
        data = json.loads(path.read_text(encoding="utf-8"))
        for m in data.get("methods", []):
            key = _norm(m.get("duplicates_v1_tool") or m.get("name", ""))
            if key not in seen:
                seen[key] = m
    return list(seen.values())


def _load_tool_registry() -> dict[str, dict]:
    """Map normalized tool name → {tool_name, tool_file, description, class_name}."""
    reg: dict[str, dict] = {}
    for f in sorted(TOOLS_DIR.glob("*.py")):
        if f.name in ("__init__.py", "base_tool.py") or f.name.startswith("_"):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'name\s*=\s*["\']([a-z0-9_]+)["\']', text)
        if not m:
            continue
        tool_name = m.group(1)
        d = re.search(r'description\s*=\s*\(?\s*["\']([^"\']{0,300})', text)
        cls = re.search(r'class\s+(\w+Tool)\s*\(', text)
        reg[_norm(tool_name)] = {
            "tool_name": tool_name,
            "tool_file": str(f.relative_to(REPO_ROOT)),
            "description": (d.group(1).strip() if d else ""),
            "class_name": (cls.group(1) if cls else _norm(tool_name).title() + "Tool"),
            "code_hash": _file_hash(f),
        }
    return reg


def _load_manifest() -> dict[str, dict]:
    """Load the synthesizer manifest for synthesis provenance."""
    manifest_path = ROOT / "pipeline" / "distillation" / "MANIFEST.json"
    if not manifest_path.exists():
        return {}
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    return {_norm(e["name"]): e for e in data.get("tools", [])}


# ── Entry construction ────────────────────────────────────────────────────────

def _make_entry(
    key: str,
    name: str,
    disc: Optional[dict],
    reg: Optional[dict],
    v3man: Optional[dict],
    posts: Optional[dict[str, dict]] = None,
) -> WildToolEntry:

    # ── Discovery ──
    scores = (disc or {}).get("scores", {})
    feas = scores.get("feasibility_v1") or scores.get("feasibility") or 0.0
    eff = scores.get("expected_efficacy") or scores.get("efficacy") or 0.0
    nov = scores.get("novelty_vs_v1") or scores.get("novelty") or 0.0
    prio = (disc or {}).get("priority") or 0.0

    evidence_post_ids = (disc or {}).get("evidence_post_ids", [])
    evidence_quotes = (disc or {}).get("evidence_quotes", [])
    sources: list[SourceRecord] = []
    for pid, quote in zip(evidence_post_ids, evidence_quotes + [""] * len(evidence_post_ids)):
        platform = "web" if pid.startswith(("web_", "blog_")) else "reddit"
        src = (posts or {}).get(pid, {})
        sources.append(SourceRecord(
            platform=platform,
            url=src.get("url") or pid,   # falls back to the post_id when stage-2 outputs are absent
            document_title=src.get("title", ""),
            discourse_cluster=src.get("cluster", ""),
            evidence_quote=quote,
            collected_at=src.get("collected_at", ""),
        ))

    discovery = DiscoveryRecord(
        idea=(disc or {}).get("idea", "") or (disc or {}).get("description", ""),
        mechanism=(disc or {}).get("mechanism", ""),
        method_type=_method_type(key, (disc or {}).get("method_type", "")),
        sub_category=(disc or {}).get("sub_category", ""),
        feasibility_score=float(feas),
        efficacy_score=float(eff),
        novelty_score=float(nov),
        priority=float(prio),
        implementation_hint=(disc or {}).get("implementation_hint", ""),
        params_sketch=(disc or {}).get("params_sketch", {}),
        sources=sources,
    )

    # ── Implementation ──
    synthesis = ""
    ssim_fl = 0.0
    validated_at = ""
    if v3man:
        synthesis = "llm_auto" if v3man.get("status") == "promoted" else "llm_assisted"
        ssim_fl = float(v3man.get("ssim_floor", 0.0))
        validated_at = v3man.get("validated_at", "")

    impl_status = "implemented" if reg else ("staged" if v3man else "sketch")
    impl = ImplementationRecord(
        status=impl_status,
        tool_name=(reg or {}).get("tool_name", name),
        tool_file=(reg or {}).get("tool_file", ""),
        class_name=(reg or {}).get("class_name", ""),
        synthesis_method=synthesis,
        code_hash=(reg or {}).get("code_hash", ""),
        smoke_test_passed=impl_status == "implemented",
        ssim_floor=ssim_fl,
        validated_at=validated_at,
    )

    description = (reg or {}).get("description", "") or (disc or {}).get("description", "")

    return WildToolEntry(
        name=name,
        description=description,
        discovery=discovery,
        implementation=impl,
    )


def _method_type(key: str, declared: str) -> str:
    if declared in ("generative", "numeric", "hybrid"):
        return declared
    if key in {_norm(t) for t in GENERATIVE_TOOLS}:
        return "generative"
    return "numeric"


# ── Builder ───────────────────────────────────────────────────────────────────

class DatasetBuilder:
    def __init__(self, out_root: Path = ROOT / "WildTool-Dataset"):
        self.out_root = Path(out_root)

    def build(self, version: str = "1.0", verbose: bool = True) -> WildToolDataset:
        if verbose:
            print("[dataset] loading artifacts …")

        registry = _load_tool_registry()
        discovered = _load_ranked_methods()
        manifest_entries = _load_manifest()
        posts = _load_sources()

        disc_by_key: dict[str, dict] = {}
        for m in discovered:
            key = _norm(m.get("duplicates_v1_tool") or m.get("name", ""))
            disc_by_key.setdefault(key, m)

        entries: list[WildToolEntry] = []
        seen: set[str] = set()

        def add(key: str, name: str, disc: Optional[dict]) -> None:
            if key in seen:
                return
            seen.add(key)
            entries.append(_make_entry(
                key=key,
                name=name,
                disc=disc,
                reg=registry.get(key),
                v3man=manifest_entries.get(key),
                posts=posts,
            ))

        # 1. Every implemented tool is an entry
        for key, reg in registry.items():
            add(key, reg["tool_name"], disc_by_key.get(key))

        # 2. Discovered-but-not-yet-implemented methods become sketch entries
        for m in discovered:
            key = _norm(m.get("duplicates_v1_tool") or m.get("name", ""))
            add(key, m.get("name", key), m)

        # 3. Staged (smoke-failed) tools not yet covered
        for key, entry in manifest_entries.items():
            add(key, entry.get("name", key), disc_by_key.get(key))

        # Sort: discovery priority desc → name
        def sort_key(e: WildToolEntry):
            return (-e.discovery.priority, e.name)

        entries.sort(key=sort_key)
        for i, e in enumerate(entries, 1):
            e.id = f"wildtool-v{version.split('.')[0]}-{i:04d}"

        # Counts
        counts = {
            "total": len(entries),
            "implemented": sum(1 for e in entries if e.implementation.status == "implemented"),
            "discovered": sum(1 for e in entries if e.discovery.sources),
            "by_method_type": {},
            "by_synthesis": {},
        }
        for e in entries:
            mt = e.discovery.method_type or "unknown"
            counts["by_method_type"][mt] = counts["by_method_type"].get(mt, 0) + 1
            sm = e.implementation.synthesis_method or "unknown"
            counts["by_synthesis"][sm] = counts["by_synthesis"].get(sm, 0) + 1

        dataset = WildToolDataset(
            generated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            counts=counts,
            entries=entries,
        )

        # Write outputs
        out_dir = self.out_root / f"v{version}"
        out_dir.mkdir(parents=True, exist_ok=True)

        json_out = out_dir / "wildtool_dataset.json"
        json_out.write_text(
            dataset.model_dump_json(indent=2),
            encoding="utf-8",
        )
        md_out = out_dir / "wildtool_dataset.md"
        _write_markdown(dataset, md_out)

        if verbose:
            print(f"[dataset] {len(entries)} entries → {json_out}")
            print(f"[dataset] counts: {counts}")

        return dataset


def _write_markdown(dataset: WildToolDataset, path: Path) -> None:
    c = dataset.counts
    lines = [
        "# WildTool-Dataset",
        "",
        f"*Generated {dataset.generated_at} · schema v{dataset.schema_version}*",
        "",
        f"**{c['total']}** entries — "
        f"**{c['implemented']}** implemented · "
        f"**{c['discovered']}** discovered",
        "",
        f"Method types: {c['by_method_type']}",
        f"Synthesis: {c['by_synthesis']}",
        "",
        "| ID | Name | Type | Synth | Status |",
        "|----|------|------|-------|--------|",
    ]

    for e in dataset.entries:
        lines.append(
            f"| {e.id} | {e.name} | {e.discovery.method_type} "
            f"| {e.implementation.synthesis_method or '—'} "
            f"| {e.implementation.status} |"
        )

    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Build WildTool-Dataset from pipeline artifacts")
    parser.add_argument("--version", default="1.0")
    parser.add_argument("--out", default=str(ROOT / "WildTool-Dataset"))
    args = parser.parse_args()

    builder = DatasetBuilder(out_root=Path(args.out))
    builder.build(version=args.version)


if __name__ == "__main__":
    main()
