"""
ToolSynthesizer — LLM-driven tool generation with automatic promotion.

Key properties:
  1. SSIM gate: tool is rejected if it corrupts the image beyond a floor
     (SSIM < ssim_floor on a 512×512 test image).
  2. Auto-promote: tools that pass validation are written directly to
     tools/<name>.py and registered in tools/_registered_tools.py.
  3. Iterative retry with error feedback: on smoke/SSIM failure the error is
     injected back into the LLM conversation for self-correction (up to
     max_retries rounds per method).
  4. Idempotency: skips methods whose tool already exists in WildFakeTools/tools/.

Usage::

    from pipeline.distillation import ToolSynthesizer
    synth = ToolSynthesizer()
    report = synth.run(ranked_methods)   # list of dicts from ranked_methods.json
"""
from __future__ import annotations

import ast
import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from openai import APIError, OpenAI, RateLimitError

# ── Paths ────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).resolve().parent.parent.parent      # WildFake/
REPO_ROOT = ROOT.parent                                     # artifact root
TOOLS_DIR = REPO_ROOT / "WildFakeTools" / "tools"
REGISTRY = TOOLS_DIR / "_registered_tools.py"
PROMPT_PATH = ROOT / "prompts" / "codegen_prompt.txt"
MANIFEST_PATH = ROOT / "pipeline" / "distillation" / "MANIFEST.json"

# ── Smoke + SSIM test script ─────────────────────────────────────────────────

_SMOKE_SCRIPT = r"""
import sys, json
from pathlib import Path

src  = Path({src!r})
root = Path({root!r})

sys.path.insert(0, str(root))
import base_tool
BaseTool = base_tool.BaseTool

import numpy as np
from PIL import Image

text = src.read_text(encoding="utf-8")
text = "\n".join(l for l in text.splitlines() if "import BaseTool" not in l)
ns = {{"BaseTool": BaseTool}}
exec(compile(text, str(src), "exec"), ns)

tool_cls = [v for v in ns.values()
            if isinstance(v, type) and issubclass(v, BaseTool) and v is not BaseTool]
assert tool_cls, "no BaseTool subclass defined"
tool = tool_cls[0]()
assert isinstance(tool.name, str) and tool.name, "tool.name missing"
assert isinstance(tool.get_param_schema(), dict), "param schema not a dict"

# 512x512 photo-like synthetic test image: smooth low-frequency structure
# (so blur/geometric/texture-matching ops aren't unfairly SSIM-penalized the
# way they would be on pure iid noise, which has no spatial correlation for
# them to preserve) plus mild fine-grain noise on top, like a real photo.
rng = np.random.RandomState(42)
small = rng.rand(24, 24, 3) * 255
base = Image.fromarray(small.astype("uint8"), "RGB").resize((512, 512), Image.BICUBIC)
arr = np.array(base).astype(float) + rng.randn(512, 512, 3) * 8
arr = np.clip(arr, 0, 255).astype("uint8")
img = Image.fromarray(arr, "RGB")
out = tool.apply(img, {{}})
assert isinstance(out, Image.Image), "apply did not return PIL.Image"
out_rgb = out.convert("RGB")
assert out_rgb.size[0] > 0 and out_rgb.size[1] > 0, "output image is empty"

# global SSIM (luminance)
a = np.array(img.convert("L")).astype(float) / 255.0
b = np.array(out_rgb.resize(img.size).convert("L")).astype(float) / 255.0
C1, C2 = 0.01**2, 0.03**2
mu1, mu2 = a.mean(), b.mean()
s1, s2 = a.std(), b.std()
s12 = ((a - mu1) * (b - mu2)).mean()
ssim = float(((2*mu1*mu2 + C1) * (2*s12 + C2)) / ((mu1**2 + mu2**2 + C1) * (s1**2 + s2**2 + C2)))

print(json.dumps({{"ok": True, "name": tool.name, "ssim": round(ssim, 4)}}))
"""


def _strip_fences(s: str) -> str:
    s = s.strip()
    if s.startswith("```"):
        lines = s.split("\n")
        lines = lines[1:] if lines[0].startswith("```") else lines
        if lines and lines[-1].strip().startswith("```"):
            lines = lines[:-1]
        s = "\n".join(lines).strip()
    return s


def _to_camel(name: str) -> str:
    return "".join(p.capitalize() for p in re.split(r"[^a-zA-Z0-9]+", name) if p)


def _file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()[:12]


def _existing_tool_names() -> set[str]:
    """Snake-case names of tools already implemented in WildFakeTools/tools/."""
    names: set[str] = set()
    for f in TOOLS_DIR.glob("*.py"):
        if f.name in ("__init__.py", "base_tool.py") or f.name.startswith("_"):
            continue
        text = f.read_text(encoding="utf-8", errors="replace")
        m = re.search(r'name\s*=\s*["\']([a-z0-9_]+)["\']', text)
        if m:
            names.add(m.group(1))
    return names


# ── Registry helpers ──────────────────────────────────────────────────────────

def _load_registry() -> list[dict]:
    """Read the manifest entries currently registered in _registered_tools.py."""
    if not REGISTRY.exists():
        return []
    text = REGISTRY.read_text(encoding="utf-8")
    # Extract the MANIFEST list embedded as a comment / constant
    m = re.search(r"_V3_MANIFEST\s*=\s*(\[.*?\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1))
        except Exception:
            pass
    return []


def _rebuild_registry(entries: list[dict]) -> None:
    """
    Regenerate tools/_registered_tools.py from the given manifest entries.

    Each entry must have: name, class_name, tool_file (relative to ROOT).
    """
    imports: list[str] = []
    instances: list[str] = []
    all_names: list[str] = []
    for e in entries:
        if e.get("status") != "promoted":
            continue
        mod = Path(e["tool_file"]).stem
        cls = e["class_name"]
        imports.append(f"from .{mod} import {cls}")
        instances.append(f"    {cls}(),  # [auto] {e.get('idea', '')[:60]}")
        all_names.append(f'    "{cls}"')

    manifest_json = json.dumps(entries, ensure_ascii=False)
    lines = [
        '"""',
        "Auto-generated by pipeline/distillation/tool_synthesizer.py — do not edit by hand.",
        "Re-run `python pipeline/pipeline.py --skip-collect --skip-analyze --skip-rank` to rebuild.",
        '"""',
        "from __future__ import annotations",
        "",
        *imports,
        "",
        f"_V3_MANIFEST = {manifest_json}",
        "",
        "__all__ = [",
        *all_names,
        "]",
        "",
        "SYNTHESIZED_TOOLS = [",
        *instances,
        "]",
        "",
    ]
    TOOLS_DIR.mkdir(parents=True, exist_ok=True)
    REGISTRY.write_text("\n".join(lines), encoding="utf-8")


def _ensure_init_imports() -> None:
    """Add SYNTHESIZED_TOOLS import to WildFakeTools/tools/__init__.py if not already present."""
    init = TOOLS_DIR / "__init__.py"
    text = init.read_text(encoding="utf-8")
    if "from ._registered_tools import SYNTHESIZED_TOOLS" in text:
        return
    # Append the registry import + extend get_all_tools
    patch = """
# ── Auto-synthesized tools (managed by pipeline/distillation/tool_synthesizer.py) ──
try:
    from ._registered_tools import SYNTHESIZED_TOOLS as _SYNTHESIZED_TOOLS
except ImportError:
    _SYNTHESIZED_TOOLS = []
"""
    text = text.replace(
        "\ndef get_all_tools(",
        patch + "\ndef get_all_tools(",
    )
    # Extend the return list in get_all_tools
    text = text.replace(
        "    if tool_filter:",
        "    all_tools = all_tools + _SYNTHESIZED_TOOLS\n    if tool_filter:",
    )
    init.write_text(text, encoding="utf-8")


# ── Main synthesizer ──────────────────────────────────────────────────────────

class ToolSynthesizer:
    """
    Generate BaseTool implementations from ranked method recipes,
    validate them, and auto-promote passing tools to WildFakeTools/tools/.
    """

    def __init__(
        self,
        model: str = "gpt-4o-mini",
        temperature: float = 0.3,
        max_tokens: int = 2400,
        max_retries: int = 3,
        ssim_floor: float = 0.25,
        client: Optional[OpenAI] = None,
        prompt_path: Path = PROMPT_PATH,
    ):
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.max_retries = max_retries
        self.ssim_floor = ssim_floor
        self.client = client or OpenAI(timeout=180)
        self.system_prompt = prompt_path.read_text(encoding="utf-8")

    # ── LLM codegen ─────────────────────────────────────────────────────────

    def _codegen(self, method: dict, previous_errors: list[str] = []) -> str:
        spec = {
            "name": method.get("name"),
            "class_name_hint": _to_camel(method.get("name", "")) + "Tool",
            "method_type": method.get("method_type"),
            "idea": method.get("idea") or method.get("description"),
            "mechanism": method.get("mechanism"),
            "implementation_hint": method.get("implementation_hint"),
            "params_sketch": method.get("params_sketch"),
        }
        messages: list[dict] = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": "Implement a BaseTool for this idea:\n"
                                        + json.dumps(spec, ensure_ascii=False, indent=2)},
        ]
        for err in previous_errors:
            messages.append({"role": "assistant", "content": "... (previous attempt)"})
            messages.append({"role": "user", "content": f"That attempt failed with:\n{err}\n\nFix it and return only the corrected Python code."})

        for attempt in range(3):
            try:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    messages=messages,
                )
                return _strip_fences(resp.choices[0].message.content or "")
            except (RateLimitError, APIError) as e:
                time.sleep(2 ** attempt)
                if attempt == 2:
                    raise RuntimeError(f"LLM failed: {e}") from e
        return ""

    # ── Smoke + SSIM test ────────────────────────────────────────────────────

    def _smoke_test(self, path: Path) -> dict:
        script = _SMOKE_SCRIPT.format(src=str(path), root=str(TOOLS_DIR))
        try:
            proc = subprocess.run(
                [sys.executable, "-c", script],
                capture_output=True, text=True, timeout=90,
            )
        except subprocess.TimeoutExpired:
            return {"ok": False, "error": "timeout (>90s)"}
        if proc.returncode != 0:
            err = (proc.stderr or proc.stdout).strip()
            last = err.splitlines()[-1] if err else "nonzero exit"
            return {"ok": False, "error": last, "full_stderr": err[-1200:]}
        try:
            result = json.loads(proc.stdout.strip().splitlines()[-1])
        except Exception:
            return {"ok": False, "error": "no JSON from smoke test"}
        if result.get("ssim", 1.0) < self.ssim_floor:
            return {
                "ok": False,
                "error": f"SSIM {result['ssim']:.3f} below floor {self.ssim_floor} (image over-corrupted)",
                "ssim": result.get("ssim"),
            }
        return result

    # ── Promote ──────────────────────────────────────────────────────────────

    def _promote(self, code: str, name: str, class_name: str) -> Path:
        """Write tool to WildFakeTools/tools/<name>.py and return the path."""
        dest = TOOLS_DIR / f"{name}.py"
        dest.write_text(code, encoding="utf-8")
        return dest

    # ── Main entry point ─────────────────────────────────────────────────────

    def run(
        self,
        methods: list[dict],
        verbose: bool = True,
        min_feasibility: int = 0,
        skip_existing: bool = True,
    ) -> dict:
        """
        Synthesize tools for the given ranked method dicts.

        Returns a report dict: {manifest, promoted, failed, skipped}.
        """
        existing_names = _existing_tool_names() if skip_existing else set()
        registry_entries = _load_registry()
        registry_by_name = {e["name"]: e for e in registry_entries}

        manifest: list[dict] = list(registry_entries)  # carry forward existing
        promoted_this_run: list[str] = []

        for i, m in enumerate(methods, 1):
            name = m.get("name", f"tool_{i}")
            feas = m.get("scores", {}).get("feasibility_v1") or m.get("scores", {}).get("feasibility", 0)
            if int(feas or 0) < min_feasibility:
                if verbose:
                    print(f"[synth] [{i}/{len(methods)}] {name} — skip (feasibility {feas} < {min_feasibility})")
                continue
            if skip_existing and name in existing_names:
                if verbose:
                    print(f"[synth] [{i}/{len(methods)}] {name} — skip (already implemented)")
                continue
            if name in registry_by_name and registry_by_name[name].get("status") == "promoted":
                if verbose:
                    print(f"[synth] [{i}/{len(methods)}] {name} — skip (already registered)")
                continue

            if verbose:
                print(f"[synth] [{i}/{len(methods)}] {name} ({m.get('method_type', '?')})")

            class_name = _to_camel(name) + "Tool"
            entry: dict = {
                "name": name,
                "class_name": class_name,
                "tool_file": f"WildFakeTools/tools/{name}.py",
                "method_type": m.get("method_type"),
                "priority": m.get("priority"),
                "idea": m.get("idea") or m.get("description", ""),
                "status": "failed",
            }

            # Staged file for retries, promoted to the tools directory on success.
            staged = TOOLS_DIR / "_staged" / f"{name}.py"
            staged.parent.mkdir(parents=True, exist_ok=True)

            errors: list[str] = []
            success = False

            for attempt in range(self.max_retries):
                try:
                    code = self._codegen(m, previous_errors=errors)
                    ast.parse(code)
                except SyntaxError as e:
                    err = f"SyntaxError: {e}"
                    errors.append(err)
                    if verbose:
                        print(f"  attempt {attempt+1} ✗ syntax: {e}")
                    continue
                except RuntimeError as e:
                    if verbose:
                        print(f"  attempt {attempt+1} ✗ LLM: {e}")
                    break

                staged.write_text(code, encoding="utf-8")
                result = self._smoke_test(staged)

                if result.get("ok"):
                    dest = self._promote(code, name, class_name)
                    entry.update(
                        status="promoted",
                        code_hash=_file_hash(dest),
                        ssim_floor=round(result.get("ssim", 0.0), 4),
                        validated_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    )
                    promoted_this_run.append(name)
                    success = True
                    if verbose:
                        print(f"  attempt {attempt+1} ✓ promoted → WildFakeTools/tools/{name}.py "
                              f"(SSIM={result.get('ssim', '?'):.3f})")
                    break
                else:
                    err = result.get("error", "unknown")
                    if verbose:
                        print(f"  attempt {attempt+1} ✗ {err}")
                    errors.append(err + "\n" + result.get("full_stderr", ""))

            if not success:
                entry["status"] = "staged_failed"
                if verbose:
                    print(f"  → staged at {staged.relative_to(REPO_ROOT)} (all {self.max_retries} attempts failed)")

            # Update manifest (replace or append)
            if name in registry_by_name:
                for existing_entry in manifest:
                    if existing_entry.get("name") == name:
                        existing_entry.update(entry)
                        break
            else:
                manifest.append(entry)
                registry_by_name[name] = entry

        # Rebuild registry and patch __init__.py
        _rebuild_registry(manifest)
        _ensure_init_imports()

        # Persist manifest
        MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
        MANIFEST_PATH.write_text(
            json.dumps(
                {
                    "generated_at": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
                    "model": self.model,
                    "ssim_floor": self.ssim_floor,
                    "tools": manifest,
                },
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )

        promoted = [e for e in manifest if e.get("status") == "promoted"]
        failed = [e for e in manifest if e.get("status") in ("staged_failed", "failed")]
        skipped = [e for e in manifest if e.get("status") == "skipped"]
        if verbose:
            print(f"\n[synth] summary: {len(promoted_this_run)} promoted this run, "
                  f"{len(promoted)} total promoted, {len(failed)} failed")
        return {
            "manifest": manifest,
            "promoted": promoted,
            "promoted_this_run": promoted_this_run,
            "failed": failed,
            "skipped": skipped,
        }


def select_methods(
    ranked: list[dict],
    top: int = 100,
    min_feasibility: int = 0,
    skip_duplicates: bool = True,
) -> list[dict]:
    """Filter and sort ranked methods for synthesis."""
    out = []
    for m in ranked:
        if skip_duplicates and m.get("duplicates_v1_tool"):
            continue
        feas = m.get("scores", {}).get("feasibility_v1") or m.get("scores", {}).get("feasibility", 0)
        if int(feas or 0) < min_feasibility:
            continue
        out.append(m)
    out.sort(key=lambda m: -float(m.get("priority", 0)))
    return out[:top]


def main():
    import argparse
    from dotenv import load_dotenv

    parser = argparse.ArgumentParser(description="Synthesize tools from ranked methods")
    parser.add_argument("--ranked", default=str(ROOT / "outputs" / "ranked_methods.json"))
    parser.add_argument("--top", type=int, default=50, help="Max methods to synthesize")
    parser.add_argument("--min-feasibility", type=int, default=5)
    parser.add_argument("--ssim-floor", type=float, default=0.25)
    parser.add_argument("--model", default="gpt-4o-mini")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--include-duplicates", action="store_true")
    args = parser.parse_args()

    load_dotenv(ROOT / ".env")
    ranked = json.loads(Path(args.ranked).read_text(encoding="utf-8")).get("methods", [])
    methods = select_methods(
        ranked, top=args.top, min_feasibility=args.min_feasibility,
        skip_duplicates=not args.include_duplicates,
    )
    print(f"[synth] selected {len(methods)}/{len(ranked)} methods")
    synth = ToolSynthesizer(model=args.model, ssim_floor=args.ssim_floor, max_retries=args.max_retries)
    synth.run(methods)


if __name__ == "__main__":
    main()
