# WildFakeTools — 156 Image-Manipulation Tools Distilled from Public Discourse

Anonymised artifact. Every tool is a self-contained `BaseTool` subclass that takes a PIL image plus a parameter dict.

These tools exist to measure detector robustness as part of academic research.
They are intended for academic purposes only, and any subsequent release will be
for academic purposes only.

## Contents

- `tools/` — 156 tool implementations plus `base_tool.py` (the interface)
- `MANIFEST.csv` — one row per tool: source URL, verbatim evidence
  quote

Every tool in this suite carries a source URL and a supporting quote.


## Idea versus implementation

Sources supply the *concept*; the implementation is elaborated from it. A source may
say only that AI images look "too polished, not like a phone snapshot", while the
corresponding tool implements highlight clipping, shadow-weighted noise, a
white-balance shift and 4:2:0 chroma subsampling. The evidence quote in the manifest
is therefore evidence for the idea, not for the specific operations or parameter
ranges.

## Interface

```python
class SomeTool(BaseTool):
    def apply(self, image: PIL.Image, params: dict) -> PIL.Image: ...
    def get_param_schema(self) -> dict: ...
```

Tools are loaded by name from the `tools` package. Run the snippets below from
inside this directory. Every parameter has a default, so `apply(image, {})` runs
without supplying any:

```python
from PIL import Image
from tools import get_tool_by_name, available_tools

available_tools()                     # -> the 156 tool names
tool = get_tool_by_name("print_rescan")
image = Image.open("../example.jpg").convert("RGB")
out = tool.apply(image, {})           # defaults for every parameter
```

### Example: applying one tool at one strength

The snippet below is **an illustrative example** — one tool at one arbitrarily
chosen strength, to show the calling convention. In the evaluation protocol,
five diverging parameter settings per image are proposed by a vision LLM and the
strongest quality-passing one is kept.

```python
from PIL import Image
from tools import get_tool_by_name

tool = get_tool_by_name("print_rescan")
image = Image.open("../example.jpg").convert("RGB")

params = {                      # one arbitrary setting, not a tuned one
    "paper_intensity": 0.112,
    "moire_strength": 0.0,
    "rotation_deg": 0.65,
    "gamma_shift": 1.078,
    "color_cast": "warm",
}
out = tool.apply(image, params)
```

On a benchmark image this setting stays comfortably above the 0.75 quality floor.
SSIM is measured after compensating for the small rotation the tool applies, so
that a geometric shift is not scored as a loss of image quality. Call
`tool.get_param_schema()` for each parameter's range and default, and
`tool.get_guidance()` for the ranges the tool itself recommends.
