# Fake-to-Real Tools in the Wild

Tool Bench (WildFake/) for our WildFakeToolBench and the code for our tool bench collection in WildFake/. It contains the pipeline that mines
public web discourse into executable image-manipulation tools, the resulting
tool suite, and an example set of attack outputs.

No absolute paths, user names, institution names or credentials are included.

## Contents

| Directory | What it holds |
|---|---|
| `WildFakeTools/` | The 156 released tools of our WildFakeToolBench, with `MANIFEST.csv` giving each tool's source URL, verbatim evidence quote |
| `WildFake/` | The code pipeline for our WildFakeToolBench collection: retrieval, method mining, ranking and de-duplication, and tool synthesis |
| `Example_ProcessedImages/` | An example set of processed benchmark images |

## Intended use

This work is academic research: the tools are built and released to measure the robustness of AI-generated-image detectors, so that weaknesses can be identified and addressed. The tool suite is intended for academic purposes only, and any
subsequent release will be for academic purposes only.

## Scope

This artifact covers knowledge collection and tool synthesis, and releases the
pipeline code together with its final output: the tools and the public sources
they were mined from (`WildFakeTools/MANIFEST.csv`). Intermediate pipeline
results (retrieved pages, mined and ranked methods) are not released; running
the pipeline regenerates them step by step, as described in
`WildFake/README.md`. Detectors
are run from their own public releases; no detector code or weights are included
here. Benchmark images are AI-generated and are not redistributed beyond the
example directory.

## Quick start

```bash
cd WildFake
pip install -r requirements.txt
cp .env.example .env        # fill in your own API keys
python pipeline/pipeline.py
```

To use the released tools directly, without re-running the pipeline:

Run from this directory (the artifact root):

```python
import sys; sys.path.insert(0, "WildFakeTools")

from PIL import Image
from tools import get_tool_by_name

tool = get_tool_by_name("print_rescan")
image = Image.open("example.jpg").convert("RGB")

params = {                      # one arbitrary setting, not a tuned one
    "paper_intensity": 0.112,
    "moire_strength": 0.0,
    "rotation_deg": 0.65,
    "gamma_shift": 1.078,
    "color_cast": "warm",
}
out = tool.apply(image, params)
```
