# WildFake — In-the-Wild Knowledge Collection and Tool Synthesis Framework

Anonymised artifact. No absolute paths, user names, institution names or
credentials are included; see `.env.example` for the variables that must be set.

## What this is

An end-to-end pipeline that retrieves public web discourse about making
AI-generated images look photographic, mines it for atomic manipulation methods
with verbatim provenance, ranks and de-duplicates those methods, and synthesises
each surviving method into a runnable image-processing tool behind a three-gate
validation loop.

## Pipeline stages

| Stage | Entry point | Model | Output |
|---|---|---|---|
| 1. Retrieval | `pipeline/sources/web_search.py` | — (Tavily API) | source corpus |
| 2. Normalization | `pipeline/nodes/web_collector.py` | — (deterministic) | `Post` records |
| 3. Method mining | `pipeline/nodes/feasibility_analyzer.py` | gpt-4o-mini | per-post verdict + candidate methods with `quoted_evidence` |
| 4. Ranking | `pipeline/nodes/method_ranker.py` | gpt-4o-mini ×3 | de-duplicated recipes |
| 5. Synthesis | `pipeline/distillation/tool_synthesizer.py` | gpt-4o-mini | validated `BaseTool` subclasses |
| 6. Dataset | `pipeline/dataset/builder.py` | — | versioned tool dataset |

We note that this artifact ships the pipeline **code** and its **final output** — the tool suite in `../WildFakeTools/tools/` and the per-tool provenance in `../WildFakeTools/MANIFEST.csv`. Thereby, the users need to rerun the whole pipeline to regenerate them step by step and generate intermediate results under `outputs/` to build new tool sets:

| Stage | Written to |
|---|---|
| 1. Retrieval | `outputs/web_harvest/sources.json` |
| 2. Normalization | `outputs/posts/web_posts.jsonl` |
| 3. Method mining | `outputs/analyses.jsonl` |
| 4. Ranking | `outputs/ranked_methods.json` |
| 5. Synthesis | new tool files in `../WildFakeTools/tools/` |
| 6. Dataset | `WildTool-Dataset/v<version>/wildtool_dataset.{json,md}` |

Each stage can be skipped once its output exists (`--skip-collect`,
`--skip-analyze`, `--skip-rank`, `--skip-synth`), so a run can be resumed from
any completed step. `--only-dataset` runs stage 6 alone; it always works from
the released tool files, and additionally fills the `discovery` fields (idea,
mechanism, scores, priority, source posts) when `outputs/ranked_methods.json`
from a prior stage-4 run is present. Because web search results and LLM
outputs change over time, a fresh run yields a similar but not byte-identical
method pool; the released tools in ``WildFakeTools/'' are the fixed reference set used in the paper.

Orchestrated by `pipeline/pipeline.py`.

### Retrieval configuration
49 general queries.
Filters: domain blocklist, minimum body length of 300 characters, exact-URL de-duplication.

### Method mining
One LLM call per document. Scores `novelty`, `feasibility`, `relevance` on 1–10,
assigns a verdict in {valuable, borderline, discard}, and extracts atomic methods
each carrying `quoted_evidence`, `idea`, `mechanism` and `method_type`.

### Ranking
Batched extraction (batch size 15), one global cross-batch de-duplication call
resolved with union-find, then archetype clustering. Priority weighting is fixed
in `prompts/ranker_prompt.txt`.

### Synthesis gates
Each candidate implementation must pass, in order:
1. `ast.parse` — syntax
2. subprocess smoke test on a 512×512 synthetic photo-like image, 90 s timeout
3. global luminance SSIM ≥ 0.25 against the input

On failure the error text is fed back into the same LLM conversation for
self-correction, up to 3 rounds. Passing tools are promoted automatically.

## Evaluating against a detector

The detector-evaluation harness is not part of this artifact. It is not needed
to reproduce the protocol, which is deliberately simple:

- Each detector is run **from its own public release, with its default
  inference procedure** — default weights, default preprocessing and the
  official inference script or prompt. We do not fine-tune, re-threshold or
  otherwise adapt any detector.
- A processed image is **fed directly to the detector**, which returns REAL or
  FAKE. No detector output, gradient or score is used when producing images.
- Images are processed by the tools first (five parameter settings per tool,
  proposed by GPT-4o-mini) and filtered on the SSIM ≥ 0.75 quality floor; only
  surviving images are submitted for classification.
- An attack on an image counts as successful when at least one quality-passing
  processed version is classified REAL. ASR is computed only over images that
  the detector classifies as FAKE before processing.

No detector code or weights are included here.

## Reproducing

```bash
pip install -r requirements.txt
cp .env.example .env      # fill in the API keys
python pipeline/pipeline.py
```
