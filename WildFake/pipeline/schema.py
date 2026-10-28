"""
WildTool-Dataset schema.

Every entry is a WildToolEntry: one tool, fully traced from source evidence
through LLM extraction → implementation.
"""
from __future__ import annotations

from typing import Optional
from pydantic import BaseModel, Field


class SourceRecord(BaseModel):
    platform: str                       # reddit | web | forum
    url: str
    document_title: str = ""
    discourse_cluster: str = ""         # realism_framing | detection_evasion | capture_pipeline | ...
    evidence_quote: str = ""            # verbatim excerpt supporting this method
    collected_at: str = ""              # ISO-8601 UTC


class DiscoveryRecord(BaseModel):
    idea: str = ""
    mechanism: str = ""                 # why it fools a detector at the signal level
    method_type: str = ""               # numeric | generative | hybrid
    sub_category: str = ""             # post_processing | capture_pipeline | compression | frequency | ...
    feasibility_score: float = 0.0
    efficacy_score: float = 0.0
    novelty_score: float = 0.0
    priority: float = 0.0
    implementation_hint: str = ""
    params_sketch: dict = Field(default_factory=dict)
    sources: list[SourceRecord] = Field(default_factory=list)


class ImplementationRecord(BaseModel):
    status: str = "sketch"             # sketch | staged | implemented
    tool_name: str = ""
    tool_file: str = ""
    class_name: str = ""
    synthesis_method: str = ""         # llm_auto | llm_assisted
    code_hash: str = ""
    smoke_test_passed: bool = False
    ssim_floor: float = 0.0            # SSIM(original, tool(original)) on 512×512 random image
    validated_at: str = ""


class WildToolEntry(BaseModel):
    id: str = ""                        # wildtool-NNNN
    schema_version: str = "3.0"
    name: str
    description: str = ""
    discovery: DiscoveryRecord = Field(default_factory=DiscoveryRecord)
    implementation: ImplementationRecord = Field(default_factory=ImplementationRecord)


class WildToolDataset(BaseModel):
    schema_version: str = "3.0"
    generated_at: str = ""
    counts: dict = Field(default_factory=dict)
    entries: list[WildToolEntry] = Field(default_factory=list)
