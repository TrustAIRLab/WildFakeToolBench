"""SourceDocument — unified document model for all collectors."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class SourceDocument:
    url: str
    title: str
    content: str                        # main text extracted from source
    platform: str                       # reddit | web | forum
    discourse_cluster: str = ""
    collected_at: str = field(
        default_factory=lambda: datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    )
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "url": self.url,
            "title": self.title,
            "content": self.content,
            "platform": self.platform,
            "cluster": self.discourse_cluster,
            "collected_at": self.collected_at,
            **self.metadata,
        }
