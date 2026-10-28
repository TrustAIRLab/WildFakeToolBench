"""
Chromatic Aberration Tool — simulates real lens color fringing.

Real camera lenses refract R/G/B wavelengths slightly differently,
causing color shifts at image edges. AI images lack this artifact.
"""

import numpy as np
from typing import Dict, Any
from PIL import Image

from .base_tool import BaseTool


class ChromaticAberrationTool(BaseTool):
    """Adds chromatic aberration (color fringing) to simulate real lens behavior."""

    def __init__(self):
        super().__init__(
            name="chromatic_aberration",
            description=(
                "Adds subtle chromatic aberration (color fringing at edges) "
                "to simulate real camera lens imperfections. AI images have "
                "perfectly aligned color channels — this breaks that perfection."
            )
        )

    def get_guidance(self) -> str:
        return "Simulates real lens color fringing at image edges. Shift 0.5-1.5 for subtle, 1.5-2.5 for visible. Targets 'too perfect edges' or 'unnaturally sharp' feedback."

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "shift": {
                "type": "float",
                "description": (
                    "Pixel shift amount for R/B channels. "
                    "0.5-1.0: subtle (barely visible), "
                    "1.0-2.0: moderate (noticeable at edges), "
                    "2.0-3.0: strong (obvious fringing). "
                    "Start with 1.0."
                ),
                "default": 0.5,
                "min": 0.3,
                "max": 3.0,
            }
        }

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        shift = params.get("shift", 1.0)

        img = np.array(image.convert("RGB")).astype(np.float32)
        h, w = img.shape[:2]

        # Shift red channel outward, blue channel inward (relative to center)
        # Using simple affine shift — red shifts +shift px, blue shifts -shift px
        from scipy.ndimage import shift as ndshift

        r, g, b = img[:, :, 0], img[:, :, 1], img[:, :, 2]
        r_shifted = ndshift(r, [0, shift], mode='reflect')
        b_shifted = ndshift(b, [0, -shift], mode='reflect')

        result = np.stack([r_shifted, g, b_shifted], axis=2)
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
