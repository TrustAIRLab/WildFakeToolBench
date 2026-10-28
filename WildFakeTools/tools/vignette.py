"""
Vignetting Tool — simulates real lens light falloff at corners.

Real camera lenses let less light through at the edges, causing
natural darkening toward corners. AI images have uniform brightness.
"""

import numpy as np
from typing import Dict, Any
from PIL import Image

from .base_tool import BaseTool


class VignetteTool(BaseTool):
    """Adds lens vignetting (corner darkening) to simulate real camera optics."""

    def __init__(self):
        super().__init__(
            name="vignette",
            description=(
                "Adds natural lens vignetting (darkened corners) to simulate "
                "real camera optics. AI images have unnaturally uniform brightness "
                "to the edges — this adds realistic light falloff."
            )
        )

    def get_guidance(self) -> str:
        return "Darkens image corners like a real camera lens. Strength 0.2-0.4 for subtle, 0.4-0.6 for visible. Targets 'uniform brightness' or 'too perfect lighting' feedback."

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "description": (
                    "Vignette darkness strength. "
                    "0.1-0.3: subtle (barely noticeable), "
                    "0.3-0.5: moderate (natural looking), "
                    "0.5-0.8: strong (dramatic). "
                    "Start with 0.3."
                ),
                "default": 0.15,
                "min": 0.05,
                "max": 0.8,
            }
        }

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.3)

        img = np.array(image.convert("RGB")).astype(np.float32)
        h, w = img.shape[:2]

        # Create radial gradient mask: 1.0 at center, 0.0 at corners
        y, x = np.mgrid[0:h, 0:w].astype(np.float32)
        cx, cy = w / 2, h / 2
        # Normalize to [-1, 1] range
        x = (x - cx) / cx
        y = (y - cy) / cy
        # Distance from center (elliptical to handle non-square images)
        dist = np.sqrt(x**2 + y**2)
        # Vignette mask: 1.0 at center, falls off toward edges
        mask = 1.0 - strength * (dist / dist.max()) ** 2
        mask = np.clip(mask, 0, 1)

        # Apply mask to all channels
        result = img * mask[:, :, np.newaxis]
        result = np.clip(result, 0, 255).astype(np.uint8)
        return Image.fromarray(result)
