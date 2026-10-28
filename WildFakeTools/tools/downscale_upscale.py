"""
Downscale-Upscale Tool — destroys AI-perfect pixel patterns.

AI-generated images have pixel-perfect patterns that real photos don't.
Downscaling then upscaling introduces interpolation artifacts similar
to social media re-compression or camera processing pipelines.
"""

from typing import Dict, Any
from PIL import Image

from .base_tool import BaseTool


class DownscaleUpscaleTool(BaseTool):
    """Downscales then upscales image to break AI-perfect pixel patterns."""

    def __init__(self):
        super().__init__(
            name="downscale_upscale",
            description=(
                "Downscales image then upscales back to original size. "
                "Destroys AI-perfect pixel patterns through interpolation, "
                "similar to social media re-compression or camera processing. "
                "Subtle but effective at breaking pixel-level AI signatures."
            )
        )

    def get_guidance(self) -> str:
        return "Breaks pixel-perfect AI patterns by downscaling then upscaling. Scale 0.7-0.8 for moderate, 0.5-0.7 for aggressive. Mimics social media recompression."

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "scale_factor": {
                "type": "float",
                "description": (
                    "How much to downscale (0.5 = half size, 0.8 = slight reduction). "
                    "0.3-0.5: aggressive (visible quality loss), "
                    "0.5-0.7: moderate (noticeable softening), "
                    "0.7-0.9: subtle (slight pattern disruption). "
                    "Start with 0.75."
                ),
                "default": 0.85,
                "min": 0.3,
                "max": 0.95,
            }
        }

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        scale = params.get("scale_factor", 0.75)

        original_size = image.size  # (width, height)
        small_size = (int(original_size[0] * scale), int(original_size[1] * scale))

        # Downscale with LANCZOS (high quality), upscale with BILINEAR (introduces softness)
        small = image.resize(small_size, Image.Resampling.LANCZOS)
        result = small.resize(original_size, Image.Resampling.BILINEAR)

        return result
