"""
Shadow/Highlight Recovery Tool — independent tone zone adjustment.

Source: blog_4e53b3f5, blog_16f53eaa — color correction targeting specific
tone zones. Reddit 18ueofn — "camera RAW methods."

Real RAW processors (Lightroom, Capture One) independently control shadow lift
and highlight recovery. AI images have globally linear tone responses without
per-zone compression. Lifting shadows and pulling down blown highlights gives
the characteristic look of professionally processed photography.
"""
from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class ShadowHighlightTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="shadow_highlight",
            description=(
                "Independently adjusts shadow and highlight tone zones: shadow_lift raises "
                "the black point and recovers crushed shadow detail (flatter, film-like); "
                "highlight_compress pulls down blown highlights and restores texture in bright "
                "areas. midpoint sets the luminance pivot between zones. Mimics Lightroom/RAW "
                "shadow+highlight sliders. Distinct from color_adjust which shifts all tones globally."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        shadow_lift = params.get("shadow_lift", 0.08)
        highlight_compress = params.get("highlight_compress", 0.10)
        midpoint = params.get("midpoint", 0.45)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0

        # Build a per-pixel tone curve
        # Shadow mask: smooth transition from 1 at black to 0 at midpoint
        shadow_mask = np.clip(1.0 - arr / (midpoint + 1e-8), 0.0, 1.0) ** 1.5
        # Highlight mask: smooth from 0 at midpoint to 1 at white
        highlight_mask = np.clip((arr - midpoint) / (1.0 - midpoint + 1e-8), 0.0, 1.0) ** 1.5

        out = arr + shadow_lift * shadow_mask - highlight_compress * highlight_mask
        out = np.clip(out, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "shadow_lift 0.05-0.12: subtle shadow recovery (film look); 0.12-0.25 strong. "
            "highlight_compress 0.05-0.15: pulls blown highlights; 0.15-0.30 strong. "
            "midpoint 0.35-0.55 controls tone zone boundary. "
            "Most useful when image has pure-black shadows or pure-white highlights (AI characteristic)."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "shadow_lift": {
                "type": "float", "min": 0.0, "max": 0.30,
                "description": "Shadow lift / black point raise (0.05-0.12 subtle, >0.15 strong)",
                "default": 0.08, "required": False,
            },
            "highlight_compress": {
                "type": "float", "min": 0.0, "max": 0.30,
                "description": "Highlight recovery / white point pull-down (0.05-0.15 subtle)",
                "default": 0.10, "required": False,
            },
            "midpoint": {
                "type": "float", "min": 0.30, "max": 0.65,
                "description": "Luminance pivot between shadow and highlight zones (0.4-0.5 typical)",
                "default": 0.45, "required": False,
            },
        }
