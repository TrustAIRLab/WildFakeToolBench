"""
Lens Distortion Tool — barrel/pincushion geometric distortion.

Source: Reddit 18ueofn — "introducing camera lens distortion" as a evasion method.

Real camera lenses exhibit radial distortion: barrel (bulging outward) in wide-angle
lenses, pincushion (pinching inward) in telephoto. AI images have pixel-perfect
rectilinear geometry. A subtle barrel/pincushion warp breaks that perfection without
being perceptible as a manipulation.
"""
from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import map_coordinates
from .base_tool import BaseTool


class LensDistortionTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="lens_distortion",
            description=(
                "Applies radial lens distortion (barrel or pincushion) by remapping pixel "
                "coordinates through a polynomial radial transform. Positive strength = barrel "
                "(wide-angle bulge), negative = pincushion (telephoto pinch). Breaks the "
                "pixel-perfect rectilinear geometry of AI-generated images, mimicking real "
                "optical lens characteristics."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.15)
        center_x = params.get("center_x", 0.5)
        center_y = params.get("center_y", 0.5)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32)
        h, w = arr.shape[:2]

        cx, cy = center_x * w, center_y * h

        # Normalized coordinate grid
        ys = np.arange(h, dtype=np.float32)
        xs = np.arange(w, dtype=np.float32)
        xx, yy = np.meshgrid(xs, ys)

        dx = (xx - cx) / w
        dy = (yy - cy) / h
        r2 = dx ** 2 + dy ** 2

        # Barrel/pincushion: r' = r * (1 + k * r^2)
        factor = 1.0 + strength * r2
        src_x = cx + dx * factor * w
        src_y = cy + dy * factor * h

        # Clip to valid range
        src_x = np.clip(src_x, 0, w - 1)
        src_y = np.clip(src_y, 0, h - 1)

        coords = [src_y, src_x]
        out = np.stack([
            map_coordinates(arr[..., c], coords, order=1, mode='nearest')
            for c in range(3)
        ], axis=-1)

        return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "strength 0.05-0.15 subtle barrel (wide-angle look); 0.15-0.3 visible. "
            "Negative strength (-0.05 to -0.2) pincushion (telephoto look). "
            "center_x/y 0.5/0.5 = image center is standard. "
            "Most useful when image has unnaturally straight lines or perfect geometry."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float", "min": -0.40, "max": 0.40,
                "description": "Distortion strength: positive=barrel (bulge), negative=pincushion (pinch)",
                "default": 0.12, "required": False,
            },
            "center_x": {
                "type": "float", "min": 0.3, "max": 0.7,
                "description": "Distortion center X (0.5 = image center)",
                "default": 0.5, "required": False,
            },
            "center_y": {
                "type": "float", "min": 0.3, "max": 0.7,
                "description": "Distortion center Y (0.5 = image center)",
                "default": 0.5, "required": False,
            },
        }
