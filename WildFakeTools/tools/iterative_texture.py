"""
Iterative Texture Tool — multi-cycle interpolation for painterly texture.

Source: r/StableDiffusion comment:
"I use iterative upscale in comfyui with siax model and get texture like it
was painted with pencil"

Each cycle performs a small downscale followed by upscale with a different
interpolation kernel. Cycles accumulate interpolation artifacts that mimic
the slightly non-uniform texture of traditional media (pencil/watercolor).
This is distinct from a single downscale-upscale pass: the rotation of kernels
across cycles creates anisotropic micro-texture rather than uniform blurring.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool

_KERNELS = [Image.Resampling.BILINEAR, Image.Resampling.BICUBIC, Image.Resampling.LANCZOS]


class IterativeTextureTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="iterative_texture",
            description=(
                "Runs multiple cycles of small downscale-then-upscale with rotating "
                "interpolation kernels (BILINEAR → BICUBIC → LANCZOS). Each cycle "
                "accumulates anisotropic interpolation artifacts that produce a "
                "painterly/pencil-like micro-texture, mimicking the iterative upscale "
                "pipeline used in ComfyUI with SIAX model."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        iterations = params.get("iterations", 3)
        cycle_scale = params.get("cycle_scale", 0.88)
        blend_strength = params.get("blend_strength", 0.75)

        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size  # (W, H)
        original_arr = np.asarray(image, dtype=np.float32) / 255.0

        current = image
        for i in range(iterations):
            kernel_down = _KERNELS[i % len(_KERNELS)]
            kernel_up = _KERNELS[(i + 1) % len(_KERNELS)]
            small_size = (
                max(16, int(original_size[0] * cycle_scale)),
                max(16, int(original_size[1] * cycle_scale)),
            )
            small = current.resize(small_size, kernel_down)
            current = small.resize(original_size, kernel_up)

        result_arr = np.asarray(current, dtype=np.float32) / 255.0

        # Blend with original so blend_strength controls how much texture is added
        out = original_arr * (1.0 - blend_strength) + result_arr * blend_strength
        out = np.clip(out, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "iterations 2-3: subtle texture; 4-5: visible painterly look. "
            "cycle_scale 0.85-0.92: slight; 0.65-0.85: more aggressive per cycle. "
            "blend_strength 0.5-0.7 moderate blend; >0.8 strong texture effect. "
            "Most useful when the image looks too digitally smooth or computer-generated."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "iterations": {
                "type": "int",
                "min": 1,
                "max": 6,
                "description": "Number of downscale-upscale cycles (2-3 subtle, 4-5 visible)",
                "default": 3,
                "required": False,
            },
            "cycle_scale": {
                "type": "float",
                "min": 0.60,
                "max": 0.95,
                "description": "Scale factor per cycle (0.85-0.92 subtle, 0.65-0.85 aggressive)",
                "default": 0.88,
                "required": False,
            },
            "blend_strength": {
                "type": "float",
                "min": 0.2,
                "max": 1.0,
                "description": "How much to blend textured result vs original (0.5-0.7 moderate)",
                "default": 0.75,
                "required": False,
            },
        }
