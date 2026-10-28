"""
Upscale-Grain-Denoise Tool — multi-step pipeline derived from Reddit community.

Source: r/StableDiffusion comment:
"Upscaling, film grain, second pass with low denoise on ZimageTurbo"

The pipeline: (1) upscale to break pixel-perfect AI patterns, (2) overlay analog
film grain for organic texture, (3) apply a gentle bilateral-style denoise pass
that smooths micro-artifacts while preserving structural edges.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class UpscaleGrainDenoiseTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="upscale_grain_denoise",
            description=(
                "Three-stage pipeline: upscales image to break pixel-perfect AI patterns, "
                "overlays spatially-correlated analog film grain for organic texture, then "
                "applies a low-strength denoise pass (edge-preserving Gaussian) that smooths "
                "micro-artifacts while preserving structural detail. Mimics the ZimageTurbo "
                "iterative upscale workflow from r/StableDiffusion."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        scale_factor = params.get("scale_factor", 1.5)
        grain_amount = params.get("grain_amount", 0.05)
        denoise_strength = params.get("denoise_strength", 0.6)

        if image.mode != "RGB":
            image = image.convert("RGB")

        original_size = image.size  # (W, H)

        # Stage 1: upscale with LANCZOS, downscale back with BILINEAR
        up_size = (int(original_size[0] * scale_factor), int(original_size[1] * scale_factor))
        upscaled = image.resize(up_size, Image.Resampling.LANCZOS)
        arr = np.asarray(upscaled.resize(original_size, Image.Resampling.BILINEAR),
                         dtype=np.float32) / 255.0

        # Stage 2: analog film grain (luminance-weighted)
        h, w = arr.shape[:2]
        rng = np.random.default_rng()
        noise = rng.standard_normal((h, w, 3)).astype(np.float32)
        for c in range(3):
            noise[..., c] = gaussian_filter(noise[..., c], sigma=0.8)
            noise[..., c] /= (noise[..., c].std() + 1e-8)

        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        shadow_weight = ((1.0 - luminance) * 0.7 + 0.3)[..., None]
        color_tint = np.array([1.0, 0.88, 0.75], dtype=np.float32)
        grain = noise * color_tint[None, None, :] * grain_amount * shadow_weight
        arr = np.clip(arr + grain, 0.0, 1.0)

        # Stage 3: low-strength denoise via channel-wise Gaussian smoothing
        # sigma is scaled so that denoise_strength=0 → no change, 1.0 → strong smoothing
        sigma = denoise_strength * 0.8  # max sigma ~0.8 px is very gentle
        if sigma > 0.05:
            smoothed = np.stack([
                gaussian_filter(arr[..., c], sigma=sigma) for c in range(3)
            ], axis=-1)
            # Blend: preserve edges by mixing based on local gradient magnitude
            grad = np.abs(np.gradient(luminance))
            edge_weight = np.clip(grad[0] + grad[1], 0.0, 1.0)[..., None]
            arr = smoothed * (1.0 - edge_weight) + arr * edge_weight

        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "scale_factor 1.3-1.6: subtle upscale artifacts; 1.6-2.0: more aggressive. "
            "grain_amount 0.03-0.06 subtle, 0.07-0.10 visible grain. "
            "denoise_strength 0.3-0.6 gentle smoothing; >0.7 noticeable softening. "
            "Most effective on images that read as pixel-perfect or overly crisp."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "scale_factor": {
                "type": "float",
                "min": 1.2,
                "max": 2.0,
                "description": "Upscale ratio before downsampling back (1.3-1.6 subtle, 1.6-2.0 aggressive)",
                "default": 1.5,
                "required": False,
            },
            "grain_amount": {
                "type": "float",
                "min": 0.02,
                "max": 0.12,
                "description": "Film grain intensity after upscale (0.03-0.06 subtle, 0.07-0.10 visible)",
                "default": 0.05,
                "required": False,
            },
            "denoise_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Edge-preserving denoise pass strength (0.3-0.6 gentle, >0.7 soft)",
                "default": 0.5,
                "required": False,
            },
        }
