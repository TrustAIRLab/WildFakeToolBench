"""
Film Grain Tool — simulates analog film stock grain (Portra 400 / Tri-X).

Derived from r/StableDiffusion community guidance that "AI images lack authentic
film grain" — grain is spatially correlated (not pure Gaussian), has film-specific
color response, and appears stronger in shadows than highlights.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class FilmGrainTool(BaseTool):
    """Apply analog-style film grain with luminance-dependent intensity."""

    def __init__(self):
        super().__init__(
            name="film_grain",
            description=(
                "Overlays spatially-correlated analog film grain with luminance-weighted "
                "distribution (stronger in shadows, weaker in highlights) and stock-specific "
                "color tint. Portra400 preset: warm color grain; Tri-X: monochrome grain; "
                "Cinestill: strong warm color grain. Adds organic texture variance to images."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        preset = params.get("preset", "portra400")
        intensity = params.get("intensity", 0.06)
        grain_size = params.get("grain_size", 1.0)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        rng = np.random.default_rng()

        if preset == "tri_x":
            noise_mono = rng.standard_normal((h, w)).astype(np.float32)
            if grain_size > 1.0:
                noise_mono = gaussian_filter(noise_mono, sigma=grain_size * 0.6)
                noise_mono /= (noise_mono.std() + 1e-8)
            noise = np.stack([noise_mono] * 3, axis=-1)
            color_tint = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            noise = rng.standard_normal((h, w, 3)).astype(np.float32)
            if grain_size > 1.0:
                for c in range(3):
                    noise[..., c] = gaussian_filter(noise[..., c], sigma=grain_size * 0.6)
                    noise[..., c] /= (noise[..., c].std() + 1e-8)
            if preset == "portra400":
                color_tint = np.array([1.0, 0.85, 0.7], dtype=np.float32)
            elif preset == "cinestill":
                color_tint = np.array([1.1, 0.9, 0.8], dtype=np.float32)
            else:
                color_tint = np.array([1.0, 1.0, 1.0], dtype=np.float32)

        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        shadow_weight = (1.0 - luminance) * 0.7 + 0.3
        shadow_weight = shadow_weight[..., None]

        grain = noise * color_tint[None, None, :] * intensity * shadow_weight
        out = np.clip(arr + grain, 0.0, 1.0)

        return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "Parameter ranges: intensity 0.04-0.08 subtle, 0.08-0.14 visible. "
            "grain_size ~1.0 matches fine 35mm film, ~2.0 matches coarse high-ISO. "
            "Most useful when the image reads as too digitally clean / lacking organic noise."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "preset": {
                "type": "str",
                "description": "Film stock preset: 'portra400' (warm color), 'tri_x' (mono), 'cinestill' (strong warm)",
                "default": "portra400",
                "required": False,
            },
            "intensity": {
                "type": "float",
                "min": 0.02,
                "max": 0.15,
                "description": "Grain amplitude (0.04-0.08 subtle, 0.08-0.14 noticeable)",
                "default": 0.06,
                "required": False,
            },
            "grain_size": {
                "type": "float",
                "min": 0.5,
                "max": 2.5,
                "description": "Grain cluster size (1.0 = fine grain, 2.0 = coarse 400-speed look)",
                "default": 1.0,
                "required": False,
            },
        }
