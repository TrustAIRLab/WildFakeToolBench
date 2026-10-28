"""
Photo Texture Blend Tool — overlays procedural real-world material textures.

Source: Blog post "How to Make AI-Generated Images Look Realistic":
"This technique involves overlaying or merging AI-generated images with real
photographs... Applying various filters to enhance texture, color grading, and
overall appearance."

Instead of requiring real photograph assets, this tool procedurally synthesizes
realistic material textures (linen fabric, matte paper, canvas weave, asphalt)
using spatially correlated noise with appropriate frequency content for each
material. These textures break the unnaturally clean surface of AI images.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


def _make_linen(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Horizontal + vertical weave at ~4-8px periodicity."""
    xs = np.linspace(0, w / 5, w)
    ys = np.linspace(0, h / 5, h)
    horiz = (np.sin(np.pi * ys * 2.0) * 0.5 + 0.5)[..., None]
    vert = (np.sin(np.pi * xs * 2.0) * 0.5 + 0.5)[None, ...]
    weave = horiz * 0.5 + vert * 0.5
    noise = rng.standard_normal((h, w)).astype(np.float32)
    noise = gaussian_filter(noise, sigma=0.5)
    return np.clip(weave + noise * 0.15, 0.0, 1.0)


def _make_paper(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Matte paper: low-frequency brightness variation + fine fiber noise."""
    base = rng.standard_normal((h, w)).astype(np.float32)
    fiber = gaussian_filter(base, sigma=0.4)
    fiber = fiber / (np.abs(fiber).max() + 1e-8)
    low_freq = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=12)
    low_freq = low_freq / (np.abs(low_freq).max() + 1e-8)
    return np.clip(0.5 + low_freq * 0.15 + fiber * 0.1, 0.0, 1.0)


def _make_canvas(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Canvas: coarser weave with diagonal grain."""
    xs = np.linspace(0, w / 8, w)
    ys = np.linspace(0, h / 8, h)
    diag = np.outer(np.sin(np.pi * ys), np.sin(np.pi * xs))
    diag = (diag * 0.5 + 0.5)
    noise = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=1.5)
    noise = noise / (np.abs(noise).max() + 1e-8)
    return np.clip(diag * 0.7 + noise * 0.3, 0.0, 1.0)


def _make_asphalt(h: int, w: int, rng: np.random.Generator) -> np.ndarray:
    """Asphalt / rough ground: multi-scale stone aggregate noise."""
    scales = [0.5, 1.2, 3.0, 8.0]
    weights = [0.5, 0.3, 0.15, 0.05]
    out = np.zeros((h, w), dtype=np.float32)
    for sigma, weight in zip(scales, weights):
        layer = gaussian_filter(rng.standard_normal((h, w)).astype(np.float32), sigma=sigma)
        layer = layer / (np.abs(layer).max() + 1e-8)
        out += layer * weight
    return np.clip(out * 0.5 + 0.5, 0.0, 1.0)


_TEXTURE_MAKERS = {
    "linen": _make_linen,
    "paper": _make_paper,
    "canvas": _make_canvas,
    "asphalt": _make_asphalt,
}


class PhotoTextureBlendTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="photo_texture_blend",
            description=(
                "Overlays a procedurally synthesized real-world material texture onto the "
                "image at low opacity. Textures available: 'linen' (horizontal/vertical weave), "
                "'paper' (matte fiber with low-frequency brightness variation), 'canvas' "
                "(coarse diagonal weave), 'asphalt' (multi-scale aggregate roughness). "
                "Breaks AI images' unnaturally clean surfaces by mimicking the subtle texture "
                "present in real photographs of physical environments."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        texture_type = params.get("texture_type", "paper")
        opacity = params.get("opacity", 0.04)
        blend_mode = params.get("blend_mode", "overlay")

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        rng = np.random.default_rng()
        maker = _TEXTURE_MAKERS.get(texture_type, _make_paper)
        texture = maker(h, w, rng)[..., None]  # (H, W, 1) broadcast over RGB

        if blend_mode == "overlay":
            # Photoshop-style overlay: darkens shadows, lightens highlights
            low = 2.0 * arr * texture
            high = 1.0 - 2.0 * (1.0 - arr) * (1.0 - texture)
            blended = np.where(arr < 0.5, low, high)
        elif blend_mode == "multiply":
            blended = arr * texture
        elif blend_mode == "screen":
            blended = 1.0 - (1.0 - arr) * (1.0 - texture)
        else:
            blended = arr + texture - 0.5  # soft light

        out = arr * (1.0 - opacity) + blended * opacity
        out = np.clip(out, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "opacity 0.02-0.05 imperceptible to human eye but changes frequency content; "
            "0.05-0.10 subtle visible grain; >0.12 obvious texture. "
            "texture_type 'paper' most neutral; 'linen'/'canvas' adds directional pattern; "
            "'asphalt' strongest for outdoor scenes. "
            "blend_mode 'overlay' preserves overall brightness best. "
            "Most useful when image surfaces look unnaturally smooth or sterile."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "texture_type": {
                "type": "str",
                "description": "Material texture to overlay: 'linen', 'paper', 'canvas', 'asphalt'",
                "default": "paper",
                "required": False,
            },
            "opacity": {
                "type": "float",
                "min": 0.01,
                "max": 0.20,
                "description": "Blend opacity (0.02-0.05 subtle, 0.05-0.10 visible, >0.12 strong)",
                "default": 0.04,
                "required": False,
            },
            "blend_mode": {
                "type": "str",
                "description": "Blend mode: 'overlay' (contrast-preserving), 'multiply' (darkening), 'screen' (lightening)",
                "default": "overlay",
                "required": False,
            },
        }
