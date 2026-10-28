"""
Print-Rescan Tool — simulates print-then-photograph analog laundering.

Derived from r/StableDiffusion guidance: "print your AI image then take a photo
of it". Introduces paper fiber texture, moire, slight perspective, gamma shift,
and color cast that are absent in any digital pipeline.
"""

from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class PrintRescanTool(BaseTool):
    """Simulate print-and-rescan: paper fiber + optional moire + perspective + gamma."""

    def __init__(self):
        super().__init__(
            name="print_rescan",
            description=(
                "Applies a print-then-photograph laundering transformation: overlays "
                "paper-fiber texture (DoG-filtered noise + low-frequency brightness "
                "variation), optionally adds moire interference, applies small rotation "
                "(handheld re-photograph misalignment), gamma shift, paper-color cast, "
                "and a mild recapture blur. Drastically perturbs digital-pipeline signatures."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        paper_intensity = params.get("paper_intensity", 0.12)
        moire_strength = params.get("moire_strength", 0.0)
        rotation_deg = params.get("rotation_deg", 0.8)
        gamma_shift = params.get("gamma_shift", 1.05)
        color_cast = params.get("color_cast", "warm")

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        rng = np.random.default_rng()
        fiber = rng.standard_normal((h, w)).astype(np.float32)
        fiber = gaussian_filter(fiber, sigma=0.8) - gaussian_filter(fiber, sigma=3.0)
        fiber /= (np.abs(fiber).max() + 1e-8)
        paper = 1.0 + fiber[..., None] * paper_intensity * 0.35

        brightness_variation = gaussian_filter(
            rng.standard_normal((h, w)).astype(np.float32), sigma=25.0
        )
        brightness_variation /= (np.abs(brightness_variation).max() + 1e-8)
        paper = paper + brightness_variation[..., None] * paper_intensity * 0.25

        arr = arr * paper

        if moire_strength > 0.0:
            y = np.arange(h, dtype=np.float32)[:, None]
            x = np.arange(w, dtype=np.float32)[None, :]
            freq1 = 2 * np.pi / 4.3
            freq2 = 2 * np.pi / 4.7
            angle = 15.0 * np.pi / 180.0
            u = x * np.cos(angle) + y * np.sin(angle)
            v = -x * np.sin(angle) + y * np.cos(angle)
            moire = np.sin(u * freq1) * np.sin(v * freq2)
            arr = arr + moire[..., None] * moire_strength * 0.5

        arr = np.clip(arr, 0.0, 1.0)
        arr = arr ** (1.0 / gamma_shift)

        if color_cast == "warm":
            cast = np.array([1.03, 1.0, 0.96], dtype=np.float32)
        elif color_cast == "cool":
            cast = np.array([0.97, 1.0, 1.03], dtype=np.float32)
        elif color_cast == "yellow":
            cast = np.array([1.04, 1.02, 0.92], dtype=np.float32)
        else:
            cast = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        arr = np.clip(arr * cast[None, None, :], 0.0, 1.0)

        img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

        if abs(rotation_deg) > 0.01:
            img = img.rotate(rotation_deg, resample=Image.BICUBIC, expand=False, fillcolor=(240, 238, 233))

        img = img.filter(ImageFilter.GaussianBlur(radius=0.4))

        return img

    def get_guidance(self) -> str:
        return (
            "Parameter ranges: paper_intensity 0.08-0.15 subtle, 0.15-0.22 visible; "
            "rotation_deg 0.5-1.5 realistic handheld angle; moire_strength keep 0 "
            "unless simulating visible scan aliasing (may add pattern). "
            "Most useful when digital-pipeline signatures need to be broken."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "paper_intensity": {
                "type": "float",
                "min": 0.02,
                "max": 0.25,
                "description": "Paper fiber texture strength (0.08-0.15 subtle, 0.15-0.22 visible)",
                "default": 0.12,
                "required": False,
            },
            "moire_strength": {
                "type": "float",
                "min": 0.0,
                "max": 0.1,
                "description": "Moire interference pattern (keep 0.0 unless simulating scan aliasing)",
                "default": 0.0,
                "required": False,
            },
            "rotation_deg": {
                "type": "float",
                "min": -3.0,
                "max": 3.0,
                "description": "Slight misalignment from handheld re-photograph (0.5-1.5 realistic)",
                "default": 0.8,
                "required": False,
            },
            "gamma_shift": {
                "type": "float",
                "min": 0.85,
                "max": 1.20,
                "description": "Gamma shift from print+capture (1.05-1.15 typical darkening)",
                "default": 1.05,
                "required": False,
            },
            "color_cast": {
                "type": "str",
                "description": "Paper/capture color cast: 'warm', 'cool', 'yellow' (aged paper), 'neutral'",
                "default": "warm",
                "required": False,
            },
        }
