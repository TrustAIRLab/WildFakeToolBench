"""
Natural Color Correction Tool — removes artificial saturation and highlight boost.

Source: r/StableDiffusion post about Flux.2 Klein LoRA:
"Addresses the tendency of many models to boost saturation artificially.
The output aims to match the true-to-life color tones."

AI-generated images consistently exhibit: (1) chrominance over-saturation relative
to real photography, (2) blown-out specular highlights with unnatural color purity,
(3) overly uniform micro-contrast. This tool corrects these three artifacts using
HSV-space saturation compression, highlight tone-mapping, and local contrast
variance adjustment.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class NaturalColorCorrectionTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="natural_color_correction",
            description=(
                "Corrects AI-typical chromatic over-saturation and blown highlights: "
                "compresses HSV saturation with a sigmoid curve (pulls vivid zones toward "
                "film-like chroma), applies highlight tone-mapping to replace pure-white "
                "blowout with a warm paper-white shoulder, and adds subtle micro-contrast "
                "variance to break the uniformly sharp AI look. Targets the 'true-to-life "
                "color tones' correction from Flux.2 Klein LoRA community findings."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        sat_compression = params.get("sat_compression", 0.6)
        highlight_rolloff = params.get("highlight_rolloff", 0.88)
        micro_contrast = params.get("micro_contrast", 0.04)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0

        # --- Saturation compression in HSV space ---
        r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
        cmax = np.maximum.reduce([r, g, b])
        cmin = np.minimum.reduce([r, g, b])
        delta = cmax - cmin + 1e-8

        # Compute HSV
        v = cmax
        s = np.where(cmax > 0, delta / (cmax + 1e-8), 0.0)

        h = np.zeros_like(r)
        mask_r = (cmax == r) & (delta > 0)
        mask_g = (cmax == g) & (delta > 0)
        mask_b = (cmax == b) & (delta > 0)
        h[mask_r] = ((g[mask_r] - b[mask_r]) / delta[mask_r]) % 6
        h[mask_g] = (b[mask_g] - r[mask_g]) / delta[mask_g] + 2
        h[mask_b] = (r[mask_b] - g[mask_b]) / delta[mask_b] + 4
        h = h / 6.0  # normalise to [0,1]

        # Sigmoid-curve saturation compression:
        # s → sat_compression * sigmoid-like mapping that pulls high-S zones down
        compressed_s = s * (1.0 - sat_compression * (s ** 1.5))
        compressed_s = np.clip(compressed_s, 0.0, 1.0)

        # Reconstruct RGB from modified HSV
        h6 = h * 6.0
        i = np.floor(h6).astype(int) % 6
        f = h6 - np.floor(h6)
        p = v * (1.0 - compressed_s)
        q = v * (1.0 - compressed_s * f)
        t = v * (1.0 - compressed_s * (1.0 - f))
        v_ = v

        out = np.zeros_like(arr)
        for idx, (r_ch, g_ch, b_ch) in enumerate(
            [(v_, t, p), (q, v_, p), (p, v_, t), (p, q, v_), (t, p, v_), (v_, p, q)]
        ):
            mask = (i == idx)
            out[mask, 0] = r_ch[mask]
            out[mask, 1] = g_ch[mask]
            out[mask, 2] = b_ch[mask]

        arr = out

        # --- Highlight rolloff: compress near-white blowout ---
        knee = highlight_rolloff
        mask_hi = arr > knee
        arr = np.where(
            mask_hi,
            knee + (arr - knee) * 0.35,  # compress highlights above rolloff
            arr
        )
        # Slight warm cast in highlights: use luminance-based scalar mask
        lum_hi = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        hi_strength = np.clip((lum_hi - knee * 0.95) / (1.0 - knee * 0.95 + 1e-8), 0.0, 1.0)
        arr[..., 2] -= 0.012 * hi_strength  # very slight blue reduction in highlights

        # --- Micro-contrast variance: break uniform AI sharpness ---
        if micro_contrast > 0:
            luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
            local_mean = gaussian_filter(luminance, sigma=3.0)
            local_var = gaussian_filter((luminance - local_mean) ** 2, sigma=3.0)
            # In flat regions (low variance), add subtle spatial variation
            flat_mask = np.clip(1.0 - local_var * 80, 0.0, 1.0)[..., None]
            rng = np.random.default_rng()
            variation = rng.standard_normal(arr.shape).astype(np.float32)
            for c in range(3):
                variation[..., c] = gaussian_filter(variation[..., c], sigma=4.0)
            variation /= (np.abs(variation).max() + 1e-8)
            arr = arr + variation * micro_contrast * flat_mask

        arr = np.clip(arr, 0.0, 1.0)
        return Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "sat_compression 0.3-0.5 gentle desaturation; 0.6-0.8 noticeable. "
            "highlight_rolloff 0.85-0.92 compresses blown highlights; lower = more rolloff. "
            "micro_contrast 0.02-0.05 subtle variance; >0.07 visible noise in flat regions. "
            "Most effective when the image looks over-vivid or has perfect white areas."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "sat_compression": {
                "type": "float",
                "min": 0.1,
                "max": 0.9,
                "description": "Saturation compression strength via sigmoid curve (0.3-0.5 gentle, 0.6-0.8 strong)",
                "default": 0.55,
                "required": False,
            },
            "highlight_rolloff": {
                "type": "float",
                "min": 0.75,
                "max": 0.97,
                "description": "Highlight knee point — above this, brightness is compressed (0.85-0.92 natural)",
                "default": 0.88,
                "required": False,
            },
            "micro_contrast": {
                "type": "float",
                "min": 0.0,
                "max": 0.10,
                "description": "Micro-contrast variance added to flat regions (0.02-0.05 subtle)",
                "default": 0.04,
                "required": False,
            },
        }
