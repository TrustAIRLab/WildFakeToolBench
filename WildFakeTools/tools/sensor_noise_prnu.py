"""
Sensor PRNU Noise Tool — camera Photo Response Non-Uniformity pattern.

Source: Reddit 18ueofn — "camera RAW methods" + sensor forensics literature.

Every physical camera sensor has a fixed spatial sensitivity pattern called PRNU
(Photo Response Non-Uniformity) caused by manufacturing variation in photodiode
size and doping. This pattern is invisible to the naked eye but detectable by
forensic tools — and crucially, AI images lack it entirely. Adding a realistic
PRNU pattern (fixed-seed spatially correlated multiplicative noise) gives images
a camera-device fingerprint without perceptible quality degradation.
"""
from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class SensorNoisePRNUTool(BaseTool):

    def __init__(self):
        super().__init__(
            name="sensor_noise_prnu",
            description=(
                "Adds a synthetic camera PRNU (Photo Response Non-Uniformity) pattern: "
                "a spatially-fixed, signal-dependent multiplicative noise layer that simulates "
                "the per-pixel sensitivity variation of a physical image sensor. Combines "
                "a smooth low-frequency sensitivity map with a correlated high-frequency "
                "component and row-level fixed-pattern noise (FPN). Imperceptible visually "
                "but introduces realistic sensor fingerprint statistics absent in AI images."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        prnu_strength = params.get("prnu_strength", 0.012)
        fpn_strength = params.get("fpn_strength", 0.004)
        seed = params.get("seed", 42)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0
        h, w = arr.shape[:2]

        rng = np.random.default_rng(seed)

        # Low-frequency sensitivity map (large-scale vignette-like variation)
        lf_prnu = rng.standard_normal((h // 8 + 1, w // 8 + 1)).astype(np.float32)
        lf_prnu = gaussian_filter(lf_prnu, sigma=3.0)
        from PIL import Image as _PIL
        lf_prnu = np.array(_PIL.fromarray(lf_prnu).resize((w, h), _PIL.Resampling.BILINEAR))
        lf_prnu /= (np.abs(lf_prnu).max() + 1e-8)

        # High-frequency PRNU: fixed per-pixel pattern (spatially correlated)
        hf_prnu = rng.standard_normal((h, w)).astype(np.float32)
        hf_prnu = gaussian_filter(hf_prnu, sigma=0.7)
        hf_prnu /= (np.abs(hf_prnu).max() + 1e-8)

        prnu = (lf_prnu * 0.4 + hf_prnu * 0.6) * prnu_strength

        # Row fixed-pattern noise (horizontal banding at sub-pixel level)
        row_fpn = rng.standard_normal(h).astype(np.float32) * fpn_strength
        row_fpn = gaussian_filter(row_fpn, sigma=0.5)
        row_fpn_2d = row_fpn[:, None]  # broadcast across columns

        # PRNU is multiplicative (signal-dependent), FPN is additive
        out = arr * (1.0 + prnu[..., None]) + row_fpn_2d[..., None]
        out = np.clip(out, 0.0, 1.0)
        return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")

    def get_guidance(self) -> str:
        return (
            "prnu_strength 0.008-0.015: realistic sensor fingerprint level; "
            "0.015-0.025: stronger (older/cheap sensor). "
            "fpn_strength 0.002-0.006: realistic row noise. "
            "seed is fixed per-call to simulate a consistent camera device. "
            "Visually imperceptible at realistic strengths but changes image statistics."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "prnu_strength": {
                "type": "float", "min": 0.003, "max": 0.030,
                "description": "PRNU pattern amplitude (0.008-0.015 realistic, 0.015-0.025 noisy sensor)",
                "default": 0.012, "required": False,
            },
            "fpn_strength": {
                "type": "float", "min": 0.0, "max": 0.015,
                "description": "Fixed-pattern row noise amplitude (0.002-0.006 realistic)",
                "default": 0.004, "required": False,
            },
            "seed": {
                "type": "int", "min": 0, "max": 9999,
                "description": "RNG seed for the PRNU pattern (same seed = same 'camera device')",
                "default": 42, "required": False,
            },
        }
