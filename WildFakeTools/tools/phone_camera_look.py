"""
Phone Camera Look Tool — simulates cheap smartphone capture pipeline.

Derived from r/StableDiffusion guidance that AI images look "too DSLR / too
cinematic". Real phone photos have: clipped highlights, shadow noise, warm/cool
auto-WB drift, aggressive in-camera JPEG, and micro-contrast loss.
"""

from typing import Dict, Any
from PIL import Image
import numpy as np
import io
from .base_tool import BaseTool


class PhoneCameraLookTool(BaseTool):
    """Simulate a phone camera pipeline: tone compression + shadow noise + WB shift + JPEG."""

    def __init__(self):
        super().__init__(
            name="phone_camera_look",
            description=(
                "Applies a cheap-phone-camera capture pipeline: clips highlights with "
                "soft shoulder compression, injects shadow-weighted sensor noise, shifts "
                "white-balance (warm/cool/neutral), reduces mid-tone contrast, and "
                "re-encodes with 4:2:0 chroma-subsampled JPEG. Compresses dynamic range "
                "and overlays amateur capture artifacts."
            )
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        highlight_clip = params.get("highlight_clip", 0.92)
        shadow_noise = params.get("shadow_noise", 0.04)
        color_temp = params.get("color_temp", "warm")
        jpeg_quality = params.get("jpeg_quality", 82)

        if image.mode != "RGB":
            image = image.convert("RGB")

        arr = np.asarray(image, dtype=np.float32) / 255.0

        arr = np.minimum(arr, highlight_clip)
        arr = arr / highlight_clip
        knee = 0.85
        mask = arr > knee
        arr = np.where(mask, knee + (arr - knee) * 0.55, arr)

        luminance = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        shadow_mask = np.clip(1.0 - luminance * 1.4, 0.0, 1.0)[..., None]
        rng = np.random.default_rng()
        noise = rng.standard_normal(arr.shape).astype(np.float32) * shadow_noise
        arr = arr + noise * shadow_mask

        if color_temp == "warm":
            wb = np.array([1.04, 1.0, 0.94], dtype=np.float32)
        elif color_temp == "cool":
            wb = np.array([0.95, 1.0, 1.05], dtype=np.float32)
        elif color_temp == "neutral":
            wb = np.array([1.0, 1.0, 1.0], dtype=np.float32)
        else:
            wb = np.array([1.04, 1.0, 0.94], dtype=np.float32)
        arr = arr * wb[None, None, :]

        contrast = 0.92
        mid = 0.5
        arr = (arr - mid) * contrast + mid

        arr = np.clip(arr, 0.0, 1.0)
        img = Image.fromarray((arr * 255.0).astype(np.uint8), mode="RGB")

        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=int(jpeg_quality), subsampling=2)
        buf.seek(0)
        return Image.open(buf).copy()

    def get_guidance(self) -> str:
        return (
            "Parameter ranges: highlight_clip 0.85-0.95 matches typical phone DR; "
            "shadow_noise 0.03-0.07 realistic; jpeg_quality 75-88 mid-tier phone output. "
            "Most useful when the image looks overly polished or cinematic."
        )

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "highlight_clip": {
                "type": "float",
                "min": 0.80,
                "max": 0.99,
                "description": "Highlight clip point (0.85-0.95 = typical phone dynamic range)",
                "default": 0.92,
                "required": False,
            },
            "shadow_noise": {
                "type": "float",
                "min": 0.0,
                "max": 0.12,
                "description": "Shadow-region noise intensity (0.03-0.07 realistic, >0.08 aggressive)",
                "default": 0.04,
                "required": False,
            },
            "color_temp": {
                "type": "str",
                "description": "Auto-WB drift: 'warm' (indoor/golden), 'cool' (shade), 'neutral'",
                "default": "warm",
                "required": False,
            },
            "jpeg_quality": {
                "type": "int",
                "min": 60,
                "max": 95,
                "description": "In-camera JPEG quality (75-88 realistic for mid-tier phones)",
                "default": 82,
                "required": False,
            },
        }
