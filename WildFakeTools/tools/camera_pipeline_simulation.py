from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class CameraPipelineSimulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_pipeline_simulation",
            description="Simulates camera pipeline effects such as Bayer/demosaic, JPEG compression artifacts, and chromatic aberration, making AI-generated images appear more authentic."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        effect_type = params.get("effect_type", "Bayer")
        intensity = params.get("intensity", 0.5)

        image = image.convert("RGB")
        img_array = np.array(image)

        if effect_type == "Bayer":
            img_array = self.apply_bayer_demosaic(img_array, intensity)
        elif effect_type == "JPEG":
            img_array = self.apply_jpeg_compression(img_array, intensity)
        elif effect_type == "chromatic_aberration":
            img_array = self.apply_chromatic_aberration(img_array, intensity)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def apply_bayer_demosaic(self, img_array: np.ndarray, intensity: float) -> np.ndarray:
        # Simple Bayer pattern simulation
        bayer_pattern = np.zeros_like(img_array)
        bayer_pattern[::2, ::2, 0] = img_array[::2, ::2, 0]  # Red
        bayer_pattern[::2, 1::2, 1] = img_array[::2, 1::2, 1]  # Green
        bayer_pattern[1::2, ::2, 1] = img_array[1::2, ::2, 1]  # Green
        bayer_pattern[1::2, 1::2, 2] = img_array[1::2, 1::2, 2]  # Blue

        return bayer_pattern * intensity + img_array * (1 - intensity)

    def apply_jpeg_compression(self, img_array: np.ndarray, intensity: float) -> np.ndarray:
        # Simulate JPEG compression artifacts
        noise = np.random.normal(0, 10 * intensity, img_array.shape)
        return img_array + noise

    def apply_chromatic_aberration(self, img_array: np.ndarray, intensity: float) -> np.ndarray:
        # Apply a simple chromatic aberration effect
        shift = int(5 * intensity)
        red_shifted = np.roll(img_array[:, :, 0], shift, axis=1)
        blue_shifted = np.roll(img_array[:, :, 2], -shift, axis=1)
        img_array[:, :, 0] = red_shifted
        img_array[:, :, 2] = blue_shifted
        return img_array

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "effect_type": {
                "type": "str",
                "description": "Type of effect to apply (Bayer, JPEG, chromatic_aberration).",
                "default": "Bayer",
                "required": False
            },
            "intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity ranges from 0.0 (no effect) to 1.0 (full effect); useful for adding realism to AI-generated images."