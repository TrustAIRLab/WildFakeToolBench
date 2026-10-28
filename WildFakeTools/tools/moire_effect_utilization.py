from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class MoirEffectUtilizationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="moire_effect_utilization",
            description="Introduce the moiré effect to enhance texture realism in AI-generated images, creating periodic patterns that mimic real photographic artifacts."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        capture_distance = params.get("capture_distance", 5.0)
        pattern_frequency = params.get("pattern_frequency", 10.0)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Create a moiré pattern
        height, width, _ = img_array.shape
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        X, Y = np.meshgrid(x, y)

        # Generate a sine wave pattern
        pattern = (np.sin(X / capture_distance * np.pi * pattern_frequency) + 
                   np.sin(Y / capture_distance * np.pi * pattern_frequency)) * 127.5 + 127.5
        pattern = pattern.astype(np.uint8)

        # Blend the original image with the pattern
        blended_image = np.clip(img_array.astype(np.int32) + pattern[:, :, np.newaxis].astype(np.int32), 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        result_image = Image.fromarray(blended_image, "RGB")
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "capture_distance": {
                "type": "float",
                "min": 1.0,
                "max": 20.0,
                "description": "Distance simulating the capture distance of the image.",
                "default": 5.0,
                "required": False
            },
            "pattern_frequency": {
                "type": "float",
                "min": 1.0,
                "max": 30.0,
                "description": "Frequency of the moiré pattern to be applied.",
                "default": 10.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters range: capture_distance (1.0 to 20.0), pattern_frequency (1.0 to 30.0). Useful for adding realistic texture."