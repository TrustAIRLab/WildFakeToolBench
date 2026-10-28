from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class SimulateLightingMismatchesTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simulate_lighting_mismatches",
            description="Introduces minor lighting mismatches to enhance the realism of AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        mismatch_intensity = params.get("mismatch_intensity", 0.1)
        mismatch_color = params.get("mismatch_color", (1.0, 1.0, 1.0))

        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Generate random lighting mismatches
        height, width, _ = img_array.shape
        random_light = np.random.uniform(-mismatch_intensity, mismatch_intensity, (height, width, 1))
        color_adjustment = np.array(mismatch_color).reshape(1, 1, 3)  # Shape for broadcasting

        # Apply the lighting mismatch
        img_array = img_array * (1 + random_light * color_adjustment)
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "mismatch_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the lighting mismatch.",
                "default": 0.1,
                "required": False
            },
            "mismatch_color": {
                "type": "array",
                "description": "Color adjustment for the lighting mismatch, as an RGB tuple [r, g, b] with each channel in 0.0-1.0.",
                "default": (1.0, 1.0, 1.0),
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust mismatch_intensity (0-1) and mismatch_color (RGB tuple) to simulate lighting inconsistencies."