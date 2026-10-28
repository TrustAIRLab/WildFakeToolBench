from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class HighIsoGrainTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="high_iso_grain",
            description="Simulates high ISO settings to introduce grainy noise typical of low-light photography, enhancing the authenticity of AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        iso_value = params.get("iso_value", 1600)
        grain_strength = params.get("grain_strength", 0.5)

        # Convert image to RGB if not already in that mode
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array
        img_array = np.array(image)

        # Generate grain noise
        noise = np.random.normal(0, iso_value / 1000, img_array.shape).astype(np.uint8)

        # Blend the noise with the original image based on grain strength
        noisy_image = np.clip(img_array + (noise * grain_strength), 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(noisy_image, 'RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "iso_value": {
                "type": "int",
                "min": 800,
                "max": 3200,
                "description": "ISO value simulating low-light conditions.",
                "default": 1600,
                "required": False
            },
            "grain_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the grain effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "ISO value (800-3200) and grain strength (0.0-1.0) to simulate low-light photography effects."