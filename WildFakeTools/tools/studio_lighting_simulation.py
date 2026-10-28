from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class StudioLightingSimulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="studio_lighting_simulation",
            description="Adjusts the lighting of an image to simulate realistic lighting conditions like golden hour or studio light."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        strength = params.get("strength", 0.5)
        lighting_type = params.get("lighting_type", "golden hour")

        # Convert image to RGB if not already
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Convert image to numpy array for manipulation
        img_array = np.array(image)

        # Apply lighting adjustments based on the selected type
        if lighting_type == "golden hour":
            # Simulate warm lighting by adjusting the red and green channels
            img_array = self.apply_golden_hour_lighting(img_array, strength)
        elif lighting_type == "studio":
            # Simulate neutral bright lighting by enhancing brightness and contrast
            img_array = self.apply_studio_lighting(img_array, strength)

        # Convert back to PIL Image
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def apply_golden_hour_lighting(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        # Increase red and green channels to simulate warm lighting
        img_array[..., 0] = img_array[..., 0] * (1 + 0.5 * strength)  # Red channel
        img_array[..., 1] = img_array[..., 1] * (1 + 0.3 * strength)  # Green channel
        return img_array

    def apply_studio_lighting(self, img_array: np.ndarray, strength: float) -> np.ndarray:
        # Enhance brightness and contrast
        img_array = Image.fromarray(img_array)
        enhancer = ImageEnhance.Brightness(img_array)
        img_array = enhancer.enhance(1 + 0.5 * strength)  # Brightness enhancement
        enhancer = ImageEnhance.Contrast(img_array)
        img_array = enhancer.enhance(1 + 0.5 * strength)  # Contrast enhancement
        return np.array(img_array)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "The strength of the lighting effect.",
                "default": 0.5,
                "required": False
            },
            "lighting_type": {
                "type": "str",
                "min": None,
                "max": None,
                "description": "The type of lighting to simulate (e.g., 'golden hour', 'studio').",
                "default": "golden hour",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Strength: 0.0-1.0; Lighting types: 'golden hour' or 'studio' for realistic lighting effects."