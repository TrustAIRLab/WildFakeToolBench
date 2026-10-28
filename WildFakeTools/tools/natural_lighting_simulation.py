from typing import Dict, Any
from PIL import Image, ImageEnhance
import numpy as np
from .base_tool import BaseTool


class NaturalLightingSimulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="natural_lighting_simulation",
            description="Simulates natural lighting conditions with realistic shadows and highlights, enhancing the three-dimensionality of the image."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters
        shadow_intensity = params.get("shadow_intensity", 0.5)
        lighting_conditions = params.get("lighting_conditions", "soft")

        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Simulate lighting effects
        if lighting_conditions == "soft":
            # Apply a soft light effect
            img_array = self.apply_soft_light(img_array, shadow_intensity)
        elif lighting_conditions == "harsh":
            # Apply a harsh light effect
            img_array = self.apply_harsh_light(img_array, shadow_intensity)

        # Convert back to PIL Image
        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def apply_soft_light(self, img_array: np.ndarray, shadow_intensity: float) -> np.ndarray:
        # Create soft light effect
        shadow_mask = np.full(img_array.shape, 255 * shadow_intensity, dtype=np.uint8)
        img_array = img_array * (1 - shadow_intensity) + shadow_mask * shadow_intensity
        return img_array

    def apply_harsh_light(self, img_array: np.ndarray, shadow_intensity: float) -> np.ndarray:
        # Create harsh light effect
        shadow_mask = np.full(img_array.shape, 255 * shadow_intensity, dtype=np.uint8)
        img_array = img_array * (1 - shadow_intensity) + shadow_mask * (2 * shadow_intensity)
        return img_array

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "shadow_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the shadow effect.",
                "default": 0.5,
                "required": False
            },
            "lighting_conditions": {
                "type": "str",
                "description": "Type of lighting condition ('soft' or 'harsh').",
                "default": "soft",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust shadow intensity (0-1) and choose between soft or harsh lighting for realistic effects."