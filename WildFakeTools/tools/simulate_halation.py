from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SimulateHalationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="simulate_halation",
            description="Simulates halation effects by creating a red-orange halo around bright areas in images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        halo_strength = params.get("halo_strength", 0.5)
        halo_color = params.get("halo_color", (255, 100, 0))  # Default to a red-orange color
        if isinstance(halo_color, str):
            try:
                halo_color = tuple(int(v) for v in halo_color.split(","))
            except ValueError:
                halo_color = (255, 100, 0)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        image_array = np.array(image)

        # Create a mask for bright areas
        brightness_mask = np.mean(image_array, axis=2) > 200  # Threshold for bright areas
        halo_layer = np.zeros_like(image_array)

        # Apply halo effect
        halo_layer[brightness_mask] = halo_color

        # Blend the halo with the original image
        blended_image = (image_array.astype(np.float32) * (1 - halo_strength) + 
                         halo_layer.astype(np.float32) * halo_strength).clip(0, 255).astype(np.uint8)

        return Image.fromarray(blended_image, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "halo_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the halo effect.",
                "default": 0.5,
                "required": False
            },
            "halo_color": {
                "type": "str",
                "description": "Color of the halo effect in RGB format (e.g., '255,100,0').",
                "default": "255,100,0",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust halo_strength (0.0-1.0) for effect intensity; halo_color for desired hue."