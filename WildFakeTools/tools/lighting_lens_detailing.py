from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class LightingLensDetailingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="lighting_lens_detailing",
            description="This tool enhances an image by simulating realistic lighting conditions and lens effects, making it appear more like a genuine photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters with defaults
        lighting_conditions = params.get("lighting_conditions", "soft")
        lens_description = params.get("lens_description", "50mm f/1.8")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Apply a Gaussian blur to simulate depth of field based on lens description
        if "f/1.8" in lens_description:
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
        elif "f/4" in lens_description:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))
        
        # Adjust brightness based on lighting conditions
        if lighting_conditions == "soft":
            image = self.adjust_brightness(image, factor=1.2)
        elif lighting_conditions == "harsh":
            image = self.adjust_brightness(image, factor=0.8)
        
        return image

    def adjust_brightness(self, image: Image.Image, factor: float) -> Image.Image:
        # Convert image to numpy array for manipulation
        img_array = np.array(image)
        img_array = np.clip(img_array * factor, 0, 255).astype(np.uint8)
        return Image.fromarray(img_array)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_conditions": {
                "type": "str",
                "description": "The lighting condition to simulate (e.g., 'soft', 'harsh').",
                "default": "soft",
                "required": False
            },
            "lens_description": {
                "type": "str",
                "description": "Description of the lens used (e.g., '50mm f/1.8').",
                "default": "50mm f/1.8",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust lighting conditions ('soft' or 'harsh') and lens description ('50mm f/1.8' or similar) to enhance realism."