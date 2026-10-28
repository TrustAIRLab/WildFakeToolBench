from typing import Dict, Any
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np
from .base_tool import BaseTool


class PhotographicBriefRealismTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="photographic_brief_realism",
            description="Enhances an image to make it appear more realistic by adjusting lighting, materials, and adding imperfections."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        
        # Read parameters
        lighting_details = params.get("lighting_details", "bright")
        material_types = params.get("material_types", "rough")
        imperfection_types = params.get("imperfection_types", "none")
        
        # Apply lighting adjustments
        if lighting_details == "bright":
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(1.2)  # Brighten the image
        elif lighting_details == "dim":
            enhancer = ImageEnhance.Brightness(image)
            image = enhancer.enhance(0.8)  # Dim the image
        
        # Apply material adjustments
        if material_types == "rough":
            image = image.filter(ImageFilter.GaussianBlur(radius=1))  # Simulate roughness
        
        # Apply imperfections
        if imperfection_types == "noise":
            noise = np.random.normal(0, 25, (image.size[1], image.size[0], 3)).astype(np.uint8)
            noisy_image = np.array(image) + noise
            noisy_image = np.clip(noisy_image, 0, 255)
            image = Image.fromarray(noisy_image.astype(np.uint8))
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "lighting_details": {
                "type": "str",
                "description": "Specify the lighting condition (e.g., 'normal', 'bright', 'dim').",
                "default": "bright",
                "required": False
            },
            "material_types": {
                "type": "str",
                "description": "Specify the material type (e.g., 'smooth', 'rough').",
                "default": "rough",
                "required": False
            },
            "imperfection_types": {
                "type": "str",
                "description": "Specify the type of imperfections to add (e.g., 'none', 'noise').",
                "default": "none",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: lighting (normal/bright/dim), material (smooth/rough), imperfections (none/noise). Use to enhance realism."