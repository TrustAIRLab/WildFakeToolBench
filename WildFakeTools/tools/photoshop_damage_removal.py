from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class PhotoshopDamageRemovalTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="photoshop_damage_removal",
            description="This tool refines an AI-generated image by simulating the removal of skin imperfections, creating a more realistic skin texture."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        refinement_steps = params.get("refinement_steps", 5)
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Simulate skin imperfection removal by applying Gaussian blur
        for _ in range(refinement_steps):
            img_array = np.clip(img_array + (np.random.normal(0, 10, img_array.shape)).astype(np.uint8), 0, 255)
            img_array = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=2))
            img_array = np.array(img_array)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "imperfection_types": {
                "type": "list",
                "description": "Types of skin imperfections to simulate.",
                "default": [],
                "required": False
            },
            "refinement_steps": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Number of refinement steps to apply for skin texture improvement.",
                "default": 5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Refinement steps: 1-10; useful for enhancing AI-generated images to appear more natural."