from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class HighResFacialRefinementTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="high_res_facial_refinement",
            description="Enhances facial features in images to improve detail and realism, making them appear more like real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        refiner_strength = params.get("refiner_strength", 5)
        
        # Convert image to RGB if not already
        image = image.convert("RGB")
        
        # Apply a series of filters to enhance details
        # The strength parameter will control the intensity of the enhancement
        for _ in range(int(refiner_strength)):
            image = image.filter(ImageFilter.DETAIL)
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "refiner_strength": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Strength of the refiner pass, higher values enhance details more.",
                "default": 5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Refiner strength ranges from 1 to 10; use higher values for more pronounced detail enhancement."