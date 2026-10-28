from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ImperfectionInclusionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="imperfection_inclusion",
            description="This tool adds small imperfections such as noise, grain, or blur to enhance the realism of an image."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        imperfection_type = params.get("imperfection_type", "noise")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert image to numpy array
        img_array = np.array(image)
        
        if imperfection_type == "noise":
            noise = np.random.normal(loc=0, scale=25, size=img_array.shape).astype(np.uint8)
            img_array = np.clip(img_array + noise, 0, 255)
        elif imperfection_type == "grain":
            grain = np.random.randint(0, 50, img_array.shape, dtype=np.uint8)
            img_array = np.clip(img_array + grain, 0, 255)
        elif imperfection_type == "blur":
            img = Image.fromarray(img_array)
            img = img.filter(ImageFilter.GaussianBlur(radius=1))
            img_array = np.array(img)
        
        # Ensure the array is still in the right format
        img_array = np.clip(img_array, 0, 255).astype(np.uint8)
        
        # Create a new PIL image from the modified array
        return Image.fromarray(img_array, 'RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "imperfection_type": {
                "type": "str",
                "description": "Type of imperfection to add (options: 'noise', 'grain', 'blur').",
                "default": "noise",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use 'imperfection_type' to specify the type of imperfection (noise, grain, blur) to enhance realism."