from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class AvoidBeautyFilterTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="avoid_beauty_filter",
            description="This tool enhances the natural texture of skin in images by applying a subtle sharpening effect, preserving imperfections and details that beauty filters often smooth out."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        image = image.convert("RGB")
        
        # Convert image to numpy array for processing
        img_array = np.array(image)
        
        # Apply a subtle sharpening filter to enhance details
        sharpened_image = Image.fromarray(img_array).filter(ImageFilter.SHARPEN)
        
        return sharpened_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "filter_instruction": {
                "type": "str",
                "description": "Instruction to avoid beauty filters (not used in processing).",
                "default": "Avoid beauty filters.",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "No parameters are needed; this tool subtly sharpens images to enhance natural skin textures."