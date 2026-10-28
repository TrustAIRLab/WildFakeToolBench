from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class PortraitSizeAdjustmentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="portrait_size_adjustment",
            description="Adjusts the aspect ratio of an image to portrait size, ensuring more of the subject's body is captured and reducing artifacts like missing limbs.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        aspect_ratio = params.get("aspect_ratio", "2:3")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Determine new dimensions based on aspect ratio
        width, height = image.size
        if aspect_ratio == "2:3":
            new_width = width
            new_height = int(width * 3 / 2)
        elif aspect_ratio == "3:4":
            new_width = width
            new_height = int(width * 4 / 3)
        elif aspect_ratio == "9:16":
            new_width = width
            new_height = int(width * 16 / 9)
        else:
            raise ValueError("Unsupported aspect ratio. Use '2:3', '3:4', or '9:16'.")

        # Resize the image to the new dimensions
        resized_image = image.resize((new_width, new_height), Image.LANCZOS)

        return resized_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "aspect_ratio": {
                "type": "str",
                "description": "The aspect ratio for the portrait size. Options are '2:3', '3:4', or '9:16'.",
                "default": "2:3",
                "required": False,
            }
        }

    def get_guidance(self) -> str:
        return "Aspect ratio options are '2:3', '3:4', or '9:16'. Use this tool to adjust images for better full-body representation."