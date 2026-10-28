from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class IntentionalCompositionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="intentional_composition",
            description="This tool adjusts the composition of an image by repositioning the subject and adding negative space, creating a more artistic and less generic appearance."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        subject_position = params.get("subject_position", "left")
        negative_space_ratio = params.get("negative_space_ratio", 0.03)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        width, height = image.size
        img_array = np.array(image)

        # Calculate new subject position based on the specified parameter
        if subject_position == "left":
            offset_x = -width * negative_space_ratio
            offset_y = 0
        elif subject_position == "right":
            offset_x = width * negative_space_ratio
            offset_y = 0
        elif subject_position == "top":
            offset_x = 0
            offset_y = -height * negative_space_ratio
        elif subject_position == "bottom":
            offset_x = 0
            offset_y = height * negative_space_ratio
        else:  # center
            offset_x = 0
            offset_y = 0

        # Create a new image with negative space
        new_width = int(width + abs(offset_x))
        new_height = int(height + abs(offset_y))
        new_image = np.zeros((new_height, new_width, 3), dtype=np.uint8)

        # Calculate the position to paste the original image
        paste_x = int(max(0, offset_x)) if offset_x > 0 else 0
        paste_y = int(max(0, offset_y)) if offset_y > 0 else 0

        # Paste the original image into the new image
        new_image[paste_y:paste_y + height, paste_x:paste_x + width] = img_array

        return Image.fromarray(new_image)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "subject_position": {
                "type": "str",
                "description": "Position of the subject in the image (left, right, top, bottom, center).",
                "default": "left",
                "required": False
            },
            "negative_space_ratio": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Ratio of negative space to be added around the subject.",
                "default": 0.03,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust subject position (left, right, top, bottom, center) and negative space ratio (0-1) to create artistic compositions."