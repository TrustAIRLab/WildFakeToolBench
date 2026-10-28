from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .base_tool import BaseTool


class FilenameAppendingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="filename_appending",
            description="Append a filename in the format used by smartphones to enhance realism.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        filename = params.get("filename", "image.jpg")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Create a draw object
        draw = ImageDraw.Draw(image)

        # Load a default font
        font = ImageFont.load_default()

        # Define text position and color
        text_position = (10, image.height - 30)  # Bottom left corner
        text_color = (255, 255, 255)  # White color

        # Draw the text on the image
        draw.text(text_position, filename, fill=text_color, font=font)

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "filename": {
                "type": "str",
                "description": "The filename to append to the image.",
                "default": "image.jpg",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Append a filename to enhance realism; use any string as filename."