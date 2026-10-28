from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .base_tool import BaseTool


class ScenarioInclusionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="scenario_inclusion",
            description="This tool adds context and narrative to an image by overlaying a specified scenario text, making it more relatable and realistic."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        scenario = params.get("scenario", "A beautiful day in the park.")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a draw object
        draw = ImageDraw.Draw(image)
        
        # Load a default font
        font = ImageFont.load_default()
        
        # Positioning the text
        text_position = (10, 10)  # Top-left corner
        draw.text(text_position, scenario, fill="white", font=font)
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "scenario": {
                "type": "str",
                "description": "A descriptive scenario to overlay on the image.",
                "default": "A beautiful day in the park.",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use a descriptive scenario string to add context to the image. Default is a simple park scenario."