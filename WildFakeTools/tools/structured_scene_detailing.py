from typing import Dict, Any
from PIL import Image, ImageDraw, ImageFont
import numpy as np
from .base_tool import BaseTool


class StructuredSceneDetailingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="structured_scene_detailing",
            description="This tool generates a structured prompt that includes environmental and technical details to enhance realism in AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        structured_prompt = params.get("structured_prompt", "A realistic scene with detailed environmental features.")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a new image to overlay the prompt
        overlay_image = Image.new("RGB", image.size)
        draw = ImageDraw.Draw(overlay_image)
        
        # Load a default font
        font_size = 20
        font = ImageFont.load_default()
        
        # Draw the structured prompt on the overlay image
        draw.text((10, 10), structured_prompt, fill=(255, 255, 255), font=font)
        
        # Blend the original image with the overlay
        blended_image = Image.blend(image, overlay_image, alpha=0.3)
        
        return blended_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "structured_prompt": {
                "type": "str",
                "description": "A structured prompt including environmental and technical details.",
                "default": "A realistic scene with detailed environmental features.",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Provide a structured prompt with environmental and technical details for better realism. Default is a generic prompt."