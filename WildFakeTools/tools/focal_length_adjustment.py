from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class FocalLengthAdjustmentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="focal_length_adjustment",
            description="Simulates focal length variations by applying a blur effect to the background of the image, enhancing the photographic quality."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        focal_length = params.get("focal_length", 50)  # Default focal length in mm
        focus_area = params.get("focus_area", (0.5, 0.5))  # Default focus area at the center

        # Convert image to RGB if needed
        image = image.convert("RGB")
        width, height = image.size

        # Create a mask for the focus area
        focus_x = int(focus_area[0] * width)
        focus_y = int(focus_area[1] * height)
        focus_radius = int(focal_length / 10)  # Simple mapping from focal length to radius

        # Create a blurred version of the image
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=max(1, focus_radius)))

        # Composite: sharp focus_radius box around (focus_x, focus_y), blurred elsewhere.
        mask = Image.new("L", (width, height), 0)
        x0 = max(0, focus_x - focus_radius)
        y0 = max(0, focus_y - focus_radius)
        x1 = min(width, focus_x + focus_radius)
        y1 = min(height, focus_y + focus_radius)
        if x1 > x0 and y1 > y0:
            mask.paste(255, (x0, y0, x1, y1))

        final_image = Image.composite(image, blurred_image, mask)
        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "focal_length": {
                "type": "int",
                "min": 10,
                "max": 200,
                "description": "Focal length in mm, affecting the amount of background blur.",
                "default": 50,
                "required": False
            },
            "focus_area": {
                "type": "array",
                "description": "Focus area as [x, y] where each is 0.0-1.0; (0,0) is top-left and (1,1) is bottom-right.",
                "default": (0.5, 0.5),
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Focal length (10-200 mm) adjusts background blur; focus area (0.0-1.0) determines the point of focus."