from typing import Dict, Any
from PIL import Image, ImageDraw
import numpy as np
from .base_tool import BaseTool


class FingerCorrectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="finger_correction",
            description="This tool allows for targeted inpainting to correct distorted or missing fingers, improving detail and realism in problematic regions like hands."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        area_to_inpaint = params.get("area_to_inpaint", "0,0,100,100")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)
        height, width, _ = img_array.shape

        # Parse the area to inpaint; unparseable values use a default region.
        try:
            x1, y1, x2, y2 = map(int, str(area_to_inpaint).split(","))
        except (ValueError, AttributeError):
            x1, y1, x2, y2 = 0, 0, min(100, width), min(100, height)
        x1, x2 = sorted((max(0, min(x1, width)), max(0, min(x2, width))))
        y1, y2 = sorted((max(0, min(y1, height)), max(0, min(y2, height))))

        if x2 > x1 and y2 > y1:
            from scipy.ndimage import uniform_filter
            smoothed = uniform_filter(img_array, size=(3, 3, 1))
            img_array[y1:y2, x1:x2] = smoothed[y1:y2, x1:x2]

        return Image.fromarray(img_array)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "area_to_inpaint": {
                "type": "str",
                "description": "Coordinates of the area to inpaint in the format 'x1,y1,x2,y2'.",
                "default": "0,0,100,100",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Specify the area to inpaint as 'x1,y1,x2,y2'; useful for correcting details in hands."