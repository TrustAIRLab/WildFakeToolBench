from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ApertureSimulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="aperture_simulation",
            description="Simulates depth of field effects by applying a blur to areas outside a specified focus region, enhancing the realism of AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        aperture_size = params.get("aperture_size", 5)
        # Default focus area is the central half of the frame.
        focus_area = params.get(
            "focus_area",
            (image.width // 4, image.height // 4,
             image.width - image.width // 4, image.height - image.height // 4),
        )

        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_np = np.array(image)

        # Create a mask for the focus area
        mask = np.zeros(image_np.shape[:2], dtype=np.uint8)
        mask[focus_area[1]:focus_area[3], focus_area[0]:focus_area[2]] = 255
        
        # Apply Gaussian blur to the entire image
        blurred_image = Image.fromarray(np.uint8(Image.fromarray(image_np).filter(ImageFilter.GaussianBlur(radius=aperture_size))))
        
        # Combine the blurred image with the original image using the mask
        result_image = Image.composite(image, blurred_image, Image.fromarray(mask))

        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "aperture_size": {
                "type": "int",
                "min": 1,
                "max": 20,
                "description": "Size of the aperture effect, controlling the amount of blur applied.",
                "default": 5,
                "required": False
            },
            "focus_area": {
                "type": "tuple",
                "description": "Coordinates of the focus area as (left, upper, right, lower).",
                "default": (0, 0, 1024, 1024),
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Aperture size: 1-20, focus area: coordinates for the region of interest. Best for enhancing realism in AI-generated images."