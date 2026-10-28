from typing import Dict, Any
from PIL import Image, ImageDraw
import numpy as np
from .base_tool import BaseTool


class HandFaceCorrectionTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="hand_face_correction",
            description="This tool selectively regenerates problem areas in images, such as hands and faces, to improve anatomical accuracy and realism. It uses a simple numeric approximation to simulate inpainting effects."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        region = params.get("region", (0, 0, image.width, image.height))
        inpainting_strength = params.get("inpainting_strength", 0.5)

        # Convert image to numpy array
        img_array = np.array(image)

        # Define the region for inpainting
        x1, y1, x2, y2 = region
        inpaint_region = img_array[y1:y2, x1:x2]

        # Apply a simple blurring effect to simulate inpainting
        kernel_size = int(5 * inpainting_strength) + 1
        if kernel_size % 2 == 0:
            kernel_size += 1  # Ensure kernel size is odd
        blur = np.ones((kernel_size, kernel_size), np.float32) / (kernel_size ** 2)
        inpainted_region = np.clip(np.convolve(inpaint_region.flatten(), blur.flatten(), mode='same').reshape(inpaint_region.shape), 0, 255).astype(np.uint8)

        # Replace the inpainted region in the original image
        img_array[y1:y2, x1:x2] = inpainted_region

        # Convert back to PIL Image
        return Image.fromarray(img_array, 'RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "region": {
                "type": "array",
                "description": "Bounding box for the region to inpaint as [x1, y1, x2, y2] (integer pixel coordinates).",
                "default": (0, 0, 100, 100),
                "required": False
            },
            "inpainting_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the inpainting effect, where 0.0 is no effect and 1.0 is maximum effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Region coordinates as (x1, y1, x2, y2) and inpainting strength from 0.0 to 1.0 for targeted corrections."