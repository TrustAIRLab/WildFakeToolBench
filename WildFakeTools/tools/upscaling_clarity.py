from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import zoom
from .base_tool import BaseTool


class UpscalingClarityTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="upscaling_clarity",
            description="This tool applies upscaling techniques to increase resolution and improve texture detail, making images appear more like real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        scale_factor = params.get("scale_factor", 2)
        algorithm = params.get("algorithm", "bicubic")

        # Convert image to RGB if not already
        image = image.convert("RGB")
        original_array = np.array(image)

        # Determine the new size
        new_size = (int(original_array.shape[1] * scale_factor), int(original_array.shape[0] * scale_factor))

        # Upscale the image using the specified algorithm
        if algorithm == "bicubic":
            upscaled_array = zoom(original_array, (scale_factor, scale_factor, 1), order=3)  # Bicubic interpolation
        elif algorithm == "lanczos":
            upscaled_array = zoom(original_array, (scale_factor, scale_factor, 1), order=5)  # Lanczos interpolation
        else:
            raise ValueError("Unsupported algorithm. Use 'bicubic' or 'lanczos'.")

        # Clip values to ensure valid pixel range and convert back to uint8
        upscaled_array = np.clip(upscaled_array, 0, 255).astype(np.uint8)

        # Create a new PIL image from the upscaled array
        upscaled_image = Image.fromarray(upscaled_array)

        # Resize back to original size to maintain the same output size
        upscaled_image = upscaled_image.resize(image.size, Image.LANCZOS)

        return upscaled_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "scale_factor": {
                "type": "float",
                "min": 2.0,
                "max": 4.0,
                "description": "Factor by which to upscale the image.",
                "default": 2.0,
                "required": False
            },
            "algorithm": {
                "type": "str",
                "min": None,
                "max": None,
                "description": "Upscaling algorithm to use: 'bicubic' or 'lanczos'.",
                "default": "bicubic",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Scale factor: 2-4; Algorithms: 'bicubic' or 'lanczos'. Use for enhancing AI-generated images."