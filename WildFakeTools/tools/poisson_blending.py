from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class PoissonBlendingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="poisson_blending",
            description="This tool seamlessly blends AI-generated elements with real images by matching gradients and colors at the boundaries, making the inserted object appear as part of the original scene."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        blend_region = params.get("blend_region", (0, 0, image.width, image.height))
        smoothness = params.get("smoothness", 0.5)

        if isinstance(blend_region, str):
            try:
                x1, y1, x2, y2 = map(int, blend_region.split(","))
            except ValueError:
                x1, y1, x2, y2 = 0, 0, image.width, image.height
        else:
            x1, y1, x2, y2 = blend_region
        x1, x2 = sorted((max(0, min(x1, image.width)), max(0, min(x2, image.width))))
        y1, y2 = sorted((max(0, min(y1, image.height)), max(0, min(y2, image.height))))
        if x2 <= x1 or y2 <= y1:
            x1, y1, x2, y2 = 0, 0, image.width, image.height
        blend_region_crop = image.crop((x1, y1, x2, y2))

        # Convert to numpy array
        blend_array = np.array(blend_region_crop)

        # Apply Gaussian smoothing based on the smoothness parameter
        sigma = smoothness * 10  # Scale the smoothness to a reasonable sigma
        blended_array = gaussian_filter(blend_array, sigma=sigma)

        # Create a new image from the blended array
        blended_image = Image.fromarray(np.uint8(blended_array))

        # Paste the blended region back into the original image
        image.paste(blended_image, (x1, y1))

        return image.convert("RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "blend_region": {
                "type": "str",
                "description": "Rectangle defining the blend region as 'x1,y1,x2,y2'.",
                "default": "0,0,1024,1024",
                "required": False
            },
            "smoothness": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Controls the smoothness of the blending effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Blend region coordinates as 'x1,y1,x2,y2' and smoothness between 0.0 and 1.0 for optimal blending."