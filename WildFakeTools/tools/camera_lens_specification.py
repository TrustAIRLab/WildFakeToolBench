from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraLensSpecificationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_lens_specification",
            description="This tool enhances an image by simulating the effects of specific camera and lens specifications, adding depth and realism to AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        camera = params.get("camera", "Standard Camera")
        lens = params.get("lens", "50mm f/1.8")
        
        # Convert image to RGB if not already
        image = image.convert("RGB")
        
        # Simulate lens effects by applying a slight blur and vignette
        image = self.apply_lens_effects(image, lens)
        
        # Add a subtle sharpening effect to enhance details
        image = image.filter(ImageFilter.SHARPEN)
        
        return image

    def apply_lens_effects(self, image: Image.Image, lens: str) -> Image.Image:
        # Create a vignette effect based on the lens specification
        width, height = image.size
        vignette = np.zeros((height, width), dtype=np.float32)

        # Create a radial gradient for the vignette
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - width / 2) ** 2 + (y - height / 2) ** 2)
                vignette[y, x] = max(0, 1 - distance / (min(width, height) / 2))

        vignette = (vignette * 255).astype(np.uint8)
        vignette_image = Image.fromarray(vignette, mode='L')

        # Apply the vignette to the image
        image = Image.composite(image, image.filter(ImageFilter.GaussianBlur(5)), vignette_image)
        
        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera": {
                "type": "str",
                "description": "The camera model to simulate.",
                "default": "Standard Camera",
                "required": False
            },
            "lens": {
                "type": "str",
                "description": "The lens specification to simulate.",
                "default": "50mm f/1.8",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: camera (string), lens (string); use to simulate photographic effects."