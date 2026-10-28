from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraSpecificTermsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_specific_terms",
            description="This tool enhances the realism of an image by applying camera-specific effects, such as depth of field and lens distortion, to make it appear more like a real photograph."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        camera_model = params.get("camera_model", "Canon EOS 5D")
        lens_type = params.get("lens_type", "50mm f/1.8")
        aperture = params.get("aperture", 1.8)

        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')

        # Apply a slight blur to simulate depth of field
        if aperture < 2.8:
            image = image.filter(ImageFilter.GaussianBlur(radius=2))
        elif aperture < 5.6:
            image = image.filter(ImageFilter.GaussianBlur(radius=1))

        # Add a vignette effect to simulate lens characteristics
        width, height = image.size
        x_center, y_center = width / 2, height / 2
        max_distance = np.sqrt(x_center**2 + y_center**2)

        vignette = np.zeros((height, width), dtype=np.float32)
        for y in range(height):
            for x in range(width):
                distance = np.sqrt((x - x_center) ** 2 + (y - y_center) ** 2)
                vignette[y, x] = max(0, 1 - (distance / max_distance))

        vignette = (vignette * 255).astype(np.uint8)
        vignette_image = Image.fromarray(vignette, mode='L')

        # Composite the vignette with the original image
        image = Image.composite(image, Image.new('RGB', image.size), vignette_image)

        return image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera_model": {
                "type": "str",
                "description": "The model of the camera used.",
                "default": "Canon EOS 5D",
                "required": False
            },
            "lens_type": {
                "type": "str",
                "description": "The type of lens used.",
                "default": "50mm f/1.8",
                "required": False
            },
            "aperture": {
                "type": "float",
                "min": 1.0,
                "max": 22.0,
                "description": "The aperture setting of the camera.",
                "default": 1.8,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust aperture between 1.0 and 22.0 for depth of field effects; use camera model and lens type for realism."