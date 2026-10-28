from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class CameraAndLensDetailsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_and_lens_details",
            description="Enhances the realism of images by simulating depth of field and optical effects typical of real photographs."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Read parameters with defaults
        camera = params.get("camera", "Standard Camera")
        lens = params.get("lens", "50mm f/1.8")
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Simulate depth of field by applying a Gaussian blur
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=2))
        
        # Create a mask for depth of field effect
        mask = np.zeros((image.height, image.width), dtype=np.uint8)
        mask[image.height // 4:image.height * 3 // 4, image.width // 4:image.width * 3 // 4] = 255
        
        # Convert images to numpy arrays
        image_np = np.array(image)
        blurred_np = np.array(blurred_image)
        
        # Blend the original and blurred images based on the mask
        for i in range(3):  # For each channel
            image_np[:, :, i] = np.where(mask == 255, image_np[:, :, i], blurred_np[:, :, i])
        
        # Convert back to PIL Image
        final_image = Image.fromarray(image_np)
        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera": {
                "type": "str",
                "description": "Camera model used for the image.",
                "default": "Standard Camera",
                "required": False,
            },
            "lens": {
                "type": "str",
                "description": "Lens type used for the image.",
                "default": "50mm f/1.8",
                "required": False,
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: camera (string), lens (string); useful for enhancing realism in AI-generated images."