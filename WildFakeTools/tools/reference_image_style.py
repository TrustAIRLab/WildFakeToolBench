from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ReferenceImageStyleTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="reference_image_style",
            description="This tool applies a style transformation to an image by utilizing a blurred version of the input image as a reference, creating a more natural appearance."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Create a reference style by blurring the input image
        reference_style = image.filter(ImageFilter.GaussianBlur(radius=5))
        
        # Convert images to numpy arrays
        image_array = np.array(image)
        reference_array = np.array(reference_style)
        
        # Blend the original image with the reference style
        blended_array = np.clip(image_array * 0.7 + reference_array * 0.3, 0, 255).astype(np.uint8)
        
        # Convert back to PIL Image
        blended_image = Image.fromarray(blended_array, 'RGB')
        return blended_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {}

    def get_guidance(self) -> str:
        return "No parameters needed; this tool is most useful for enhancing the realism of AI-generated images."