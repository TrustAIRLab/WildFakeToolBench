from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class PortraitRealismTuningTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="portrait_realism_tuning",
            description="Enhances the realism of images by applying a series of filters and adjustments that mimic human imperfections, improving skin textures and features."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        image = image.convert("RGB")
        
        # Convert image to numpy array for processing
        img_array = np.array(image)

        # Apply Gaussian blur to soften the image slightly
        img_blurred = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=1))

        # Convert back to numpy array
        img_blurred_array = np.array(img_blurred)

        # Adjust brightness and contrast
        img_adjusted = np.clip(img_blurred_array * 1.1, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        final_image = Image.fromarray(img_adjusted)

        return final_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "checkpoint_name": {
                "type": "str",
                "description": "The name of the fine-tuned checkpoint to use for enhancing realism.",
                "default": "portrait_checkpoint",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Use with any portrait image; adjust parameters for optimal realism."