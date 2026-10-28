from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class ColorTransferTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="color_transfer",
            description="Adjusts the color statistics of AI-generated images to better mimic real photographs by harmonizing global color statistics.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        transfer_method = params.get("transfer_method", "mean")
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        image_np = np.array(image).astype(np.float32) / 255.0
        
        # Create a reference image by blurring the input image to simulate a target
        reference_image_np = np.clip(image_np.mean(axis=(0, 1)), 0, 1)
        
        # Perform color transfer
        image_mean = image_np.mean(axis=(0, 1))
        if transfer_method == "mean":
            image_np += reference_image_np - image_mean
        elif transfer_method == "variance":
            # Recenter the image on the reference means.
            image_np = image_np - image_mean + reference_image_np
        
        # Clip values to [0, 1] and convert back to uint8
        image_np = np.clip(image_np, 0, 1)
        result_image = Image.fromarray((image_np * 255).astype(np.uint8), mode="RGB")
        
        return result_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "transfer_method": {
                "type": "str",
                "description": "Method for color transfer: 'mean' or 'variance'.",
                "default": "mean",
                "required": False,
            }
        }

    def get_guidance(self) -> str:
        return "Use 'mean' for basic color adjustment or 'variance' for more complex adjustments. Suitable for enhancing AI-generated images."