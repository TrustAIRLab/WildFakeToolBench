from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class ExposureAdjustmentTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="exposure_adjustment",
            description="Adjusts exposure and color settings to create a more natural look, reducing the artificiality of AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        exposure = params.get("exposure", 1.04)
        temperature = params.get("temperature", 4.0)
        tint = params.get("tint", 0.0)

        # Convert image to numpy array
        img_array = np.array(image.convert("RGB"), dtype=np.float32)

        # Adjust exposure
        img_array = np.clip(img_array * exposure, 0, 255)

        # Adjust color temperature
        temp_adjustment = np.array([temperature, 0, -temperature])
        img_array += temp_adjustment
        img_array = np.clip(img_array, 0, 255)

        # Adjust tint
        tint_adjustment = np.array([tint, tint, 0])
        img_array += tint_adjustment
        img_array = np.clip(img_array, 0, 255)

        # Convert back to PIL Image
        adjusted_image = Image.fromarray(img_array.astype(np.uint8), "RGB")
        return adjusted_image

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "exposure": {
                "type": "float",
                "min": 0.0,
                "max": 3.0,
                "description": "Adjusts the exposure level of the image.",
                "default": 1.0,
                "required": False
            },
            "temperature": {
                "type": "float",
                "min": -100.0,
                "max": 100.0,
                "description": "Adjusts the color temperature; negative values cool the image, positive values warm it.",
                "default": 0.0,
                "required": False
            },
            "tint": {
                "type": "float",
                "min": -100.0,
                "max": 100.0,
                "description": "Adjusts the tint of the image; negative values shift towards green, positive values towards magenta.",
                "default": 0.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Exposure: 0.0-3.0, Temperature: -100.0 to 100.0, Tint: -100.0 to 100.0; useful for fine-tuning AI-generated images."