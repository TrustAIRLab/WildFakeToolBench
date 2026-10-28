from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class CameraLightingSpecificityTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="camera_lighting_specificity",
            description="Enhances the realism of AI-generated images by applying specific photographic characteristics such as camera type, lighting conditions, and subject details."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        
        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Simulate the effect of specific prompting by adjusting contrast and brightness
        camera_type = params.get("camera_type", "DSLR")
        lighting_conditions = params.get("lighting_conditions", "natural")
        subject_details = params.get("subject_details", "portrait")

        # Adjustments based on parameters (these are arbitrary for the sake of this example)
        if camera_type == "DSLR":
            img_array = np.clip(img_array * 1.1, 0, 255)  # Increase brightness slightly
        elif camera_type == "smartphone":
            img_array = np.clip(img_array * 1.05, 0, 255)  # Less increase for smartphone

        if lighting_conditions == "natural":
            img_array = np.clip(img_array + 10, 0, 255)  # Simulate natural light
        elif lighting_conditions == "artificial":
            img_array = np.clip(img_array - 10, 0, 255)  # Simulate artificial light

        # Convert back to PIL Image
        return Image.fromarray(img_array.astype('uint8'), 'RGB')

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "camera_type": {
                "type": "str",
                "description": "Type of camera used (e.g., DSLR, smartphone).",
                "default": "DSLR",
                "required": False
            },
            "lighting_conditions": {
                "type": "str",
                "description": "Lighting conditions (e.g., natural, artificial).",
                "default": "natural",
                "required": False
            },
            "subject_details": {
                "type": "str",
                "description": "Details about the subject (e.g., portrait, landscape).",
                "default": "portrait",
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Parameters: camera_type (DSLR/smartphone), lighting_conditions (natural/artificial), subject_details (portrait/landscape). Use to enhance realism in AI-generated images."