from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class ExaggeratedFeaturesIntegrationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="exaggerated_features_integration",
            description="Integrates exaggerated facial features into objects to enhance realism. This process adds detail that can distract from the artificiality of AI-generated images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        feature_type = params.get("feature_type", "eyes")
        intensity = params.get("intensity", 0.5)

        # Convert image to RGB if needed
        image = image.convert("RGB")
        img_array = np.array(image)

        # Create a blurred version of the image to simulate texture
        blurred_image = image.filter(ImageFilter.GaussianBlur(radius=5))
        blurred_array = np.array(blurred_image)

        # Exaggerate features based on the specified type
        if feature_type == "eyes":
            # Simulate exaggerated eyes by enhancing brightness and contrast in a specific area
            eye_mask = np.zeros_like(img_array)
            eye_mask[100:200, 100:300] = [255, 255, 255]  # Dummy eye region
            eye_mask = np.clip(eye_mask * intensity, 0, 255).astype(np.uint8)
            img_array = np.where(eye_mask > 0, img_array + eye_mask, img_array)
        elif feature_type == "mouth":
            # Simulate exaggerated mouth by enhancing brightness and contrast in a specific area
            mouth_mask = np.zeros_like(img_array)
            mouth_mask[300:400, 100:300] = [255, 255, 255]  # Dummy mouth region
            mouth_mask = np.clip(mouth_mask * intensity, 0, 255).astype(np.uint8)
            img_array = np.where(mouth_mask > 0, img_array + mouth_mask, img_array)

        # Blend the original image with the blurred version to enhance realism
        blended_array = (img_array * (1 - intensity) + blurred_array * intensity).astype(np.uint8)

        return Image.fromarray(blended_array, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "feature_type": {
                "type": "str",
                "description": "Type of facial feature to exaggerate (e.g., 'eyes', 'mouth').",
                "default": "eyes",
                "required": False
            },
            "intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the exaggeration effect.",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity ranges from 0.0 (no effect) to 1.0 (full effect); use this tool to enhance realism in AI-generated images."