from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SkinDetailEmphasisTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="skin_detail_emphasis",
            description="Enhances the realism of an image by applying weighted adjustments to specific features like skin texture and sharpness of eyes."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        feature_weights = params.get("feature_weights", [1.0, 1.0])  # Default weights for skin texture and eye sharpness
        
        # Convert image to RGB if not already
        image = image.convert("RGB")
        img_array = np.array(image)

        # Apply weighted adjustments
        if len(feature_weights) >= 2:
            # Emphasize skin texture (using Gaussian blur)
            skin_weight = feature_weights[0]
            img_array = Image.fromarray(img_array).filter(ImageFilter.GaussianBlur(radius=1 * skin_weight))
            img_array = np.array(img_array)

            # Emphasize eye sharpness (using sharpening filter)
            eye_weight = feature_weights[1]
            if eye_weight > 0:
                kernel = np.array([[0, -1, 0],
                                   [-1, 5 + eye_weight, -1],
                                   [0, -1, 0]])
                img_array = self.convolve2d(img_array, kernel)

        return Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))

    def convolve2d(self, image: np.ndarray, kernel: np.ndarray) -> np.ndarray:
        from scipy.signal import convolve2d
        channels = []
        for i in range(image.shape[2]):
            channels.append(convolve2d(image[:, :, i], kernel, mode='same', boundary='wrap'))
        return np.stack(channels, axis=-1)

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "feature_weights": {
                "type": "list",
                "description": "Weights for specific features to enhance (e.g., [skin_texture_weight, eye_sharpness_weight]).",
                "default": [1.0, 1.0],
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust feature_weights to emphasize skin texture and eye sharpness; default is [1.0, 1.0]."