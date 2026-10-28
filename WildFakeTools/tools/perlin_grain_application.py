from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class PerlinGrainApplicationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="perlin_grain_application",
            description="Applies realistic grain to images using Perlin noise, mimicking the structure of real film grain."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        grain_intensity = params.get("grain_intensity", 0.5)
        scale = params.get("scale", 5.0)

        # Convert image to numpy array
        img_array = np.array(image.convert("RGB"))

        # Generate Perlin noise
        noise = self.perlin_noise(img_array.shape[0], img_array.shape[1], scale)
        
        # Normalize noise to [0, 1] and scale by grain intensity
        noise = (noise - np.min(noise)) / (np.max(noise) - np.min(noise))
        noise = noise * grain_intensity

        # Add noise to the image
        noisy_image = img_array / 255.0 + noise[:, :, np.newaxis]
        noisy_image = np.clip(noisy_image, 0, 1) * 255

        # Convert back to PIL Image
        return Image.fromarray(noisy_image.astype(np.uint8), mode="RGB")

    def perlin_noise(self, height: int, width: int, scale: float) -> np.ndarray:
        # Create a grid of points
        x = np.linspace(0, width, width)
        y = np.linspace(0, height, height)
        x, y = np.meshgrid(x, y)

        # Generate Perlin noise
        noise = np.zeros((height, width))
        for i in range(0, height, int(scale)):
            for j in range(0, width, int(scale)):
                noise[i:i+int(scale), j:j+int(scale)] = np.random.rand()

        # Smooth the noise with a Gaussian filter
        noise = gaussian_filter(noise, sigma=scale/2)
        return noise

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "grain_intensity": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Intensity of the grain effect.",
                "default": 0.5,
                "required": False
            },
            "scale": {
                "type": "float",
                "min": 1.0,
                "max": 10.0,
                "description": "Scale of the grain pattern.",
                "default": 5.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Grain intensity: 0.0 to 1.0, Scale: 1.0 to 10.0. Use to add realistic grain to AI-generated images."