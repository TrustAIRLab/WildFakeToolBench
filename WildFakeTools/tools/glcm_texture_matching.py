from typing import Dict, Any
from PIL import Image
import numpy as np
from scipy.ndimage import gaussian_filter
from .base_tool import BaseTool


class GlcmTextureMatchingTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="glcm_texture_matching",
            description="This tool enhances the texture of an AI-generated image to resemble that of a real photograph by applying Gray-Level Co-occurrence Matrix (GLCM) features."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        matching_strength = params.get("matching_strength", 0.5)
        
        # Convert image to grayscale for GLCM calculation
        gray_image = image.convert("L")
        gray_array = np.array(gray_image)

        # Compute GLCM of the input image
        glcm = self.compute_glcm(gray_array)
        texture_features = self.extract_glcm_features(glcm)

        # Create a reference texture based on the input image
        reference_texture = self.create_reference_texture(gray_array)

        # Blend the textures based on matching strength
        blended_texture = (1 - matching_strength) * gray_array + matching_strength * reference_texture

        # Convert back to an RGB image
        result_image = Image.fromarray(np.clip(blended_texture, 0, 255).astype(np.uint8), mode="L").convert("RGB")
        return result_image

    def compute_glcm(self, gray_array: np.ndarray, distances=[1], angles=[0]):
        # Compute the GLCM for a single distance and angle.
        glcm = np.zeros((256, 256), dtype=np.float32)
        for i in range(gray_array.shape[0] - 1):
            for j in range(gray_array.shape[1] - 1):
                pixel_value = gray_array[i, j]
                neighbor_value = gray_array[i + 1, j]  # Simple horizontal adjacency
                glcm[pixel_value, neighbor_value] += 1
        return glcm / np.sum(glcm)

    def extract_glcm_features(self, glcm: np.ndarray):
        # Extract features like contrast, dissimilarity, homogeneity, etc.
        contrast = np.sum((np.arange(256)[:, None] - np.arange(256)) ** 2 * glcm)
        dissimilarity = np.sum(np.abs(np.arange(256)[:, None] - np.arange(256)) * glcm)
        return contrast, dissimilarity

    def create_reference_texture(self, gray_array: np.ndarray):
        # Create a reference texture by applying Gaussian blur
        blurred_texture = gaussian_filter(gray_array, sigma=2)
        return blurred_texture

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "matching_strength": {
                "type": "float",
                "min": 0.0,
                "max": 1.0,
                "description": "Strength of the texture matching (0.0 to 1.0).",
                "default": 0.5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Adjust matching_strength (0.0 to 1.0) to control texture blending; useful for enhancing AI-generated images."