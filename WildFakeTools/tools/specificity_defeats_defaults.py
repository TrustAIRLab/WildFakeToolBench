from typing import Dict, Any
from PIL import Image, ImageFilter
import numpy as np
from .base_tool import BaseTool


class SpecificityDefeatsDefaultsTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="specificity_defeats_defaults",
            description="This tool enhances the realism of an image by applying specific imperfections, such as noise or blurriness, to avoid idealized representations.",
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        imperfection_types = params.get("imperfection_types", ["noise", "blur"])
        detail_level = params.get("detail_level", 5)

        # Convert image to RGB if not already
        image = image.convert("RGB")
        image_array = np.array(image)

        # Apply specified imperfections
        if "noise" in imperfection_types:
            noise = np.random.normal(0, detail_level * 10, image_array.shape).astype(np.uint8)
            image_array = np.clip(image_array + noise, 0, 255)

        if "blur" in imperfection_types:
            blur_radius = detail_level / 10.0
            image = Image.fromarray(image_array).filter(ImageFilter.GaussianBlur(blur_radius))
            image_array = np.array(image)

        return Image.fromarray(np.clip(image_array, 0, 255).astype(np.uint8))

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "imperfection_types": {
                "type": "list",
                "description": "Types of imperfections to apply (e.g., noise, blur).",
                "default": ["noise", "blur"],
                "required": False
            },
            "detail_level": {
                "type": "int",
                "min": 1,
                "max": 10,
                "description": "Level of detail for imperfections, where higher values indicate more pronounced effects.",
                "default": 5,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Detail level ranges from 1 to 10; useful for generating more realistic images by adding imperfections."