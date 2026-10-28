from typing import Dict, Any
from PIL import Image
import numpy as np
from .base_tool import BaseTool


class FilmStockEmulationTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="film_stock_emulation",
            description="Applies authentic film stock emulations to simulate specific color rendering, altering the color palette and rendering to give a more organic feel to images."
        )

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        self.validate_params(params)
        film_stock = params.get("film_stock", "Kodak_Ektar")
        intensity = params.get("intensity", 50) / 100.0
        
        # Convert image to numpy array
        img_array = np.array(image.convert("RGB"), dtype=np.float32)

        # Define film stock emulation presets
        film_presets = {
            "Kodak_Ektar": np.array([1.1, 1.05, 0.9]),  # Boost reds, greens, lower blues
            "Fujifilm_Provia": np.array([1.0, 1.1, 1.0]),  # Boost greens
            "Kodak_Tri-X": np.array([0.8, 0.8, 0.8]),  # Desaturate all colors
            "Ilford_HP5": np.array([0.9, 0.9, 0.9]),  # Slightly desaturate
        }

        # Get the selected film stock preset
        film_effect = film_presets.get(film_stock, film_presets["Kodak_Ektar"])

        # Apply the film stock effect
        emulated_image = img_array * film_effect * intensity + img_array * (1 - intensity)
        emulated_image = np.clip(emulated_image, 0, 255).astype(np.uint8)

        # Convert back to PIL Image
        return Image.fromarray(emulated_image, mode="RGB")

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "film_stock": {
                "type": "str",
                "description": "The film stock to emulate.",
                "default": "Kodak_Ektar",
                "required": False
            },
            "intensity": {
                "type": "float",
                "min": 0.0,
                "max": 100.0,
                "description": "The intensity of the film stock effect (0-100%).",
                "default": 50.0,
                "required": False
            }
        }

    def get_guidance(self) -> str:
        return "Intensity ranges from 0 to 100%, with various film stocks available for emulation."