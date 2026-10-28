"""
Qwen-Image-Edit-2511 Tool for intelligent image editing.

Uses Qwen's dedicated image editing model (QwenImageEditPlusPipeline,
diffusion-based) for high-quality, targeted edits.

Model: Qwen/Qwen-Image-Edit-2511
Pipeline: QwenImageEditPlusPipeline (diffusers >= 0.37)
"""

import random
from typing import Dict, Any
from PIL import Image

from .base_tool import BaseTool


class IterativeDenoisePassTool(BaseTool):
    """
    Qwen-Image-Edit-2511 based intelligent image editing tool.

    Best for: anatomical fixes, texture refinement, lighting corrections.
    Use sparingly — each pass degrades image quality.
    """

    def __init__(self, model_path: str = "models/Qwen-Image-Edit-2511"):
        super().__init__(
            name="iterative_denoise_pass",
            description=(
                "Intelligent image editing using Qwen-Image-Edit-2511. "
                "Fixes STRUCTURAL issues: distorted faces, wrong proportions, "
                "unnatural textures, lighting inconsistencies. "
                "Each pass slightly degrades image quality. "
                "Provide SPECIFIC instructions from detector feedback."
            )
        )
        self.model_path = model_path
        self.pipeline = None
        self.initialized = False

    def _load_model(self):
        """Lazy load Qwen-Image-Edit-2511 pipeline."""
        if self.initialized:
            return

        from ._diffusion import load_pipeline

        self.pipeline = load_pipeline(self.name, self.model_path)
        self.pipeline.set_progress_bar_config(disable=None)
        self.initialized = True

    def unload(self):
        """Unload model from GPU to free VRAM."""
        if not self.initialized:
            return
        import gc
        self.pipeline.to("cpu")
        del self.pipeline
        self.pipeline = None
        self.initialized = False
        gc.collect()
        import torch

        torch.cuda.empty_cache()
        print("Qwen-Image-Edit-2511 unloaded from GPU")

    def get_guidance(self) -> str:
        return """**cfg_scale guidance for iterative_denoise_pass:**
- true_cfg_scale controls how aggressively the edit changes the image
- TEXTURE/COLOR fixes (smooth skin, lighting, color balance) → LOW cfg (2.0-3.0)
- STRUCTURAL/GEOMETRIC fixes (resize object, fix proportions, reshape) → HIGH cfg (3.5-5.0)
- If detector feedback is UNCHANGED after an edit, consider why:
  * cfg was too low (the edit was too subtle to matter)
  * the instruction targeted the wrong issue (detector cares about something else)
  * iterative_denoise_pass cannot fix this type of issue (try surface tools instead)
  * higher cfg risks introducing new artifacts — weigh the trade-off
- If detector feedback CHANGED, your edit had an effect — consider keeping similar cfg or adjusting based on the new feedback
- Instruction must be SPECIFIC: "smooth the pixelated nostril area" not "fix the face"
- Each pass slightly degrades quality — balance number of uses vs improvement"""

    def get_param_schema(self) -> Dict[str, Any]:
        return {
            "edit_instruction": {
                "type": "str",
                "description": (
                    "SPECIFIC, ACTIONABLE editing instruction. "
                    "BAD: 'make it look realistic'. "
                    "GOOD: 'reduce the shark mouth size to natural proportions', "
                    "'smooth the pixelated nostril area and add natural skin pores'. "
                    "Extract exact issues from detector feedback!"
                ),
                "default": (
                    "restore natural skin and surface texture: add fine pores and "
                    "subtle irregularities, remove the oversmoothed plastic sheen"
                ),
                "required": False
            },
            "negative_prompt": {
                "type": "str",
                "description": (
                    "What to AVOID in the edit. Helps prevent artifacts. "
                    "Example: 'blurry, distorted, noisy, color artifacts, low quality'. "
                    "Use when you want to prevent the edit from introducing problems."
                ),
                "default": "blurry, low quality, distorted, artifacts, noise, AI-generated, synthetic, digital art, unrealistic",
                "required": False
            },
            "cfg_scale": {
                "type": "float",
                "description": (
                    "How strongly to follow the instruction (true_cfg_scale). "
                    "LOWER = more subtle, preserves more of original. "
                    "2.0-3.0: subtle refinements (recommended for most cases), "
                    "3.0-4.0: moderate edits, "
                    "4.0-5.0: significant changes (use with caution). "
                    "Start at 3.0 — only increase if edit is too subtle."
                ),
                "default": 4.0,
                "required": False,
                "min": 1.0,
                "max": 8.0
            },
            "num_steps": {
                "type": "int",
                "description": (
                    "Denoising steps. More = higher quality but slower. "
                    "20: fast, 28: balanced, 40: best quality."
                ),
                "default": 28,
                "required": False,
                "min": 20,
                "max": 50
            },
        }

    def apply(self, image: Image.Image, params: Dict[str, Any]) -> Image.Image:
        """Apply Qwen-Image-Edit-2511 to the image."""
        self.validate_params(params)
        self._load_model()

        original_size = image.size
        original_mode = image.mode

        edit_instruction = params.get(
            "edit_instruction",
            "restore natural skin and surface texture: add fine pores and "
            "subtle irregularities, remove the oversmoothed plastic sheen",
        )
        negative_prompt = params.get("negative_prompt", "blurry, low quality, distorted, artifacts, noise, AI-generated, synthetic, digital art, unrealistic")
        cfg_scale = params.get("cfg_scale", 4.0)
        num_steps = params.get("num_steps", 28)

        print(f"\nQwen-Image-Edit-2511:")
        print(f"  Original size: {original_size}")
        print(f"  Instruction: {edit_instruction}")
        print(f"  Negative: {negative_prompt}")
        print(f"  true_cfg_scale: {cfg_scale}")
        print(f"  num_steps: {num_steps}")

        try:
            if image.mode != "RGB":
                image = image.convert("RGB")

            # Draw a fresh seed for each call.
            seed = random.randint(0, 2**31)

            import torch

            with torch.inference_mode():
                output = self.pipeline(
                    image=image,
                    prompt=edit_instruction,
                    negative_prompt=negative_prompt,
                    true_cfg_scale=cfg_scale,
                    guidance_scale=1.0,
                    num_inference_steps=num_steps,
                    num_images_per_prompt=1,
                    generator=torch.manual_seed(seed),
                )

            edited_image = output.images[0]

            if edited_image.size != original_size:
                edited_image = edited_image.resize(original_size, Image.Resampling.LANCZOS)

            if edited_image.mode != original_mode:
                edited_image = edited_image.convert(original_mode)

            print("Qwen-Image-Edit-2511 completed")
            return edited_image

        except Exception as e:
            print(f"Qwen-Image-Edit-2511 failed: {e}")
            print("  Returning original image unchanged")
            return image
