from dataclasses import dataclass
from enum import Enum
from typing import List, Optional, Union

import numpy as np
import PIL

from ...utils import BaseOutput, is_torch_available, is_transformers_available


@dataclass
class SpatiallyComposableStableDiffusionPipelineOutput(BaseOutput):
    """
    Output class for Stable Diffusion pipelines.

    Args:
        images (`List[PIL.Image.Image]` or `np.ndarray`)
            List of denoised PIL images of length `batch_size` or numpy array of shape `(batch_size, height, width,
            num_channels)`. PIL images or numpy array present the denoised images of the diffusion pipeline.
        nsfw_content_detected (`List[bool]`)
            List of flags denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, or `None` if safety checking could not be performed.
    """

    images: Union[List[PIL.Image.Image], np.ndarray]
    nsfw_content_detected: Optional[List[bool]]


if is_transformers_available() and is_torch_available():
    from .pipeline_spatial_composable_stable_diffusion import SpatiallyComposableStableDiffusionPipeline
