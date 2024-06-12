import inspect
from itertools import repeat
from typing import Callable, List, Optional, Union

import torch
# from transformers import CLIPFeatureExtractor, CLIPTextModel, CLIPTokenizer
import torchvision

# from ...models import AutoencoderKL, UNet2DConditionModel
# from ...pipeline_utils import DiffusionPipeline
# from ...pipelines.stable_diffusion.safety_checker import StableDiffusionSafetyChecker
# from ...schedulers import KarrasDiffusionSchedulers
from ...utils import logging, randn_tensor
from . import SpatiallyComposableStableDiffusionPipelineOutput
from ..stable_diffusion.pipeline_stable_diffusion import StableDiffusionPipeline


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name
EXAMPLE_DOC_STRING = """
    Examples:
        ```py
        >>> import torch
        >>> from diffusers import SemanticStableDiffusionPipeline

        >>> pipe = SemanticStableDiffusionPipeline.from_pretrained(
        ...     "runwayml/stable-diffusion-v1-5", torch_dtype=torch.float16
        ... )
        >>> pipe = pipe.to("cuda")

        >>> out = pipe(
        ...     prompt="a photo of the face of a woman",
        ...     num_images_per_prompt=1,
        ...     guidance_scale=7,
        ...     editing_prompt=[
        ...         "smiling, smile",  # Concepts to apply
        ...         "glasses, wearing glasses",
        ...         "curls, wavy hair, curly hair",
        ...         "beard, full beard, mustache",
        ...     ],
        ...     reverse_editing_direction=[
        ...         False,
        ...         False,
        ...         False,
        ...         False,
        ...     ],  # Direction of guidance i.e. increase all concepts
        ...     edit_warmup_steps=[10, 10, 10, 10],  # Warmup period for each concept
        ...     edit_guidance_scale=[4, 5, 5, 5.4],  # Guidance scale for each concept
        ...     edit_threshold=[
        ...         0.99,
        ...         0.975,
        ...         0.925,
        ...         0.96,
        ...     ],  # Threshold for each concept. Threshold equals the percentile of the latent space that will be discarded. I.e. threshold=0.99 uses 1% of the latent dimensions
        ...     edit_momentum_scale=0.3,  # Momentum scale that will be added to the latent guidance
        ...     edit_mom_beta=0.6,  # Momentum beta
        ...     edit_weights=[1, 1, 1, 1, 1],  # Weights of the individual concepts against each other
        ... )
        >>> image = out.images[0]
        ```
"""

class SpatialFilter:
    def __init__(self, pos_str: str, mix_val: float):
        pos_split = pos_str.split("-")
        pos_div = pos_split[0].split(":")
        pos_target = pos_split[1].split(":")
        self.y_divisions = int(pos_div[0])
        self.x_divisions = int(pos_div[1])
        self.y_target = int(pos_target[0])
        self.x_target = int(pos_target[1])

        assert self.y_target < self.y_divisions
        assert self.x_target < self.x_divisions

        self.weight = mix_val

    def create_tensor(self, batch_size: int, num_channels: int, height: int, width: int, device: str) -> torch.Tensor:

        tensor = torch.zeros(batch_size, num_channels, height, width).to(device).to(torch.float16)

        y_start = int(self.y_target / self.y_divisions * height)
        y_end = int((self.y_target+1) / self.y_divisions * height)
        x_start = int(self.x_target / self.x_divisions * width)
        x_end = int((self.x_target+1) / self.x_divisions * width)

        tensor[:, :, y_start:y_end, x_start:x_end] = self.weight

        return tensor

class SpatiallyComposableStableDiffusionPipeline(StableDiffusionPipeline):
    r"""
    Pipeline for text-to-image generation with spatial composition.

    This model inherits from [`StableDiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode images to and from latent representations.
        text_encoder ([`CLIPTextModel`]):
            Frozen text-encoder. Stable Diffusion uses the text portion of
            [CLIP](https://huggingface.co/docs/transformers/model_doc/clip#transformers.CLIPTextModel), specifically
            the [clip-vit-large-patch14](https://huggingface.co/openai/clip-vit-large-patch14) variant.
        tokenizer (`CLIPTokenizer`):
            Tokenizer of class
            [CLIPTokenizer](https://huggingface.co/docs/transformers/v4.21.0/en/model_doc/clip#transformers.CLIPTokenizer).
        unet ([`UNet2DConditionModel`]): Conditional U-Net architecture to denoise the encoded image latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `unet` to denoise the encoded image latens. Can be one of
            [`DDIMScheduler`], [`LMSDiscreteScheduler`], or [`PNDMScheduler`].
        safety_checker ([`Q16SafetyChecker`]):
            Classification module that estimates whether generated images could be considered offensive or harmful.
            Please, refer to the [model card](https://huggingface.co/CompVis/stable-diffusion-v1-4) for details.
        feature_extractor ([`CLIPFeatureExtractor`]):
            Model that extracts features from generated images to be used as inputs for the `safety_checker`.
    
    Position strings should be structured as follows: "<num_height_splits>:<num_width_splits>-<height_split_index>:<width_split_index>".
    A string "1:2-0:1" splits the image into a left side and a right side and targets the right side.
    """

    @torch.no_grad()
    def __call__(
        self,
        prompt: Union[List[str], List[List[str]]],
        pos: List[str],
        mix_val: Union[float, List[float]],
        mask_guidance_bootstrapping_ratio: float = 0.2,
        height: Optional[int] = None,
        width: Optional[int] = None,
        num_inference_steps: int = 50,
        guidance_scale: float = 7.5,
        negative_prompt: Optional[Union[str, List[str]]] = None,
        num_images_per_prompt: Optional[int] = 1,
        eta: float = 0.0,
        generator: Optional[torch.Generator] = None,
        latents: Optional[torch.FloatTensor] = None,
        output_type: Optional[str] = "pil",
        return_dict: bool = True,
        callback: Optional[Callable[[int, int, torch.FloatTensor], None]] = None,
        callback_steps: Optional[int] = 1,
    ):
        r"""
        Function invoked when calling the pipeline for generation.

        Args:
            prompt (`str` or `List[str]`):
                The prompt or prompts to guide the image generation.
            height (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The height in pixels of the generated image.
            width (`int`, *optional*, defaults to self.unet.config.sample_size * self.vae_scale_factor):
                The width in pixels of the generated image.
            num_inference_steps (`int`, *optional*, defaults to 50):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, *optional*, defaults to 7.5):
                Guidance scale as defined in [Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598).
                `guidance_scale` is defined as `w` of equation 2. of [Imagen
                Paper](https://arxiv.org/pdf/2205.11487.pdf). Guidance scale is enabled by setting `guidance_scale >
                1`. Higher guidance scale encourages to generate images that are closely linked to the text `prompt`,
                usually at the expense of lower image quality.
            negative_prompt (`str` or `List[str]`, *optional*):
                The prompt or prompts not to guide the image generation. Ignored when not using guidance (i.e., ignored
                if `guidance_scale` is less than `1`).
            num_images_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            eta (`float`, *optional*, defaults to 0.0):
                Corresponds to parameter eta (η) in the DDIM paper: https://arxiv.org/abs/2010.02502. Only applies to
                [`schedulers.DDIMScheduler`], will be ignored for others.
            generator (`torch.Generator`, *optional*):
                A [torch generator](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make generation
                deterministic.
            latents (`torch.FloatTensor`, *optional*):
                Pre-generated noisy latents, sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor will ge generated by sampling using the supplied random `generator`.
            output_type (`str`, *optional*, defaults to `"pil"`):
                The output format of the generate image. Choose between
                [PIL](https://pillow.readthedocs.io/en/stable/): `PIL.Image.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] instead of a
                plain tuple.
            callback (`Callable`, *optional*):
                A function that will be called every `callback_steps` steps during inference. The function will be
                called with the following arguments: `callback(step: int, timestep: int, latents: torch.FloatTensor)`.
            callback_steps (`int`, *optional*, defaults to 1):
                The frequency at which the `callback` function will be called. If not specified, the callback will be
                called at every step.

        Returns:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] or `tuple`:
            [`~pipelines.stable_diffusion.StableDiffusionPipelineOutput`] if `return_dict` is True, otherwise a `tuple.
            When returning a tuple, the first element is a list with the generated images, and the second element is a
            list of `bool`s denoting whether the corresponding generated image likely represents "not-safe-for-work"
            (nsfw) content, according to the `safety_checker`.
        """
        # Generated image size must be divisible by 8
        height = height - height % 8
        width = width - width % 8

        # 0. Default height and width to unet
        height = height or self.unet.config.sample_size * self.vae_scale_factor
        width = width or self.unet.config.sample_size * self.vae_scale_factor

        # 1. Check inputs. Raise error if not correct
        self.check_inputs(prompt[0], height, width, callback_steps)

        # 2. Define call parameters
        batch_size = 1 if isinstance(prompt[0], str) else len(prompt[0])
        device = self._execution_device
        # here `guidance_scale` is defined analog to the guidance weight `w` of equation (2)
        # of the Imagen paper: https://arxiv.org/pdf/2205.11487.pdf . `guidance_scale = 1`
        # corresponds to doing no classifier free guidance.
        do_classifier_free_guidance = guidance_scale > 1.0

        # 3. Encode input prompt
        text_embeddings = []
        for i in range(len(prompt)):
          one_text_embeddings = self._encode_prompt(
              prompt[i], device, num_images_per_prompt, do_classifier_free_guidance, negative_prompt[i]
          )
          text_embeddings.append(one_text_embeddings)

        # 4. Prepare timesteps
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables
        num_channels_latents = self.unet.in_channels
        latents = self.prepare_latents(
            batch_size * num_images_per_prompt,
            num_channels_latents,
            height,
            width,
            text_embeddings[0].dtype,
            device,
            generator,
            latents,
        )

        # 6. Prepare extra step kwargs. TODO: Logic should ideally just be moved out of the pipeline
        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        composition_filters = [SpatialFilter(p, m).create_tensor(batch_size, 4, height // 8, width // 8, device) for p, m in zip(pos, mix_val)]

        # 7. Denoising loop
        #num_warmup_steps = len(timesteps) - num_inference_steps# * self.scheduler.order
        num_mask_guidance_bs_steps = int(mask_guidance_bootstrapping_ratio * num_inference_steps)
        for j, t in enumerate(self.progress_bar(timesteps)):
            # expand the latents if we are doing classifier free guidance
            latent_model_input = torch.cat([latents] * 2) if do_classifier_free_guidance else latents
            latent_model_input = self.scheduler.scale_model_input(latent_model_input, t)

            # predict the noise residual
            noise_preds = []
            for i in range(len(prompt)):
                noise_preds.append(self.unet(latent_model_input, t, encoder_hidden_states=text_embeddings[i]).sample)
            # perform guidance
            if do_classifier_free_guidance:
                # init with unconditional noise
                noise_pred_agg = noise_preds[0].chunk(2)[0]

                # only attend to masks during bootstrapping
                if j < num_mask_guidance_bs_steps:
                    # assume the first prompt is the 'background' and skip it during bootstrapping
                    for i in range(1, len(prompt)):
                        noise_pred_uncond, noise_pred_cond = noise_preds[i].chunk(2)
                        noise_pred_agg += guidance_scale * (noise_pred_cond - noise_pred_uncond) * composition_filters[i]
                else:
                    # create aggregate noise prediction for each prompt with combination of
                    # unconditional noise predication and scaled text-conditioned noise prediction
                    # apply masks
                    for i in range(len(prompt)):
                        noise_pred_uncond, noise_pred_cond = noise_preds[i].chunk(2)
                        noise_pred_agg += guidance_scale * (noise_pred_cond - noise_pred_uncond) * composition_filters[i]

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred_agg, t, latents, **extra_step_kwargs).prev_sample
        
                # call the callback, if provided
                if callback is not None and j % callback_steps == 0:
                    callback(j, t, latents)

        # 8. Post-processing
        image = self.decode_latents(latents)

        # 9. Run safety checker
        image, has_nsfw_concept = self.run_safety_checker(image, device, text_embeddings[0].dtype)
        
        # 10. Convert to PIL
        if output_type == "pil":
            image = self.numpy_to_pil(image)
            output = []

        if not return_dict:
            return (image, has_nsfw_concept)

        return SpatiallyComposableStableDiffusionPipelineOutput(images=image, nsfw_content_detected=has_nsfw_concept)
