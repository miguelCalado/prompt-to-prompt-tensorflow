"""TensorFlow/Keras implementation of Stable Diffusion and Prompt-to-Prompt papers.

References
----------
- "High-Resolution Image Synthesis With Latent Diffusion Models"
  Rombach, Robin and Blattmann, Andreas and Lorenz, Dominik and Esser, Patrick and Ommer, Bjorn
  Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)
  https://arxiv.org/abs/2112.10752
- "Prompt-to-Prompt Image Editing with Cross-Attention Control."
  Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or.
  https://arxiv.org/abs/2208.01626

Credits
----------
- [keras-cv](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/generative/stable_diffusion) \
  for the TensorFlow/Keras implementation of Stable Diffusion.
- [bloc97/CrossAttentionControl](https://github.com/bloc97/CrossAttentionControl) unofficial implementation of \
  the paper, where the method `get_matching_sentence_tokens` and code logic were used.
- [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt) official implementation of the paper in PyTorch.
"""

import math
from typing import List, Optional, Tuple, Union

import numpy as np
import tensorflow as tf
from keras_cv.models.stable_diffusion.clip_tokenizer import SimpleTokenizer
from keras_cv.models.stable_diffusion.constants import (
    _ALPHAS_CUMPROD,
    _UNCONDITIONAL_TOKENS,
)
from keras_cv.models.stable_diffusion.decoder import Decoder
from keras_cv.models.stable_diffusion.diffusion_model import (
    DiffusionModel,
    DiffusionModelV2,
)
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder, TextEncoderV2
from tensorflow import keras

import ptp_utils

MAX_PROMPT_LENGTH = 77
NUM_TRAIN_TIMESTEPS = 1000


class StableDiffusionBase:
    """Implementation of Stable Diffusion and Prompt-to-Prompt papers in TensorFlow/Keras.

    Parameters
    ----------
    strategy : tf.distribute
        TensorFlow strategy for running computations across multiple devices.
    img_height : int, optional
        Image height, by default 512
    img_width : int, optional
        Image width, by default 512
    jit_compile : bool, optional
        Flag to compile the models to XLA, by default False.
    download_weights : bool, optional
        Flag to download the models weights, by default True.

    Examples
    --------
    >>> import tensorflow as tf
    >>> from PIL import Image
    >>> from stable_diffusion import StableDiffusion
    >>> strategy = tf.distribute.get_strategy() # To use only one GPU
    >>> generator = StableDiffusion(
            strategy=strategy,
            img_height=512,
            img_width=512,
            jit_compile=False,
        )
    >>> img = generator.text_to_image(
            prompt="teddy bear with sunglasses relaxing in a pool",
            num_steps=50,
            unconditional_guidance_scale=8,
            seed=3345435,
            batch_size=1,
        )
    >>> Image.fromarray(img[0]).save("original_prompt.png")

    Now lets edit the image to customize the teddy bear's sunglasses

    >>> img = generator.text_to_image_ptp(
            prompt="teddy bear with sunglasses relaxing in a pool",
            prompt_edit="teddy bear with heart-shaped red colored sunglasses relaxing in a pool",
            num_steps=50,
            unconditional_guidance_scale=8,
            cross_attn2_replace_steps_start=0.0,
            cross_attn2_replace_steps_end=1.0,
            cross_attn1_replace_steps_start=1.0,
            cross_attn1_replace_steps_end=1.0,
            seed=3345435,
            batch_size=1,from keras_cv.models.stable_diffusion.decoder import Decoder
        )
    >>> Image.fromarray(img[0]).save("edited_prompt.png")
    """

    def __init__(
        self,
        img_height: int = 512,
        img_width: int = 512,
        jit_compile: bool = False,
    ):

        # UNet requires multiples of 2**7 = 128
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        self.img_height = img_height
        self.img_width = img_width

        # lazy initialize the component models and the tokenizer
        self._image_encoder = None
        self._text_encoder = None
        self._diffusion_model = None
        self._diffusion_model_ptp = None
        self._decoder = None
        self._tokenizer = None

        self.jit_compile = jit_compile

    def text_to_image(
        self,
        prompt: str,
        negative_prompt: Optional[str] = None,
        num_steps: int = 50,
        unconditional_guidance_scale: float = 7.5,
        batch_size: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an image based on a prompt text.

        Parameters
        ----------
        prompt : str
            Text containing the information for the model to generate.
        negative_prompt : str
            A string containing information to negatively guide the image
            generation (e.g. by removing or altering certain aspects of the
            generated image).
        num_steps : int, optional
            Number of diffusion steps (controls image quality), by default 50.
        unconditional_guidance_scale : float, optional
            Controls how closely the image should adhere to the prompt, by default 7.5.
        batch_size : int, optional
            Batch size (number of images to generate), by default 1.
        seed : Optional[int], optional
            Number to seed the random noise, by default None.

        Returns
        -------
        np.ndarray
            Generated image.
        """

        # Tokenize and encode prompt
        encoded_text = self.encode_text(prompt)

        conditional_context = self._expand_tensor(encoded_text, batch_size)

        if negative_prompt is None:
            unconditional_context = tf.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )

        # Get initial random noise
        latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Scheduler
        # TODO: Add diffusers LMSDiscreteScheduler
        timesteps = tf.range(1, 1000, 1000 // num_steps)

        # Get Initial parameters
        alphas, alphas_prev = self._get_initial_alphas(timesteps)

        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        # Diffusion stage
        for index, timestep in list(enumerate(timesteps))[::-1]:

            t_emb = self._get_timestep_embedding(timestep, batch_size)

            # Predict the unconditional noise residual
            unconditional_latent = self.diffusion_model.predict_on_batch(
                [latent, t_emb, unconditional_context]
            )

            # Predict the conditional noise residual
            conditional_latent = self.diffusion_model.predict_on_batch(
                [latent, t_emb, conditional_context]
            )

            # Perform guidance
            e_t = unconditional_latent + unconditional_guidance_scale * (
                conditional_latent - unconditional_latent
            )

            a_t, a_prev = alphas[index], alphas_prev[index]
            latent = self._get_x_prev(latent, e_t, a_t, a_prev)

            iteration += 1
            progbar.update(iteration)

        # Decode image
        img = self._get_decoding_stage(latent)

        return img

    def encode_text(self, prompt):
        """Encodes a prompt into a latent text encoding.
        The encoding produced by this method should be used as the
        `encoded_text` parameter of `StableDiffusion.generate_image`. Encoding
        text separately from generating an image can be used to arbitrarily
        modify the text encoding priot to image generation, e.g. for walking
        between two prompts.
        Args:
            prompt: a string to encode, must be 77 tokens or shorter.
        Example:
        ```python
        from keras_cv.models import StableDiffusion
        model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
        encoded_text  = model.encode_text("Tacos at dawn")
        img = model.generate_image(encoded_text)
        ```
        """
        # Tokenize prompt (i.e. starting context)
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)

        context = self.text_encoder.predict_on_batch([phrase, self._get_pos_ids()])

        return context

    def text_to_image_ptp(
        self,
        prompt: str,
        prompt_edit: str,
        method: str,
        self_attn_steps: Union[float, Tuple[float, float]],
        cross_attn_steps: Union[float, Tuple[float, float]],
        attn_edit_weights: np.ndarray = np.array([]),
        negative_prompt: Optional[str] = None,
        num_steps: int = 50,
        unconditional_guidance_scale: float = 7.5,
        batch_size: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an image based on the Prompt-to-Prompt editing method.

        Edit a generated image controlled only through text.
        Paper: https://arxiv.org/abs/2208.01626

        Parameters
        ----------
        prompt : str
            Text containing the information for the model to generate.
        prompt_edit : str
            Second prompt used to control the edit of the generated image.
        method : str
            Prompt-to-Prompt method to chose. Can be ['refine', 'replace', 'reweigh'].
        self_attn_steps : Union[float, Tuple[float, float]]

        cross_attn_steps : Union[float, Tuple[float, float]]

        attn_edit_weights: np.array([]), optional
            Set of weights for each edit prompt token.
            This is used for manipulating the importance of the edit prompt tokens,
            increasing or decreasing the importance assigned to each word.
        negative_prompt : Optional[str] = None
            A string containing information to negatively guide the image
            generation (e.g. by removing or altering certain aspects of the
            generated image).
        num_steps : int, optional
            Number of diffusion steps (controls image quality), by default 50.
        unconditional_guidance_scale : float, optional
            Controls how closely the image should adhere to the prompt, by default 7.5.
        batch_size : int, optional
            Batch size (number of images to generate), by default 1.
        seed : Optional[int], optional
            Number to seed the random noise, by default None.

        Returns
        -------
        np.ndarray
            Generated image with edited prompt method.

        Examples
        --------
        >>> import tensorflow as tf
        >>> from PIL import Image
        >>> from stable_diffusion import StableDiffusion
        >>> strategy = tf.distribute.get_strategy() # To use only one GPU
        >>> generator = StableDiffusion(
                strategy=strategy,
                img_height=512,
                img_width=512,
                jit_compile=False,
            )

        Edit the original generated image by adding heart-shaped red colored to the sunglasses.

        >>> img = generator.text_to_image_ptp(
                prompt="teddy bear with sunglasses relaxing in a pool",
                prompt_edit="teddy bear with heart-shaped red colored sunglasses relaxing in a pool",
                num_steps=50,
                unconditional_guidance_scale=8,
                self_attn_steps=0.0,
                cross_attn_steps=1.0,
                seed=3345435,
                batch_size=1,
            )
        >>> Image.fromarray(img[0]).save("edited_prompt.png")
        """

        # Tokenize and encode prompt
        encoded_text = self.encode_text(prompt)
        conditional_context = self._expand_tensor(encoded_text, batch_size)

        # Tokenize and encode edit prompt
        encoded_text_edit = self.encode_text(prompt_edit)
        conditional_context_edit = self._expand_tensor(encoded_text_edit, batch_size)

        if negative_prompt is None:
            unconditional_context = tf.repeat(
                self._get_unconditional_context(), batch_size, axis=0
            )
        else:
            unconditional_context = self.encode_text(negative_prompt)
            unconditional_context = self._expand_tensor(
                unconditional_context, batch_size
            )

        ## PTP stuff
        if isinstance(self_attn_steps, float):
            self_attn_steps = (0.0, self_attn_steps)
        if isinstance(cross_attn_steps, float):
            cross_attn_steps = (0.0, cross_attn_steps)

        if method == "refine":
            # Get the mask and indices of the difference between the original prompt token's and the edited one
            tokens_conditional = self.tokenize_prompt(prompt)
            tokens_conditional_edit = self.tokenize_prompt(prompt_edit)
            mask, indices = ptp_utils.get_matching_sentence_tokens(
                tokens_conditional[0].numpy(), tokens_conditional_edit[0].numpy()
            )

            # Add the mask and indices to the diffusion model
            self._diffusion_model_ptp = ptp_utils.put_mask_dif_model(
                self.diffusion_model_ptp, mask, indices
            )

        # Update prompt weights variable
        if attn_edit_weights.size:
            self._diffusion_model_ptp = ptp_utils.add_prompt_weights(
                diff_model=self.diffusion_model_ptp, prompt_weights=attn_edit_weights
            )

        # Get initial random noise
        latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Scheduler
        # TODO: Add diffusers LMSDiscreteScheduler
        timesteps = tf.range(1, 1000, 1000 // num_steps)

        # Get Initial parameters
        alphas, alphas_prev = self._get_initial_alphas(timesteps)

        progbar = keras.utils.Progbar(len(timesteps))
        iteration = 0
        # Diffusion stage
        for index, timestep in list(enumerate(timesteps))[::-1]:

            t_emb = self._get_timestep_embedding(timestep, batch_size)

            # Change this!
            t_scale = 1 - (timestep / NUM_TRAIN_TIMESTEPS)

            # Update Cross-Attention mode to 'unconditional'
            self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                diff_model=self.diffusion_model_ptp, mode="unconditional"
            )

            # Predict the unconditional noise residual
            unconditional_latent = self.diffusion_model_ptp.predict_on_batch(
                [latent, t_emb, unconditional_context]
            )

            # Save last cross attention activations
            self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                diff_model=self.diffusion_model_ptp, mode="save"
            )
            # Predict the conditional noise residual
            _ = self.diffusion_model_ptp.predict_on_batch(
                [latent, t_emb, conditional_context]
            )

            # TODO: # if method=='reweigh':
            # Edit the Cross-Attention layer activations
            if cross_attn_steps[0] <= t_scale <= cross_attn_steps[1]:
                if method == "replace":
                    # Use cross attention from the original prompt (M_t)
                    self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                        diff_model=self.diffusion_model_ptp,
                        mode="use_last",
                        attn_suffix="attn2",
                    )
                elif method == "refine":
                    self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                        diff_model=self.diffusion_model_ptp,
                        mode="edit",
                        attn_suffix="attn2",
                    )
            else:
                # Use cross attention from the edited prompt (M^*_t)
                self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                    diff_model=self.diffusion_model_ptp,
                    mode="injection",
                    attn_suffix="attn2",
                )

            # Edit the self-Attention layer activations
            if self_attn_steps[0] <= t_scale <= self_attn_steps[1]:
                # Use self attention from the original prompt (M_t)
                self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                    diff_model=self.diffusion_model_ptp,
                    mode="use_last",
                    attn_suffix="attn1",
                )
            else:
                # Use self attention from the edited prompt (M^*_t)
                self._diffusion_model_ptp = ptp_utils.update_cross_attn_mode(
                    diff_model=self.diffusion_model_ptp,
                    mode="injection",
                    attn_suffix="attn1",
                )

            # Predict the edited conditional noise residual
            conditional_latent_edit = self.diffusion_model_ptp.predict_on_batch(
                [latent, t_emb, conditional_context_edit],
            )

            # Perform guidance
            e_t = unconditional_latent + unconditional_guidance_scale * (
                conditional_latent_edit - unconditional_latent
            )

            a_t, a_prev = alphas[index], alphas_prev[index]
            latent = self._get_x_prev(latent, e_t, a_t, a_prev)

            iteration += 1
            progbar.update(iteration)

        # Decode image
        img = self._get_decoding_stage(latent)

        # Reset control variables
        self._diffusion_model_ptp = ptp_utils.reset_initial_tf_variables(
            self.diffusion_model_ptp
        )

        return img

    def _get_unconditional_context(self):
        unconditional_tokens = tf.convert_to_tensor(
            [_UNCONDITIONAL_TOKENS], dtype=tf.int32
        )
        unconditional_context = self.text_encoder.predict_on_batch(
            [unconditional_tokens, self._get_pos_ids()]
        )

        return unconditional_context

    def tokenize_prompt(self, prompt: str) -> tf.Tensor:
        """Tokenize a phrase prompt.

        Parameters
        ----------
        prompt : str
            The prompt string to tokenize, must be 77 tokens or shorter.
        batch_size : int
            Batch size.

        Returns
        -------
        np.ndarray
            Array of tokens.
        """
        inputs = self.tokenizer.encode(prompt)
        if len(inputs) > MAX_PROMPT_LENGTH:
            raise ValueError(
                f"Prompt is too long (should be <= {MAX_PROMPT_LENGTH} tokens)"
            )
        phrase = inputs + [49407] * (MAX_PROMPT_LENGTH - len(inputs))
        phrase = tf.convert_to_tensor([phrase], dtype=tf.int32)
        return phrase

    def create_prompt_weights(
        self, prompt: str, prompt_weights: List[Tuple[str, float]]
    ) -> np.ndarray:
        """Create an array of weights for each prompt token.

        This is used for manipulating the importance of the prompt tokens,
        increasing or decreasing the importance assigned to each word.

        Parameters
        ----------
        prompt : str
            The prompt string to tokenize, must be 77 tokens or shorter.
        prompt_weights : List[Tuple[str, float]]
            A list of tuples containing the pair of word and weight to be manipulated.
        batch_size : int
            Batch size.

        Returns
        -------
        np.ndarray
            Array of weights to control the importance of each prompt token.
        """

        # Initialize the weights to 1.
        weights = np.ones(MAX_PROMPT_LENGTH)

        # Get the prompt tokens
        tokens = self.tokenize_prompt(prompt)

        # Extract the new weights and tokens
        edit_weights = [weight for word, weight in prompt_weights]
        edit_tokens = [
            self.tokenizer.encode(word)[1:-1] for word, weight in prompt_weights
        ]

        # Get the indexes of the tokens
        index_edit_tokens = np.in1d(tokens, edit_tokens).nonzero()[0]

        # Replace the original weight values
        weights[index_edit_tokens] = edit_weights
        return weights

    def _expand_tensor(self, text_embedding, batch_size):
        """Extends a tensor by repeating it to fit the shape of the given batch size."""
        text_embedding = tf.squeeze(text_embedding)
        if text_embedding.shape.rank == 2:
            text_embedding = tf.repeat(
                tf.expand_dims(text_embedding, axis=0), batch_size, axis=0
            )
        return text_embedding

    def _get_initial_alphas(self, timesteps):

        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size: int, seed: Optional[int]):
        return tf.random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4), seed=seed
        )

    def _get_timestep_embedding(self, timestep, batch_size, dim=320, max_period=10000):
        half = dim // 2
        freqs = tf.math.exp(
            -math.log(max_period) * tf.range(0, half, dtype=tf.float32) / half
        )
        args = tf.convert_to_tensor([timestep], dtype=tf.float32) * freqs
        embedding = tf.concat([tf.math.cos(args), tf.math.sin(args)], 0)
        embedding = tf.reshape(embedding, [1, -1])
        return tf.repeat(embedding, batch_size, axis=0)

    def _get_decoding_stage(self, latent):
        decoded = self.decoder.predict_on_batch(latent)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    # e_t -> latent | x -> latent -> latent_prev
    def _get_x_prev(self, x, e_t, a_t, a_prev):
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)
        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev) * e_t
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev

    @staticmethod
    def _get_pos_ids():
        return tf.convert_to_tensor([list(range(MAX_PROMPT_LENGTH))], dtype=tf.int32)

    @property
    def image_encoder(self):
        """image_encoder returns the VAE Encoder with pretrained weights.
        Usage:
        ```python
        sd = keras_cv.models.StableDiffusion()
        my_image = np.ones((512, 512, 3))
        latent_representation = sd.image_encoder.predict(my_image)
        ```
        """
        if self._image_encoder is None:
            self._image_encoder = ImageEncoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._image_encoder.compile(jit_compile=True)
        return self._image_encoder

    @property
    def text_encoder(self):
        pass

    @property
    def diffusion_model(self):
        pass

    @property
    def decoder(self):
        """decoder returns the diffusion image decoder model with pretrained weights.
        Can be overriden for tasks where the decoder needs to be modified.
        """
        if self._decoder is None:
            self._decoder = Decoder(self.img_height, self.img_width)
            if self.jit_compile:
                self._decoder.compile(jit_compile=True)
        return self._decoder

    @property
    def tokenizer(self):
        """tokenizer returns the tokenizer used for text inputs.
        Can be overriden for tasks like textual inversion where the tokenizer needs to be modified.
        """
        if self._tokenizer is None:
            self._tokenizer = SimpleTokenizer()
        return self._tokenizer


class StableDiffusion(StableDiffusionBase):
    """Keras implementation of Stable Diffusion.

    Note that the StableDiffusion API, as well as the APIs of the sub-components
    of StableDiffusion (e.g. ImageEncoder, DiffusionModel) should be considered
    unstable at this point. We do not guarantee backwards compatability for
    future changes to these APIs.
    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text description
    (called a "prompt").
    Arguments:
        img_height: Height of the images to generate, in pixel. Note that only
            multiples of 128 are supported; the value provided will be rounded
            to the nearest valid value. Default: 512.
        img_width: Width of the images to generate, in pixel. Note that only
            multiples of 128 are supported; the value provided will be rounded
            to the nearest valid value. Default: 512.
        jit_compile: Whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Default: False.
    Example:
    ```python
    from keras_cv.models import StableDiffusion
    from PIL import Image
    model = StableDiffusion(img_height=512, img_width=512, jit_compile=True)
    img = model.text_to_image(
        prompt="A beautiful horse running through a field",
        batch_size=1,  # How many images to generate at once
        num_steps=25,  # Number of iterations (controls image quality)
        seed=123,  # Set this to always get the same image from the same prompt
    )
    Image.fromarray(img[0]).save("horse.png")
    print("saved at horse.png")
    ```
    References:
    - [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
    - [Original implementation](https://github.com/CompVis/stable-diffusion)
    """

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=False,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL-M license at "
            "https://raw.githubusercontent.com/CompVis/stable-diffusion/main/LICENSE"
        )

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoder(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self) -> tf.keras.Model:
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be modified.
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModel(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model

    @property
    def diffusion_model_ptp(self) -> tf.keras.Model:
        """diffusion_model_ptp returns the diffusion model with modifications for the Prompt-to-Prompt method.

        References
        ----------
        - "Prompt-to-Prompt Image Editing with Cross-Attention Control."
        Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or.
        https://arxiv.org/abs/2208.01626
        """
        if self._diffusion_model_ptp is None:
            if self._diffusion_model is None:
                self._diffusion_model_ptp = self.diffusion_model
            else:
                # Reset the graph - this is to save up memory
                self._diffusion_model.compile(jit_compile=self.jit_compile)
                self._diffusion_model_ptp = self._diffusion_model

            # Add extra variables and callbacks
            self._diffusion_model_ptp = ptp_utils.rename_cross_attention_layers(
                self._diffusion_model_ptp
            )
            self._diffusion_model_ptp = ptp_utils.overwrite_forward_call(
                self._diffusion_model_ptp
            )
            self._diffusion_model_ptp = ptp_utils.set_initial_tf_variables(
                self._diffusion_model_ptp
            )

        return self._diffusion_model_ptp


class StableDiffusionV2(StableDiffusionBase):
    """Keras implementation of Stable Diffusion v2.
    Note that the StableDiffusion API, as well as the APIs of the sub-components
    of StableDiffusionV2 (e.g. ImageEncoder, DiffusionModelV2) should be considered
    unstable at this point. We do not guarantee backwards compatability for
    future changes to these APIs.
    Stable Diffusion is a powerful image generation model that can be used,
    among other things, to generate pictures according to a short text description
    (called a "prompt").
    Arguments:
        img_height: Height of the images to generate, in pixel. Note that only
            multiples of 128 are supported; the value provided will be rounded
            to the nearest valid value. Default: 512.
        img_width: Width of the images to generate, in pixel. Note that only
            multiples of 128 are supported; the value provided will be rounded
            to the nearest valid value. Default: 512.
        jit_compile: Whether to compile the underlying models to XLA.
            This can lead to a significant speedup on some systems. Default: False.
    Example:
    ```python
    from keras_cv.models import StableDiffusionV2
    from PIL import Image
    model = StableDiffusionV2(img_height=512, img_width=512, jit_compile=True)
    img = model.text_to_image(
        prompt="A beautiful horse running through a field",
        batch_size=1,  # How many images to generate at once
        num_steps=25,  # Number of iterations (controls image quality)
        seed=123,  # Set this to always get the same image from the same prompt
    )
    Image.fromarray(img[0]).save("horse.png")
    print("saved at horse.png")
    ```
    References:
    - [About Stable Diffusion](https://stability.ai/blog/stable-diffusion-announcement)
    - [Original implementation](https://github.com/Stability-AI/stablediffusion)
    """

    def __init__(
        self,
        img_height=512,
        img_width=512,
        jit_compile=False,
    ):
        super().__init__(img_height, img_width, jit_compile)
        print(
            "By using this model checkpoint, you acknowledge that its usage is "
            "subject to the terms of the CreativeML Open RAIL++-M license at "
            "https://github.com/Stability-AI/stablediffusion/main/LICENSE-MODEL"
        )

    @property
    def text_encoder(self):
        """text_encoder returns the text encoder with pretrained weights.
        Can be overriden for tasks like textual inversion where the text encoder
        needs to be modified.
        """
        if self._text_encoder is None:
            self._text_encoder = TextEncoderV2(MAX_PROMPT_LENGTH)
            if self.jit_compile:
                self._text_encoder.compile(jit_compile=True)
        return self._text_encoder

    @property
    def diffusion_model(self) -> tf.keras.Model:
        """diffusion_model returns the diffusion model with pretrained weights.
        Can be overriden for tasks where the diffusion model needs to be modified.
        """
        if self._diffusion_model is None:
            self._diffusion_model = DiffusionModelV2(
                self.img_height, self.img_width, MAX_PROMPT_LENGTH
            )
            if self.jit_compile:
                self._diffusion_model.compile(jit_compile=True)
        return self._diffusion_model

    @property
    def diffusion_model_ptp(self) -> tf.keras.Model:
        """diffusion_model_ptp returns the diffusion model with modifications for the Prompt-to-Prompt method.

        References
        ----------
        - "Prompt-to-Prompt Image Editing with Cross-Attention Control."
        Amir Hertz, Ron Mokady, Jay Tenenbaum, Kfir Aberman, Yael Pritch, Daniel Cohen-Or.
        https://arxiv.org/abs/2208.01626
        """
        if self._diffusion_model_ptp is None:
            if self._diffusion_model is None:
                self._diffusion_model_ptp = self.diffusion_model
            else:
                # Reset the graph - this is to save up memory
                self._diffusion_model.compile(jit_compile=self.jit_compile)
                self._diffusion_model_ptp = self._diffusion_model

            # Add extra variables and callbacks
            self._diffusion_model_ptp = ptp_utils.rename_cross_attention_layers(
                self._diffusion_model_ptp
            )
            self._diffusion_model_ptp = ptp_utils.overwrite_forward_call(
                self._diffusion_model_ptp
            )
            self._diffusion_model_ptp = ptp_utils.set_initial_tf_variables(
                self._diffusion_model_ptp
            )

        return self._diffusion_model_ptp
