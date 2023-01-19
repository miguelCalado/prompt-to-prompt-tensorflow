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
from keras_cv.models.stable_diffusion.diffusion_model import DiffusionModel
from keras_cv.models.stable_diffusion.image_encoder import ImageEncoder
from keras_cv.models.stable_diffusion.text_encoder import TextEncoder
from tqdm import tqdm

import ptp_utils

MAX_TEXT_LEN = 77
NUM_TRAIN_TIMESTEPS = 1000


class StableDiffusion:
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
        strategy: tf.distribute,
        img_height: int = 512,
        img_width: int = 512,
        jit_compile: bool = False,
        download_weights: bool = True,
    ):
        self.strategy = strategy

        # UNet requires multiples of 2**7 = 128
        img_height = round(img_height / 128) * 128
        img_width = round(img_width / 128) * 128
        self.img_height = img_height
        self.img_width = img_width

        self.tokenizer = SimpleTokenizer()

        text_encoder, diffusion_model, decoder, encoder = get_models(
            strategy, img_height, img_width, download_weights=download_weights
        )
        self.text_encoder = text_encoder
        self.diffusion_model = diffusion_model
        self.decoder = decoder
        self.encoder = encoder

        if jit_compile:
            self.text_encoder.compile(jit_compile=True)
            self.diffusion_model.compile(jit_compile=True)
            self.decoder.compile(jit_compile=True)
            self.encoder.compile(jit_compile=True)

        # Add extra variables and callbacks
        self.diffusion_model = ptp_utils.rename_cross_attention_layers(
            self.diffusion_model
        )
        self.diffusion_model = ptp_utils.overwrite_forward_call(self.diffusion_model)
        self.diffusion_model = ptp_utils.set_initial_tf_variables(self.diffusion_model)

    def text_to_image(
        self,
        prompt: str,
        num_steps: int = 50,
        unconditional_guidance_scale: float = 7.5,
        prompt_weights: np.ndarray = np.array([]),
        batch_size: int = 1,
        seed: Optional[int] = None,
    ) -> np.ndarray:
        """Generate an image based on a prompt text.

        Parameters
        ----------
        prompt : str
            Text containing the information for the model to generate.
        num_steps : int, optional
            Number of diffusion steps (controls image quality), by default 50.
        unconditional_guidance_scale : float, optional
            Controls how closely the image should adhere to the prompt, by default 7.5.
        prompt_weights : List[float], optional
            Set of weights for each prompt token.
            This is used for manipulating the importance of the prompt tokens,
            increasing or decreasing the importance assigned to each word.
        batch_size : int, optional
            Batch size (number of images to generate), by default 1.
        seed : Optional[int], optional
            Number to seed the random noise, by default None.

        Returns
        -------
        np.ndarray
            Generated image.
        """
        # Tokenize prompt
        tokens_conditional = self.tokenize_prompt(prompt, batch_size)

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(MAX_TEXT_LEN)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        conditional_context = self.text_encoder.predict(
            [tokens_conditional, pos_ids], batch_size=batch_size, verbose=0
        )

        # Encode unconditional tokens (and their positions into an "unconditional context vector"
        # TODO: arrange this
        unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
        unconditional_context = self.text_encoder.predict(
            [unconditional_tokens, pos_ids], batch_size=batch_size, verbose=0
        )

        # Update prompt weights variable
        if prompt_weights.size:
            self.diffusion_model = ptp_utils.add_prompt_weights(
                diff_model=self.diffusion_model, prompt_weights=prompt_weights
            )
            self.diffusion_model = ptp_utils.update_prompt_weights_usage(
                diff_model=self.diffusion_model, use=True
            )

        # Scheduler
        # TODO: Add diffusers LMSDiscreteScheduler
        timesteps = np.arange(1, 1000, 1000 // num_steps)

        # Get initial random noise
        latent = self._get_initial_diffusion_noise(batch_size, seed)

        # Get Initial parameters
        alphas, alphas_prev = self._get_initial_alphas(timesteps)

        # Diffusion stage
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")

            timesteps_t = np.array([timestep])
            t_emb = self._get_timestep_embedding(timesteps_t)
            t_emb = np.repeat(t_emb, batch_size, axis=0)

            # Predict the unconditional noise residual
            unconditional_latent = self.diffusion_model.predict(
                [latent, t_emb, unconditional_context], batch_size=batch_size, verbose=0
            )

            # Predict the conditional noise residual
            conditional_latent = self.diffusion_model.predict(
                [latent, t_emb, conditional_context], batch_size=batch_size, verbose=0
            )

            # Perform guidance
            e_t = unconditional_latent + unconditional_guidance_scale * (
                conditional_latent - unconditional_latent
            )

            a_t, a_prev = alphas[index], alphas_prev[index]
            latent = self._get_x_prev(latent, e_t, a_t, a_prev)

        # Decode image
        img = self._get_decoding_stage(latent, batch_size)

        # Reset control variables
        self.diffusion_model = ptp_utils.reset_initial_tf_variables(
            self.diffusion_model
        )

        return img

    def text_to_image_ptp(
        self,
        prompt: str,
        prompt_edit: str,
        method: str,
        self_attn_steps: Union[float, Tuple[float, float]],
        cross_attn_steps: Union[float, Tuple[float, float]],
        attn_edit_weights: np.ndarray = np.array([]),
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
        # Tokenize prompt
        tokens_conditional = self.tokenize_prompt(prompt, batch_size)

        # Tokenize prompt edit
        tokens_conditional_edit = self.tokenize_prompt(prompt_edit, batch_size)

        # Encode prompt tokens (and their positions) into a "context vector"
        pos_ids = np.array(list(range(MAX_TEXT_LEN)))[None].astype("int32")
        pos_ids = np.repeat(pos_ids, batch_size, axis=0)
        conditional_context = self.text_encoder.predict(
            [tokens_conditional, pos_ids], batch_size=batch_size, verbose=0
        )

        # Encode prompt edit tokens
        conditional_context_edit = self.text_encoder.predict(
            [tokens_conditional_edit, pos_ids], batch_size=batch_size, verbose=0
        )

        # Encode unconditional tokens (and their positions into an "unconditional context vector"
        unconditional_tokens = np.array(_UNCONDITIONAL_TOKENS)[None].astype("int32")
        unconditional_tokens = np.repeat(unconditional_tokens, batch_size, axis=0)
        unconditional_tokens = tf.convert_to_tensor(unconditional_tokens)
        unconditional_context = self.text_encoder.predict(
            [unconditional_tokens, pos_ids], batch_size=batch_size, verbose=0
        )

        ## PTP stuff
        if isinstance(self_attn_steps, float):
            self_attn_steps = (0.0, self_attn_steps)
        if isinstance(cross_attn_steps, float):
            cross_attn_steps = (0.0, cross_attn_steps)

        if method=='refine':
            # Get the mask and indices of the difference between the original prompt token's and the edited one
            mask, indices = ptp_utils.get_matching_sentence_tokens(
                tokens_conditional[0], tokens_conditional_edit[0]
            )

            # Add the mask and indices to the diffusion model
            self.diffusion_model = ptp_utils.put_mask_dif_model(
                self.diffusion_model, mask, indices
            )

        # Update prompt weights variable
        if attn_edit_weights.size:
            self.diffusion_model = ptp_utils.add_prompt_weights(
                diff_model=self.diffusion_model, prompt_weights=attn_edit_weights
            )

        # Scheduler
        # TODO: Add diffusers LMSDiscreteScheduler
        timesteps = np.arange(1, 1000, 1000 // num_steps)

        # Get initial random noise
        latent = self._get_initial_diffusion_noise(batch_size, seed)
        # Get Initial parameters
        alphas, alphas_prev = self._get_initial_alphas(timesteps)

        # Diffusion stage
        progbar = tqdm(list(enumerate(timesteps))[::-1])
        for index, timestep in progbar:
            progbar.set_description(f"{index:3d} {timestep:3d}")

            timesteps_t = np.array([timestep])
            t_emb = self._get_timestep_embedding(timesteps_t)
            t_emb = np.repeat(t_emb, batch_size, axis=0)

            t_scale = 1 - (timestep / NUM_TRAIN_TIMESTEPS)

            # Update Cross-Attention mode to 'unconditional'
            self.diffusion_model = ptp_utils.update_cross_attn_mode(
                diff_model=self.diffusion_model, mode="unconditional"
            )

            # Predict the unconditional noise residual
            unconditional_latent = self.diffusion_model.predict(
                [latent, t_emb, unconditional_context], batch_size=batch_size, verbose=0
            )

            # Save last cross attention activations
            self.diffusion_model = ptp_utils.update_cross_attn_mode(
                diff_model=self.diffusion_model, mode="save"
            )
            # Predict the conditional noise residual
            _ = self.diffusion_model.predict(
                [latent, t_emb, conditional_context], batch_size=batch_size, verbose=0
            )

            # TODO: # if method=='reweigh':
            # Edit the Cross-Attention layer activations
            if cross_attn_steps[0] <= t_scale <= cross_attn_steps[1]:
                if method=='replace':
                    # Use cross attention from the original prompt (M_t)
                    self.diffusion_model = ptp_utils.update_cross_attn_mode(
                        diff_model=self.diffusion_model, mode="use_last", attn_suffix="attn2"
                    )
                elif method=='refine':
                    self.diffusion_model = ptp_utils.update_cross_attn_mode(
                        diff_model=self.diffusion_model, mode="edit", attn_suffix="attn2"
                    )
            else:
                # Use cross attention from the edited prompt (M^*_t)
                self.diffusion_model = ptp_utils.update_cross_attn_mode(
                    diff_model=self.diffusion_model, mode="injection", attn_suffix="attn2"
                )
            
            # Edit the self-Attention layer activations
            if self_attn_steps[0] <= t_scale <= self_attn_steps[1]:
                # Use self attention from the original prompt (M_t)
                self.diffusion_model = ptp_utils.update_cross_attn_mode(
                    diff_model=self.diffusion_model, mode="use_last", attn_suffix="attn1"
                )
            else:
                # Use self attention from the edited prompt (M^*_t)
                self.diffusion_model = ptp_utils.update_cross_attn_mode(
                    diff_model=self.diffusion_model, mode="injection", attn_suffix="attn1"
                )

            # Predict the edited conditional noise residual
            conditional_latent_edit = self.diffusion_model.predict(
                [latent, t_emb, conditional_context_edit],
                batch_size=batch_size,
                verbose=0,
            )
            
            # Perform guidance
            e_t = unconditional_latent + unconditional_guidance_scale * (
                conditional_latent_edit - unconditional_latent
            )

            a_t, a_prev = alphas[index], alphas_prev[index]
            latent = self._get_x_prev(latent, e_t, a_t, a_prev)

        # Decode image
        img = self._get_decoding_stage(latent, batch_size)

        # Reset control variables
        self.diffusion_model = ptp_utils.reset_initial_tf_variables(
            self.diffusion_model
        )

        return img

    def tokenize_prompt(self, prompt: str, batch_size: int) -> np.ndarray:
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
        if len(inputs) > MAX_TEXT_LEN:
            raise ValueError(f"Prompt is too long (should be <= {MAX_TEXT_LEN} tokens)")
        phrase = inputs + [49407] * (MAX_TEXT_LEN - len(inputs))
        phrase = np.array(phrase)[None].astype("int32")
        phrase = np.repeat(phrase, batch_size, axis=0)

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
        weights = np.ones(MAX_TEXT_LEN)

        # Get the prompt tokens
        tokens = self.tokenize_prompt(prompt, batch_size=1)

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

    def _get_initial_alphas(self, timesteps):

        alphas = [_ALPHAS_CUMPROD[t] for t in timesteps]
        alphas_prev = [1.0] + alphas[:-1]

        return alphas, alphas_prev

    def _get_initial_diffusion_noise(self, batch_size: int, seed: Optional[int]):
        return tf.random.normal(
            (batch_size, self.img_height // 8, self.img_width // 8, 4), seed=seed
        )

    def _get_timestep_embedding(
        self, timesteps, dim: int = 320, max_period: int = 10000
    ):
        half = dim // 2
        freqs = np.exp(
            -math.log(max_period) * np.arange(0, half, dtype="float32") / half
        )
        args = np.array(timesteps) * freqs
        embedding = np.concatenate([np.cos(args), np.sin(args)])
        return tf.convert_to_tensor(embedding.reshape(1, -1), dtype=tf.float32)

    def _get_decoding_stage(self, latent, batch_size):
        decoded = self.decoder.predict(latent, batch_size=batch_size, verbose=0)
        decoded = ((decoded + 1) / 2) * 255
        return np.clip(decoded, 0, 255).astype("uint8")

    def _get_x_prev(self, x, e_t, a_t, a_prev):
        sqrt_one_minus_at = math.sqrt(1 - a_t)
        pred_x0 = (x - sqrt_one_minus_at * e_t) / math.sqrt(a_t)
        # Direction pointing to x_t
        dir_xt = math.sqrt(1.0 - a_prev) * e_t
        x_prev = math.sqrt(a_prev) * pred_x0 + dir_xt
        return x_prev


def get_models(
    strategy: tf.distribute,
    img_height: int,
    img_width: int,
    download_weights: bool = True,
) -> Union[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.Model]:
    """Load and download the models weights.

    Parameters
    ----------
    strategy : tf.distribute
        TensorFlow strategy for running computations across multiple devices.
    img_height : int
        Image hight.
    img_width : int
        Image width.
    download_weights : bool, optional
        Flag used to download the models weights, by default True.

    Returns
    -------
    Union[tf.keras.Model, tf.keras.Model, tf.keras.Model, tf.keras.Model]
        The text encoder, diffusion model, decoder and encoder model's
    """
    with strategy.scope():

        # Create text encoder
        text_encoder = TextEncoder(MAX_TEXT_LEN, download_weights=download_weights)

        # Creation diffusion UNet
        diffusion_model = DiffusionModel(
            img_height, img_width, MAX_TEXT_LEN, download_weights=download_weights
        )

        # Create decoder
        decoder = Decoder(img_height, img_width, download_weights=download_weights)

        # Create encoder
        encoder = ImageEncoder(img_height, img_width, download_weights=download_weights)

    return text_encoder, diffusion_model, decoder, encoder
