# Prompt-to-Prompt: Tensorflow Implementation

<sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AoRRd-6oXtFEfx9Ff85GNuTcwSssb5zz?usp=sharing) [![Hugging Face Demo](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/fchollet/stable-diffusion) 
</sub>


### Unofficial Implementation of the paper Prompt-to-Prompt Image Editing with Cross Attention Control

![teaser](assets/teaser.svg)

[Link to the paper](https://arxiv.org/abs/2208.01626) | [Official PyTorch implementation](https://github.com/google/prompt-to-prompt/) | [Project page](https://prompt-to-prompt.github.io/)

This repository contains the Tensorflow/Keras code implementation for the paper "**[Prompt-to-Prompt Image Editing with Cross Attention Control](https://arxiv.org/abs/2208.01626)**".

# 🚀 Quickstart

Current state-of-the-art methods require the user to provide a spatial mask to localize the edit which ignores the original structure and content within the masked region.
The paper proposes a novel technique to edit the generated content of large-scale language models such as [DALL·E 2](https://openai.com/dall-e-2/), [Imagen](https://imagen.research.google/) or [Stable Diffusion](https://github.com/CompVis/stable-diffusion), **by only manipulating the text of the original parsed prompt**.

To achieve this result, the authors present the *Prompt-to-Prompt* framework comprised of two functionalities:

- **Prompt Editing**: where the key idea to edit the generated images is to inject cross-attention maps during the diffusion process, controlling which pixels attend to which tokens of the prompt text.

- **Attention Re-weighting**: that amplifies or attenuates the effect of a word in the generated image. This is done by first attributing a weight to each token and later scaling the attention map assigned to the token. It's a nice alternative to **negative prompting** and **multi-prompting**.

## :gear: Installation

Install dependencies using the `requirements.txt`.

```bash
pip install -r requirements.txt
```

Essentially, you need to have installed [TensorFlow](https://github.com/tensorflow/tensorflow) and [Keras-cv](https://github.com/keras-team/keras-cv/).
## 📚 Notebooks

Try it yourself:

- [**Prompt-to-Prompt: Prompt Editing** - Stable Diffusion](Prompt%20Editing.ipynb) <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1AoRRd-6oXtFEfx9Ff85GNuTcwSssb5zz?usp=sharing) </sub> <br>
Notebook with the paper examples for Stable Diffusion **(WIP)**.

- [**Prompt-to-Prompt: Attention Re-weighting** - Stable Diffusion]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub> <br>
Notebook with examples for the *Prompt-to-Prompt* attention re-weighting approach for Stable Diffusion **(WIP)**.

- [**Prompt-to-Prompt: Demo** - Stable Diffusion]() <sub> [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]() </sub> <br>
Notebook containing an interactive demo of the paper's results **(WIP)**.

# :dart: Prompt-to-Prompt Examples

To start using the *Prompt-to-Prompt* framework, you first need to setup a Tensorflow [strategy](https://www.tensorflow.org/api_docs/python/tf/distribute/Strategy) for running computations across multiple devices (in case you have many).

For example, you can check the available hardware with:

```python
gpus = tf.config.list_physical_devices("GPU")
tpus = tf.config.list_physical_devices("TPU")
print(f"Num GPUs Available: {len(gpus)} | Num TPUs Available: {len(tpus)}")
```

And adjust accordingly to your needs:

```python
import tensorflow as tf

# For running on multiple GPUs
strategy = tf.distribute.MirroredStrategy(devices=["GPU:0", "GPU:1", ...])
# To get the default strategy
strategy = tf.distribute.get_strategy()
...
```

## Prompt Editing

Once the strategy is set, you can start generating images just like in [Keras-cv](https://github.com/keras-team/keras-cv/):

```python
# Imports
import tensorflow as tf
from stable_diffusion import StableDiffusion

generator = StableDiffusion(
    strategy=strategy,
    img_height=512,
    img_width=512,
    jit_compile=False,
)

# Generate text-to-image
img = generator.text_to_image(
    prompt="a photo of a chiwawa with sunglasses and a bandana",
    num_steps=50,
    unconditional_guidance_scale=8,
    seed=5681067,
    batch_size=1,
)
# Generate Prompt-to-Prompt
img_edit = generator.text_to_image_ptp(
    prompt="a photo of a chiwawa with sunglasses and a bandana",
    prompt_edit="a photo of a chiwawa with sunglasses and a pirate bandana",
    num_steps=50,
    unconditional_guidance_scale=8,
    cross_attn2_replace_steps_start=0.0,
    cross_attn2_replace_steps_end=1.0,
    cross_attn1_replace_steps_start=0.8,
    cross_attn1_replace_steps_end=1.0,
    seed=5681067,
    batch_size=1,
)
```

This generates the original and pirate bandana images shown below. You can play around and change the `<bandana>` and `<sunglasses>` attributes and many others!

![teaser](assets/chiwawa.svg)

Another example of prompt editing where one can control the content of the basket just by replacing a couple of words in the prompt:

```python
img_edit = generator.text_to_image_ptp(
    prompt="a photo of basket with apples",
    prompt_edit="a photo of basket with oranges",
    num_steps=50,
    unconditional_guidance_scale=8,
    cross_attn2_replace_steps_start=0.0,
    cross_attn2_replace_steps_end=1.0,
    cross_attn1_replace_steps_start=0.0,
    cross_attn1_replace_steps_end=1.0,
    seed=1597337,
    batch_size=1,
)
```

The image below showcases examples where only the word `<apples>` was replaced with other fruits or animals. Try changing `<basket`> to other recipients (e.g. bowl or nest) and see what happens!

![teaser](assets/bowl.svg)

## Attetion Re-weighting

To manipulate the relative importance of tokens, we've added an argument to pass in both the `text_to_image` and `text_to_image_ptp` methods. You can create an array of weights using our method `create_prompt_weights`.

For example, you generated a pizza that doesn't have enough pineapple on it, you can edit the weights of your prompt:

```python
prompt = "a photo of a pizza with pineapple"
prompt_weights = generator.create_prompt_weights(prompt, [('pineapple', 2)])
```

This will create an array with 1's except on the `pineapple` word position where it will be a 2.

To generate a pizza with more pineapple (yak!), you just need to pass the variable `prompt_weights` to the `text_to_image` method:

```python
img = generator.text_to_image(
    prompt="a photo of a pizza with pineapple",
    num_steps=50,
    unconditional_guidance_scale=8,
    prompt_weights=prompt_weights,
    seed=1234,
    batch_size=1,
)
```

![teaser](assets/pizza_example.svg)

Now you want to reduce the amount of blossom in a tree:

```python
prompt = "A photo of a blossom tree"
prompt_weights = generator.create_prompt_weights(prompt, [('blossom', -1)])

img = generator.text_to_image(
    prompt="A photo of a blossom tree",
    num_steps=50,
    unconditional_guidance_scale=8,
    prompt_weights=prompt_weights,
    seed=1407923,
    batch_size=1,
)
```

Decreasing the weight associated to `<blossom>` will generate the following images.

![teaser](assets/tree.svg)

## Note about the cross-attention parameters

For the prompt editing method, implemented in the function `text_to_image_ptp`, varying the parameters that indicate in which phase of the diffusion process the edited cross-attention maps should get injected (e.g. `cross_attn2_replace_steps_start`, `cross_attn1_replace_steps_start`), may output different results (image below).

The cross-attention and prompt weights hyperparamterers should be tuned according to the users' necessities and desired outputs.

![teaser](assets/doggy.svg)

More info in [bloc97/CrossAttentionControl](https://github.com/bloc97/CrossAttentionControl#usage) and the [paper](https://arxiv.org/abs/2208.01626).

# :ballot_box_with_check: TODO

- [ ] Add tutorials and Google Colabs.
- [ ] Add multi-batch support.
- [ ] Add examples for Stable Diffusion 2.x.

# 👨‍🎓 References

- [keras-cv](https://github.com/keras-team/keras-cv/tree/master/keras_cv/models/generative/stable_diffusion) for the TensorFlow implementation of Stable Diffusion.
- [bloc97/CrossAttentionControl](https://github.com/bloc97/CrossAttentionControl) unofficial implementation of the paper, where the method `get_matching_sentence_tokens` and code logic were used.
- [google/prompt-to-prompt](https://github.com/google/prompt-to-prompt) Official implementation of the paper in PyTorch.

# 🔬 Contributing

Feel free to open an [issue](https://github.com/miguelcalado/prompt-to-prompt-tensorflow/issues) or create a [Pull Request](https://github.com/miguelcalado/prompt-to-prompt-tensorflow/pulls).

For PRs, after implementing the changes please run the `Makefile` for formatting and linting the submitted code:

- `make init`: to create a python environment with all the developer packages (Optional).
- `make format`: to format the code.
- `make lint`: to lint the code.
- `make type_check`: to check for type hints.
- `make all`: to run all the checks.

# :scroll: License

Licensed under the Apache License 2.0. See [LICENSE](LICENSE) to read it in full.
