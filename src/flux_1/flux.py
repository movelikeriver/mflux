import PIL
import mlx.core as mx
from PIL import Image
from tqdm import tqdm

from flux_1.config.config import Config
from flux_1.config.model_config import ModelConfig
from flux_1.config.runtime_config import RuntimeConfig
from flux_1.models.text_encoder.clip_encoder.clip_encoder import CLIPEncoder
from flux_1.models.text_encoder.t5_encoder.t5_encoder import T5Encoder
from flux_1.models.transformer.transformer import Transformer
from flux_1.models.vae.vae import VAE
from flux_1.post_processing.image_util import ImageUtil
from flux_1.tokenizer.clip_tokenizer import TokenizerCLIP
from flux_1.tokenizer.t5_tokenizer import TokenizerT5
from flux_1.tokenizer.tokenizer_handler import TokenizerHandler
from flux_1.weights.weight_handler import WeightHandler


class Flux1:

    def __init__(self, repo_id: str):
        self.model_config = ModelConfig.from_repo(repo_id)

        # Initialize the tokenizers
        tokenizers = TokenizerHandler.load_from_disk_or_huggingface(repo_id, self.model_config.max_sequence_length)
        self.t5_tokenizer = TokenizerT5(tokenizers.t5, max_length=self.model_config.max_sequence_length)
        self.clip_tokenizer = TokenizerCLIP(tokenizers.clip)

        # Initialize the models
        weights = WeightHandler.load_from_disk_or_huggingface(repo_id)
        self.vae = VAE(weights.vae)
        self.transformer = Transformer(weights.transformer)
        self.t5_text_encoder = T5Encoder(weights.t5_encoder)
        self.clip_text_encoder = CLIPEncoder(weights.clip_encoder)

    @staticmethod
    def from_repo(repo_id: str) -> "Flux1":
        return Flux1(repo_id)

    @staticmethod
    def from_alias(alias: str) -> "Flux1":
        return Flux1(ModelConfig.from_alias(alias).model_name)

    def generate_image(self, seed: int, prompt: str, config: Config = Config()) -> PIL.Image.Image:
        # Create a new runtime config based on the model type and input parameters
        config = RuntimeConfig(config, self.model_config)

        # Create the latents
        latents = mx.random.normal(
            shape=[1, (config.height // 16) * (config.width // 16), 64],
            key=mx.random.key(seed)
        )

        # Embedd the prompt
        t5_tokens = self.t5_tokenizer.tokenize(prompt)
        clip_tokens = self.clip_tokenizer.tokenize(prompt)
        prompt_embeds = self.t5_text_encoder.forward(t5_tokens)
        pooled_prompt_embeds = self.clip_text_encoder.forward(clip_tokens)

        for t in tqdm(range(config.num_inference_steps)):
            # Predict the noise
            noise = self.transformer.predict(
                t=t,
                prompt_embeds=prompt_embeds,
                pooled_prompt_embeds=pooled_prompt_embeds,
                hidden_states=latents,
                config=config,
            )

            # Take one denoise step
            dt = config.sigmas[t + 1] - config.sigmas[t]
            latents += noise * dt

            # To enable progress tracking
            mx.eval(latents)

        # Decode the latent array
        latents = Flux1._unpack_latents(latents, config.height, config.width)
        decoded = self.vae.decode(latents)
        return ImageUtil.to_image(decoded)

    @staticmethod
    def _unpack_latents(latents: mx.array, height: int, width: int) -> mx.array:
        latents = mx.reshape(latents, (1, width // 16, height // 16, 16, 2, 2))
        latents = mx.transpose(latents, (0, 3, 1, 4, 2, 5))
        latents = mx.reshape(latents, (1, 16, width // 16 * 2, height // 16 * 2))
        return latents

    def encode(self, path: str) -> mx.array:
        array = ImageUtil.to_array(Image.open(path))
        return self.vae.encode(array)

    def decode(self, code: mx.array) -> PIL.Image.Image:
        decoded = self.vae.decode(code)
        return ImageUtil.to_image(decoded)
