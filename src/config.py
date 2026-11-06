import json
from dataclasses import asdict, dataclass

FORMAT = "anima-flat-1"


@dataclass
class FlatConfig:
    """Architecture hyperparameters of the original checkpoint and Flat variant.

    Attributes:
      layers: Numbers of transformer blocks.
      heads: Attention heads per block.
      head_dim: Width of one attention and head.
      text_dim: Width of the text-context embeddings fed to cross-attention.
      lora_dim: Rank of the low-rank adaLN modulation projections.
      mlp_dim: Hidden width of the MLP.
      qk_eps: Epsilon of the q/k RMS norms.
      rope_scale: NTK scaling of the `(height, width)` rotary frequencies.
    """

    layers: int = 28
    heads: int = 16
    head_dim: int = 128
    text_dim: int = 1024
    lora_dim: int = 256
    mlp_dim: int = 8192
    in_ch: int = 16
    qk_eps: float = 1e-5
    rope_scale: tuple = (4.0, 4.0)

    @property
    def dim(self) -> int:
        """Model width.

        Returns:
          ``heads * head_dim``.
        """
        return self.heads * self.head_dim

    def dumps(self) -> str:
        """Serialize the config to JSON.

        Returns:
          JSON string for safetensors metadata.
        """
        return json.dumps(asdict(self))

    @classmethod
    def loads(cls, text: str) -> "FlatConfig":
        """Deserialize a config produced by ``dumps``.

        Args:
          text: JSON string.

        Returns:
          The reconstructed config.
        """
        d = json.loads(text)
        d["rope_scale"] = tuple(d["rope_scale"])
        return cls(**d)
