from dataclasses import dataclass

import torch


@dataclass
class Dataset:
    """Pre-encoded training data held in memory.

    Attributes:
        latents: Clean latents, shape `[N, H, W, C]`.
        context: Text embeddings, shape `[N, S, text_dim]`.
        uncond: Empty-prompt embeddings, shape `[1, S, text_dim]`.
    """

    latents: torch.Tensor
    context: torch.Tensor
    uncond: torch.Tensor

    def batch(
        self, size: int, device: torch.device
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Draw a random batch with replacement.

        Args:
            size: Number of samples.
            device: Device of the result.

        Returns:
            `(latents, context)` for the sampled indices.
        """
        idx = torch.randint(self.latents.shape[0], (size,))
        return self.latents[idx].to(device), self.context[idx].to(device)


def load_dataset(path: str) -> Dataset:
    """Read a `torch.save`d dict with `latents`, `context` and `uncond` tensors.

    Args:
        path: Path of the file.

    Returns:
        The dataset.
    """
    d = torch.load(path)
    return Dataset(d["latents"], d["context"], d["uncond"])
