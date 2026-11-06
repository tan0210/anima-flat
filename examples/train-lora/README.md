# LoRA Training

A small, self-contained LoRA training loop for pre-encoded latents and text embeddings. It runs on a single GPU or across multiple GPUs with `torchrun`, and is designed to be easy to copy and adapt.

```bash
python -m examples.train_lora --checkpoint MODEL.safetensors --data data.pt
torchrun --nproc-per-node 2 -m examples.train_lora --checkpoint MODEL.safetensors --data data.pt
```

Run these commands from the repository root.

### Arguments

The main configuration options are:

- `--steps`: number of training steps
- `--lr`: learning rate
- `--warmup`: warmup steps
- `--rank`: LoRA rank
- `--batch`: batch size per GPU
- `--accum`: number of micro-batches to accumulate per step
- `--out`: output path for the LoRA adapter

## What It Does

1. Loads the checkpoint and attaches rank-`--rank` LoRA adapters to every block. All base-model weights remain frozen.
2. For each step, draws `--accum` micro-batches, computes the flow-matching loss with `flow_loss`, and accumulates gradients.
3. Clips the gradient norm to 1, updates the adapters with AdamW, and applies a warmup-plus-cosine learning-rate schedule.
4. Saves the trained adapter to `--out`. The adapter can later be loaded with `load_lora` or `load(..., lora=...)`.

Block recomputation (`model.checkpointing`) is enabled by default. This reduces activation memory at the cost of additional compute.

## Data Format

The `--data` argument points to a file created with `torch.save`. It must contain a dictionary with three tensors:

| Key       | Shape          | Meaning                                                                             |
| --------- | -------------- | ----------------------------------------------------------------------------------- |
| `latents` | `[N, H, W, C]` | Clean VAE latents, channels last.                                                   |
| `context` | `[N, S, D]`    | Text embeddings, zero-padded to a common sequence length `S`, as at inference time. |
| `uncond`  | `[1, S, D]`    | Empty-prompt embeddings with the same `S`; used for caption dropout.                |

All samples must have the same latent spatial dimensions because batches are sampled randomly and stacked. For mixed resolutions, bucket the data by latent size and draw each batch from a single bucket.

## Files

| File             | Role                                                                                       |
| ---------------- | ------------------------------------------------------------------------------------------ |
| `__main__.py`    | Command-line arguments and the training loop.                                              |
| `data.py`        | Loads the dataset and draws random batches. Replace this to use a different data pipeline. |
| `schedule.py`    | Learning-rate schedule.                                                                    |
| `distributed.py` | Multi-GPU setup and the two collectives required by the training loop.                     |

## Multi-GPU Training

Multi-GPU training uses data parallelism over the LoRA adapters without `DistributedDataParallel`. Every GPU holds the complete frozen model and its own copy of the adapters.

The loop performs three distributed operations:

1. **`broadcast_params`** runs once at startup so every rank begins with identical adapter weights.
2. **`average_grads`** runs after the final micro-batch of each step, communicating gradients once per optimizer step rather than once per micro-batch.
3. Each rank clips and applies the averaged gradients locally. Because the gradients are identical, the adapter copies remain synchronized.

Each rank uses a different random seed, so ranks draw different batches and noise. The effective global batch size is:

```text
--batch × --accum × number of GPUs
```

Only rank 0 writes the output adapter.

This example does not cover model sharding (FSDP2) or full-model training.

## Adapting the Example

- **Different data:** Replace `data.py`. The training loop only requires `latents`, `context`, and `uncond`.
- **Different objective or noise schedule:** Replace `flow_loss` with your own loss. `sample_sigmas` provides the noise levels.
- **Different optimizer or learning-rate schedule:** Modify `__main__.py` and `schedule.py`.
- **Resuming training:** Save and restore the optimizer state alongside the adapter. This example only saves the adapter and does not implement optimizer-state checkpointing.
