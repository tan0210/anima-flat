# Examples

The training loops in this directory are intended to be **copied and adapted**. The library provides the pieces that need to be correct; the examples keep the surrounding training code explicit so you can read, modify, and replace it.

| Example                               | What it shows                                                       |
| ------------------------------------- | ------------------------------------------------------------------- |
| [`train-lora/`](train-lora/README.md) | LoRA training on one or more GPUs, provided as a template to adapt. |

## Remarks

Examples prioritize clarity and ease of modification over performance. They may therefore be substantially less 
optimized than the underlying library.  Profile the example and optimize it for your specific model, hardware, 
and workload if such is your requirement.

## Running an Example

Run examples from the repository root so that both `src` and `examples` are importable. Use the module form with `-m`:

```bash
python -m examples.<name> ...
torchrun --nproc-per-node N -m examples.<name> ...
```

Each example is a self-contained directory with its own README documenting what it does, the expected inputs, and how to run it.
