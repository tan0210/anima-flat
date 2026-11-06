import functools
import math

import torch
import torch.nn.functional as F

try:
    from quack.gemm_interface import gemm_act as _gemm_act
    from quack.gemm_interface import gemm_dact as _gemm_dact
except Exception: # noqa: BLE001
    _gemm_act = None
    _gemm_dact = None

_DTYPES = (torch.bfloat16, torch.float16)


@functools.cache
def _sm90_plus(index: int) -> bool:
    return torch.cuda.get_device_capability(index)[0] >= 9


def have_quack(device: torch.device | None = None) -> bool:
    """Report whether the fused QuACK path is usable.

    Args:
        device: CUDA device to check; defaults to the current device.

    Returns:
        True if QuACK imported and the device is SM90 or newer.
    """
    if _gemm_act is None or not torch.cuda.is_available():
        return False
    index = torch.cuda.current_device() if device is None else (device.index or 0)
    return _sm90_plus(index)


@torch.library.custom_op("anima_flat::linear_gelu_quack", mutates_args=())
def _linear_gelu_quack(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    _, post = _gemm_act(  # type: ignore
        x,
        weight.T,
        activation="gelu_erf",  # type: ignore
        store_preact=False,
    )
    return post


@_linear_gelu_quack.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return x.new_empty(x.shape[0], weight.shape[0])


@torch.library.custom_op("anima_flat::linear_gelu_quack_train", mutates_args=())
def _linear_gelu_quack_train(
    x: torch.Tensor, weight: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    pre, post = _gemm_act(  # type: ignore
        x,
        weight.T,
        activation="gelu_erf",  # type: ignore
        store_preact=True,
    )
    return post, pre  # type: ignore


@_linear_gelu_quack_train.register_fake
def _(x: torch.Tensor, weight: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    n = weight.shape[0]
    return x.new_empty(x.shape[0], n), x.new_empty(x.shape[0], n)


def _setup_context(ctx, inputs, output) -> None:
    x, weight = inputs
    _, pre = output
    ctx.mark_non_differentiable(pre)
    ctx.save_for_backward(x, weight, pre)


def _backward(ctx, grad_post: torch.Tensor, grad_pre: torch.Tensor | None):
    x, weight, pre = ctx.saved_tensors
    # d/dh gelu(h) = Phi(h) + h * phi(h)
    h = pre.float()
    cdf = 0.5 * (1.0 + torch.erf(h * (1.0 / math.sqrt(2.0))))
    pdf = torch.exp(-0.5 * h * h) * (1.0 / math.sqrt(2.0 * math.pi))
    dh = (grad_post.float() * (cdf + h * pdf)).to(x.dtype)
    dx = dh @ weight if ctx.needs_input_grad[0] else None
    dw = dh.T @ x if ctx.needs_input_grad[1] else None
    return dx, dw


_linear_gelu_quack_train.register_autograd(_backward, setup_context=_setup_context)


def linear_gelu(x: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ weight.T)`` (exact erf GELU, no bias).

    Uses QuACK's fused forward epilogue on SM90+ and ``F.gelu(F.linear(...))``
    otherwise.

    Args:
        x: (..., K) bf16/fp16 activations.
        weight: (N, K) weight, same dtype as ``x``.

    Returns:
        (..., N) tensor with the dtype of ``x``.
    """
    if (
        _gemm_act is not None
        and x.is_cuda
        and x.dtype in _DTYPES
        and x.dtype == weight.dtype
        and _sm90_plus(x.device.index or 0)
    ):
        x2 = x.reshape(-1, x.shape[-1])
        if torch.is_grad_enabled() and (x.requires_grad or weight.requires_grad):
            y, _ = _linear_gelu_quack_train(x2, weight)
        else:
            y = _linear_gelu_quack(x2, weight)
        return y.view(*x.shape[:-1], weight.shape[0])
    return F.gelu(F.linear(x, weight))


def _quack_ok(x: torch.Tensor, *weights: torch.Tensor) -> bool:
    return (
        _gemm_act is not None
        and x.is_cuda
        and x.dtype in _DTYPES
        and all(w.dtype == x.dtype for w in weights)
        and _sm90_plus(x.device.index or 0)
    )


class _MLP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w2):  # x: (M, K), w1: (H, K), w2: (N, H)
        pre, post = _gemm_act(x, w1.T, activation="gelu_erf", store_preact=True)  # type: ignore
        ctx.save_for_backward(x, w1, w2, pre)  # post is rebuilt in backward
        return F.linear(post, w2)

    @staticmethod
    def backward(ctx, dout):  # type: ignore
        x, w1, w2, pre = ctx.saved_tensors
        dout = dout.contiguous()
        dpre, post = _gemm_dact(dout, w2, pre, activation="gelu_erf")  # type: ignore
        dx = dpre @ w1 if ctx.needs_input_grad[0] else None
        dw1 = dpre.T @ x if ctx.needs_input_grad[1] else None
        dw2 = dout.T @ post if ctx.needs_input_grad[2] else None
        return dx, dw1, dw2


def mlp(x: torch.Tensor, w_up: torch.Tensor, w_down: torch.Tensor) -> torch.Tensor:
    """Compute ``gelu(x @ w_up.T) @ w_down.T`` (exact erf GELU, no biases).

    Faster and lighter on memory than separate layers on SM90+ when gradients are needed;
    otherwise equivalent to ``F.linear(linear_gelu(x, w_up), w_down)``.

    Args:
        x: (..., K) bf16/fp16 activations.
        w_up: (H, K) weight, same dtype as ``x``.
        w_down: (N, H) weight, same dtype as ``x``.

    Returns:
        (..., N) tensor with the dtype of ``x``.
    """
    needs_grad = torch.is_grad_enabled() and (
        x.requires_grad or w_up.requires_grad or w_down.requires_grad
    )
    if needs_grad and _quack_ok(x, w_up, w_down):
        y = _MLP.apply(x.reshape(-1, x.shape[-1]), w_up, w_down)
        return y.view(*x.shape[:-1], w_down.shape[0])
    return F.linear(linear_gelu(x, w_up), w_down)
