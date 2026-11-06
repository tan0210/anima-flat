from collections.abc import Callable, Hashable

from triton.testing import do_bench


def _check(*ts):
    for t in ts:
        if not t.is_cuda:
            raise ValueError("CUDA tensors required")


_ROWS: dict[Hashable, int] = {}


def tuned_rows(
    key: Hashable,
    launch: Callable[[int], object],
    choices: tuple[int, ...] = (8, 16, 32, 64),
) -> int:
    """Pick how many tokens each backward program handles, timing the options once per `key`.

    Args:
        key: Identifies the problem (shapes, dtype, variant); the result is cached under it.
        launch: Runs the kernel with the given token count per program.
        choices: Candidate token counts.

    Returns:
        The fastest candidate.
    """
    if key not in _ROWS:
        _ROWS[key] = min( # type: ignore
            choices, key=lambda r: do_bench(lambda: launch(r), warmup=5, rep=20) # type: ignore
        )
    return _ROWS[key]
