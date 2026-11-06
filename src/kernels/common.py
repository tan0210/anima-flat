def _check(*ts):
    for t in ts:
        if not t.is_cuda:
            raise ValueError("CUDA tensors required")
