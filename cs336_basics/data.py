import numpy as np
import torch


def get_batch(dataset, batch_size, context_length, device):
    max_idx = len(dataset) - context_length - 1

    ix = torch.randint(low=0, high=max_idx + 1, size=(batch_size,))

    x = torch.stack([torch.from_numpy((dataset[i : i + context_length]).astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy((dataset[i + 1 : i + context_length + 1]).astype(np.int64)) for i in ix])

    if "cuda" in device and torch.cuda.is_available():
        x = x.pin_memory().to(device, non_blocking=True)
        y = y.pin_memory().to(device, non_blocking=True)
    else:
        x = x.to(device)
        y = y.to(device)

    return x, y
