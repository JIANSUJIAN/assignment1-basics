import json

import torch


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {"model": model.state_dict(), "optimizer": optimizer.state_dict(), "iteration": iteration}
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["iteration"]


def save_config(config: dict, path: str):
    with open(path, "w") as f:
        json.dump(config, f, indent=4)


def load_config(path: str):
    with open(path) as f:
        return json.load(f)
