from collections.abc import Callable, Iterable
import math
from typing import Optional

import torch


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate

    if it >= cosine_cycle_iters:
        return min_learning_rate

    decay_ratio = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return min_learning_rate + coeff * (max_learning_rate - min_learning_rate)


def gradient_clipping(parameters: Iterable[torch.nn.Parameter], max_l2_norm: float, eps: float = 1e-6):
    grads = [p.grad.detach() for p in parameters if p.grad is not None]
    if not grads:
        return

    total_norm = torch.norm(torch.stack([torch.norm(g, 2) for g in grads]), 2)

    if total_norm > max_l2_norm:
        clip_coef = max_l2_norm / (total_norm + eps)
        for g in grads:
            g.mul_(clip_coef)


class SGD(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        defaults = {"lr": lr}
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None if closure is None else closure()
        for group in self.param_groups:
            lr = group["lr"]
            for p in group["params"]:
                if p.grad is None:
                    continue

                state = self.state[p]
                t = state.get("t", 0)
                grad = p.grad.data
                p.data -= lr / math.sqrt(t + 1) * grad
                state["t"] = t + 1

        return loss


class AdamW(torch.optim.Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")

        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure: Optional[Callable] = None):
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            beta1, beta2 = group["betas"]
            eps = group["eps"]
            lambda_ = group["weight_decay"]

            for p in group["params"]:
                if p.grad is None:
                    continue

                # State Init
                state = self.state[p]
                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p.data)  # m
                    state["exp_avg_sq"] = torch.zeros_like(p.data)  # v

                m = state["exp_avg"]
                v = state["exp_avg_sq"]
                state["step"] += 1
                t = state["step"]

                g = p.grad.data

                # 1. Update first moment: m <- beta1 * m + (1 - beta1) * g
                m.mul_(beta1).add_(g, alpha=1 - beta1)

                # 2. Update second moment: v <- beta2 * v + (1 - beta2) * g^2
                v.mul_(beta2).addcmul_(g, g, value=1 - beta2)

                # 3. Compute bias-corrected learning rate
                # alpha_t <- alpha * sqrt(1 - beta2^t) / (1-beta1^t)
                bias_corrections1 = 1 - beta1**t
                bias_corrections2 = 1 - beta2**t
                alpha_t = lr * math.sqrt(bias_corrections2) / (bias_corrections1)

                # 4. theta <- theta - alpha_t * m / (sqrt(v) + eps)
                denom = v.sqrt().add_(eps)
                p.data.addcdiv_(m, denom, value=-alpha_t)

                # Weight deday: theta <- theta - alpha * lambda * theta
                if lambda_ != 0:
                    p.data.mul_(1 - lr * lambda_)

        return loss
