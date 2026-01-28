import torch

from cs336_basics.optimizer import SGD


def run_experiment(lr):
    print(f"\nTesting LR: {lr}")
    weights = torch.nn.Parameter(5 * torch.randn((10, 10)))
    opt = SGD([weights], lr)

    for t in range(100):
        opt.zero_grad()
        loss = (weights**2).mean()
        print(f"Iter {t}: {loss.item():.2f}")
        loss.backward()
        opt.step()


if __name__ == "__main__":
    for lr in [1.0, 10.0, 100.0, 1000.0]:
        run_experiment(lr)
