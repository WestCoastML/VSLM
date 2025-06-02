import torch
import torch.nn.functional as F


def gumbel_softmax(logits, temperature=1.0, hard=False):
    # Sample Gumbel noise
    noise = -torch.log(-torch.log(torch.rand_like(logits) + 1e-10) + 1e-10)
    y = F.softmax((logits + noise) / temperature, dim=-1)

    if hard:
        # Convert to one-hot
        y_hard = torch.zeros_like(y)
        y_hard.scatter_(1, y.argmax(dim=1, keepdim=True), 1.0)
        y = (y_hard - y).detach() + y  # Straight-through estimator
    return y
