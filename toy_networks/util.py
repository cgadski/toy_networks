from torch.nn.functional import cross_entropy
import torch as t

def compute_loss(model, x, y):
    logits = model(x)
    loss = t.nn.functional.cross_entropy(logits, y)
    accuracy = (logits.argmax(dim=-1) == y).float().mean()
    return loss, accuracy

def svd_proj(x, d=2):
    return t.linalg.svd(x).U[:, :d]
