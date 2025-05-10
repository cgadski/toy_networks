import argparse
import torch as t
from tqdm import tqdm
import vandc
from torch.nn.functional import cross_entropy
from einops import einsum
from typing import Tuple


def get_off_diagonal_data(n: int):
    values = t.arange(n)
    i, j = t.meshgrid(values, values, indexing="ij")
    mask = i != j
    x = t.stack([i[mask], j[mask]], dim=1)
    y = x.sum(dim=1) % n
    return x, y


def get_data(n: int):
    values = t.arange(n)
    i, j = t.meshgrid(values, values, indexing="ij")
    x = t.stack([i.flatten(), j.flatten()], dim=1)
    y = x.sum(dim=1) % n
    return x, y


def get_random_data(n: int, b: int):
    x_all, y_all = get_data(n)
    indices = t.randperm(n * n)[:b]
    return x_all[indices], y_all[indices]


class ComplexProduct(t.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.embed = 2

        self.w_embed = t.nn.Parameter(t.randn(args.modulus, self.embed))
        self.multiply = t.tensor(
            [[[1, 0], [0, 1]], [[0, 1], [-1, 0]]],
            dtype=t.float,
        )
        self.confidence = t.nn.Parameter(t.tensor(1.0))

        self.project()

    def project(self):
        self.w_embed.data.div_(self.w_embed.data.norm(dim=-1, keepdim=True))

    def forward(self, x):
        embedded = self.w_embed[x]
        mult: t.Tensor = einsum(
            embedded[:, 0],
            embedded[:, 1],
            self.multiply,
            "b i, b j, i j k -> b k",
        )
        return mult @ self.w_embed.T


def compute_loss(model, x, y):
    logits = model(x)
    loss = t.nn.functional.cross_entropy(logits, y)
    accuracy = (logits.argmax(dim=-1) == y).float().mean()
    return loss, accuracy


class ComplexProductExperiment:
    def __init__(self, args):
        self.args = args
        t.manual_seed(args.seed)

        self.model = ComplexProduct(args)

        if args.n_train is None:
            self.x_train, self.y_train = get_off_diagonal_data(args.modulus)
        else:
            self.x_train, self.y_train = get_random_data(args.modulus, args.n_train)
        self.x_test, self.y_test = get_data(args.modulus)

        self.optimizer = t.optim.SGD(self.model.parameters(), lr=args.lr)

    def get_test_accuracy(self):
        self.model.eval()
        with t.no_grad():
            _, test_acc = compute_loss(self.model, self.x_test, self.y_test)
        return test_acc.item()

    def log(self, d):
        vandc.log(d)

    def run(self):
        self.embedding_history = []

        for step in range(self.args.steps):
            self.model.train()
            self.optimizer.zero_grad()
            train_loss, train_acc = compute_loss(self.model, self.x_train, self.y_train)

            train_loss.backward()
            self.optimizer.step()
            self.model.project()

            self.model.eval()
            with t.no_grad():
                test_loss, test_acc = compute_loss(self.model, self.x_test, self.y_test)

            self.embedding_history.append(self.model.w_embed.data.clone())

            d = {
                "train_loss": train_loss.item(),
                "test_loss": test_loss.item(),
                "train_accuracy": train_acc.item(),
                "test_accuracy": test_acc.item(),
            }
            self.log(d)

    def save(self, name: str):
        t.save(
            {
                "model": self.model.state_dict(),
                "embedding_history": t.stack(self.embedding_history),
            },
            f".models/{vandc.run_name()}.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=12)
    parser.add_argument("--n_train", type=int)
    parser.add_argument("--lr", type=float, default=1)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)
    args = parser.parse_args()

    vandc.init(args)
    experiment = ComplexProductExperiment(args)
    experiment.run()
    experiment.save(name=vandc.run_name())
    vandc.commit()

    df = vandc.fetch(vandc.run_name())
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(data=df, x="step", y="test_accuracy")
    sns.lineplot(data=df, x="step", y="train_loss")
    plt.show()
