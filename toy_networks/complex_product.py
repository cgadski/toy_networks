import argparse
import torch as t
from tqdm import tqdm
import vandc
from einops import einsum
from typing import Tuple


def get_train_data(n: int):
    values = t.arange(n)
    i, j = t.meshgrid(values, values, indexing="ij")
    mask = i != j
    x = t.stack([i[mask], j[mask]], dim=1)
    y = x.sum(dim=1) % n
    return x, y


def get_data(b: int, n: int):
    x = t.randint(0, n, (b, 2))
    y = x.sum(-1) % n
    return x, y


class ComplexProduct(t.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.embed = 2
        if not args.fixed_mult:
            self.embed = args.embed or 2

        self.w_embed = t.nn.Parameter(t.randn(args.modulus, self.embed))
        self.unembed = t.nn.Parameter(t.randn(self.embed, args.modulus))
        if args.fixed_mult:
            self.multiply = t.tensor(
                [[[1, 0], [0, 1]], [[0, 1], [-1, 0]]], dtype=t.float
            )
        else:
            print("Using trainable bilinear layer!")
            self.multiply = t.nn.Parameter(t.randn([self.embed] * 3))

        self.project()

    def project(self):
        self.w_embed.data = self.w_embed.data / self.w_embed.data.norm(
            dim=-1, keepdim=True
        )

    def forward(self, x):
        embedded = self.w_embed[x]
        mult: t.Tensor = einsum(
            embedded[:, 0],
            embedded[:, 1],
            self.multiply,
            "b i, b j, i j k -> b k",
        )
        return mult @ self.unembed


def compute_loss(model, x, y):
    logits = model(x)
    loss = t.nn.functional.cross_entropy(logits, y)
    accuracy = (logits.argmax(dim=-1) == y).float().mean()
    return loss, accuracy


def run(args):
    t.manual_seed(args.seed)

    vandc.init(args)
    x_train, y_train = get_train_data(args.modulus)
    x_test, y_test = get_data(1024, args.modulus)
    model = ComplexProduct(args)

    optimizer = t.optim.Adam(model.parameters(), lr=args.lr)

    embedding_history = []

    mult_decay = args.mult_decay

    for step in tqdm(range(args.steps)):
        model.train()
        optimizer.zero_grad()
        train_loss, train_acc = compute_loss(model, x_train, y_train)

        # Add weight decay for multiply parameter if enabled
        if mult_decay > 0 and not args.fixed_mult:
            mult_norm = model.multiply.norm()
            train_loss = train_loss + mult_decay * mult_norm

        train_loss.backward()
        optimizer.step()
        model.project()

        model.eval()
        with t.no_grad():
            test_loss, test_acc = compute_loss(model, x_test, y_test)

        embedding_history.append(model.w_embed.data.clone())

        d = {
            "train_loss": train_loss.item(),
            "train_accuracy": train_acc.item(),
            "test_loss": test_loss.item(),
            "test_accuracy": test_acc.item(),
        }
        vandc.log(d)

    t.save(
        {
            "model": model.state_dict(),
            "embedding_history": t.stack(embedding_history),
        },
        f".models/{vandc.run_name()}.pt",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--fixed-mult", type=bool, action=argparse.BooleanOptionalAction
    )
    parser.add_argument("--embed", type=int)
    parser.add_argument(
        "--mult-decay",
        type=float,
        default=0.1,
        help="Weight decay coefficient for multiply parameter",
    )
    args = parser.parse_args()

    run(args)
