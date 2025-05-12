# %%
import argparse
import torch as t
from tqdm import tqdm
import vandc
from torch.nn.functional import cross_entropy
from einops import einsum
from toy_networks.complex_product import get_data, get_random_data, get_off_diagonal_data
from toy_networks.util import compute_loss


class SimpleMLP(t.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.modulus = args.modulus
        self.embed = args.embed
        self.hidden = args.hidden

        self.w_embed = t.nn.Linear(self.modulus, self.embed)
        self.w_up = t.nn.Linear(self.embed, self.hidden)
        self.w_down = t.nn.Linear(self.hidden, self.embed)
        self.w_unembed = t.nn.Linear(self.embed, self.modulus)

    def forward(self, x):
        args = self.args
        # x : (b, 2)
        x_hot = t.eye(args.modulus)[x, :]  # (b, 2, modulus)
        pair_embed = self.w_embed(x_hot).sum(1)
        hidden = t.relu(self.w_up(pair_embed))
        down = self.w_down(hidden)
        return self.w_unembed(down)


class MLPExperiment:
    def __init__(self, args):
        self.args = args
        t.manual_seed(args.seed)

        self.model = SimpleMLP(args)

        if args.n_train is None:
            self.x_train, self.y_train = get_off_diagonal_data(args.modulus)
        else:
            self.x_train, self.y_train = get_random_data(args.modulus, args.n_train)
        self.x_test, self.y_test = get_data(args.modulus)

        self.optimizer = t.optim.Adam(self.model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

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

            self.model.eval()
            with t.no_grad():
                test_loss, test_acc = compute_loss(self.model, self.x_test, self.y_test)

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
            },
            f".models/{vandc.run_name()}.pt",
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--modulus", type=int, default=12)
    parser.add_argument("--n_train", type=int)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--weight_decay", type=float, default=0.0, help="L2 regularization coefficient")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=43)

    parser.add_argument("--embed", type=float, default=50)
    parser.add_argument("--hidden", type=float, default=100)
    args = parser.parse_args()

    vandc.init(args)
    experiment = MLPExperiment(args)
    experiment.run()
    experiment.save(name=vandc.run_name())
    vandc.commit()

    df = vandc.fetch(vandc.run_name())
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.lineplot(data=df, x="step", y="train_loss")
    sns.lineplot(data=df, x="step", y="test_loss")
    plt.show()
