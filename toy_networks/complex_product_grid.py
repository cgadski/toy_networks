import argparse
import torch as t
from tqdm import tqdm
import vandc
from toy_networks.complex_product import ComplexProductExperiment


def make_args(n_train):
    return argparse.Namespace(
        modulus=12,
        n_train=n_train,
        lr=1,
        steps=300,
        seed=43,
    )


class MyExperiment(ComplexProductExperiment):
    def __init__(self, n_train):
        super().__init__(make_args(n_train))
        self.iter = 0

    def log(self, d):
        vandc.log(
            {
                "n_train": self.args.n_train,
                "iter": self.iter,
                **d,
            }
        )
        self.iter += 1


if __name__ == "__main__":
    vandc.init(argparse.Namespace())

    for n_train in tqdm(range(1, 12 * 12)):
        experiment = MyExperiment(n_train)
        experiment.run()
