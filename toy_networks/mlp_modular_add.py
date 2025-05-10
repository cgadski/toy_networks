import argparse
import torch as t
from tqdm import tqdm
import vandc
from torch.nn.functional import cross_entropy
from einops import einsum


class SimpleMLP(t.nn.Module):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.modulus = args.modulus
        self.embed = args.embed
        self.hidden = args.hidden

        self.w_embed = t.nn.Linear(self.modulus, self.embed)
        self.w_up = t.nn.Linear(self.embed * 2, self.hidden)
        self.w_down = t.nn.Linear(self.hidden, self.embed)
        self.w_unembed = t.nn.Linear(self.embed, self.modulus)

    def forward(self, x):
        embedded = self.w_embed[x]
