import torch as t
import matplotlib.pyplot as plt
from pathlib import Path
import matplotlib.animation as animation
import numpy as np
from matplotlib.patches import Circle
import matplotlib.colors as mcolors
import argparse
import vandc
import os


class Anim:
    def __init__(self, embeddings, logs):
        self.embedding_history = embeddings
        self.logs = logs

        self.num_steps, self.modulus, _ = embeddings.shape
        self.fig, (self.ax1, self.ax2) = plt.subplots(1, 2, figsize=(16, 8))

        self.num_frames = 150
        self.frame_to_step = np.linspace(
            0, self.num_steps - 1, self.num_frames, dtype=int
        )

        self.setup_embedding_plot()
        self.setup_loss_plot()

        self.ani = animation.FuncAnimation(
            self.fig,
            self.update,
            frames=range(self.num_frames),
            init_func=self.init,
            blit=True,
            interval=5000 / self.num_frames,  # Show all frames within 2 seconds
            repeat=True,
        )

        plt.tight_layout()

    def setup_embedding_plot(self):
        ax = self.ax1

        ax.set_xlim(-1.2, 1.2)
        ax.set_ylim(-1.2, 1.2)
        ax.set_aspect("equal")
        ax.grid(True, linestyle="--", alpha=0.7)
        ax.set_title("Embeddings")

        circle = Circle((0, 0), 1, fill=False, color="gray", linestyle="--")
        ax.add_patch(circle)

        colors = plt.cm.viridis(np.linspace(0, 1, self.modulus))

        self.scatters = []
        for i in range(self.modulus):
            scatter = ax.scatter([], [], color=colors[i], s=100, label=f"{i}")
            self.scatters.append(scatter)

        self.texts = []
        for i in range(self.modulus):
            text = ax.text(0, 0, str(i), fontsize=12, ha="center", va="center")
            self.texts.append(text)

        self.step_text = ax.text(0.05, 0.95, "", transform=ax.transAxes, fontsize=12)

        if self.modulus <= 12:
            ax.legend(loc="upper right")

    def setup_loss_plot(self):
        ax = self.ax2

        ax.set_xlabel("Steps")
        ax.set_ylabel("Loss")
        ax.set_title("Training and Test Loss")
        ax.grid(True, linestyle="--", alpha=0.7)

        (self.train_line,) = ax.plot(
            self.logs.index, self.logs["train_loss"], label="Train Loss", color="blue"
        )
        (self.test_line,) = ax.plot(
            self.logs.index, self.logs["test_loss"], label="Test Loss", color="orange"
        )

        (self.vline,) = ax.plot([], [], color="r", linestyle="--")

        ax.legend()

    def update_vline(self, frame):
        current_step = self.frame_to_step[frame]
        if current_step <= self.logs.index.max():
            ymin, ymax = self.ax2.get_ylim()
            self.vline.set_data([current_step, current_step], [ymin, ymax])
        else:
            max_step = self.logs.index.max()
            ymin, ymax = self.ax2.get_ylim()
            self.vline.set_data([max_step, max_step], [ymin, ymax])

    def init(self):
        for scatter in self.scatters:
            scatter.set_offsets(np.empty((0, 2)))
        for text in self.texts:
            text.set_position((0, 0))
            text.set_text("")
        self.step_text.set_text("")
        self.vline.set_data([], [])
        return (
            self.scatters
            + self.texts
            + [self.step_text, self.vline, self.train_line, self.test_line]
        )

    def update(self, frame):
        step = self.frame_to_step[frame]
        embeddings = self.embedding_history[step].numpy()

        for i, (scatter, text) in enumerate(zip(self.scatters, self.texts)):
            x, y = embeddings[i]
            scatter.set_offsets(np.array([[x, y]]))

            text.set_position((x * 1.05, y * 1.05))
            text.set_text(str(i))

        self.step_text.set_text(f"Step: {step}")

        self.update_vline(frame)

        return (
            self.scatters
            + self.texts
            + [self.step_text, self.vline, self.train_line, self.test_line]
        )

    def show(self):
        plt.show()
        return self.ani


def main():
    parser = argparse.ArgumentParser(description="Visualize embedding evolution")
    parser.add_argument(
        "--run",
        type=str,
        help="Name of the run (will load from .models/{run}.pt)",
    )
    args = parser.parse_args()

    meta = vandc.meta(args.run)
    print(meta)
    model_path = Path(".models") / (meta["run"] + ".pt")
    checkpoint = t.load(model_path)
    embeddings = checkpoint["embedding_history"]

    logs = vandc.fetch(meta["run"])

    anim = Anim(embeddings, logs)
    anim.show()


if __name__ == "__main__":
    main()
