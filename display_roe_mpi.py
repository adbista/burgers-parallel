#!/usr/bin/env python3
"""Wizualizacja wyników równoległej metody Roe dla równania Burgersa."""

import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio


RESULTS_FILE = "results_roe_mpi.txt"
GIF_FILENAME = "burgers_roe_mpi.gif"


def load_results(filename):
    """Czyta plik z kolumnami x, t, u i grupuje je według czasu."""
    frames = []
    saved_times = []
    xs = []
    us = []
    x_template = None
    last_t = None

    with open(filename, "r") as f:
        lines = f.readlines()

    for line in lines:
        line = line.strip()

        if line.startswith("#"):
            continue

        if not line:
            if xs:
                frames.append(np.array(us))
                saved_times.append(last_t)
                if x_template is None:
                    x_template = np.array(xs)
                xs = []
                us = []
            continue

        x_str, t_str, u_str = line.split()
        xs.append(float(x_str))
        us.append(float(u_str))
        last_t = float(t_str)

    if xs:
        frames.append(np.array(us))
        saved_times.append(last_t)
        if x_template is None:
            x_template = np.array(xs)

    if not frames or x_template is None:
        raise ValueError("Brak danych w pliku wynikowym.")

    return frames, saved_times, x_template


def generate_gif(frames, saved_times, x, gif_filename):
    umin = min(np.min(frame) for frame in frames)
    umax = max(np.max(frame) for frame in frames)
    L = x[-1]

    images = []
    for u_frame, t in zip(frames, saved_times):
        fig, ax = plt.subplots()
        ax.plot(x, u_frame, lw=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Roe Burgers MPI, t = {t:.3f}")
        ax.set_xlim(0.0, L)
        ax.set_ylim(umin - 0.1 * abs(umin), umax + 0.1 * abs(umax))
        fig.tight_layout()
        fig.canvas.draw()

        image = np.asarray(fig.canvas.buffer_rgba())
        images.append(image[:, :, :3])  # drop alpha channel

        plt.close(fig)

    imageio.mimsave(gif_filename, images, fps=10)
    print(f"GIF zapisany jako {gif_filename}")


if __name__ == "__main__":
    print(f"Wczytuję wyniki z: {RESULTS_FILE}")
    frames, times, x = load_results(RESULTS_FILE)
    print(f"Liczba zapisanych kroków czasowych: {len(frames)}")
    generate_gif(frames, times, x, GIF_FILENAME)
