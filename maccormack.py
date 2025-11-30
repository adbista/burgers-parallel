import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
import time


def initial_condition(x):
    """
    Prosty skok: u = 2 na fragmencie [0.1, 0.4], 0 poza nim.
    Daje ładny, stromy front.
    """
    u0 = np.zeros_like(x)
    u0[(x > 0.1) & (x < 0.4)] = 2.0
    return u0


def maccormack_step(u, dt, dx, mu):
    """
    Jeden krok czasowy metody MacCormacka (wersja zwektoryzowana).
    
    Predictor (u*):
      Strumień: Forward difference (F_{j+1} - F_j)
      Dyfuzja: Central difference
      
    Corrector (u^{n+1}):
      Strumień: Backward difference (F*_j - F*_{j-1})
      Dyfuzja: Central difference
    """
    # Stałe
    r = mu * dt / (dx ** 2)
    coef = dt / dx
    
    # --- 1. PREDYKTOR ---
    F = 0.5 * u ** 2
    u_star = u.copy()
    
    # Odpowiednik: F[j+1] - F[j]
    # Indeksy 1:-1 to wnętrze. 2: to przesunięcie w prawo (j+1), 1:-1 to środek (j)
    flux_fwd = F[2:] - F[1:-1]
    diff_term = u[2:] - 2.0*u[1:-1] + u[:-2]
    
    u_star[1:-1] = u[1:-1] - coef * flux_fwd + r * diff_term
    
    # Warunki brzegowe predyktora
    u_star[0] = 0.0
    u_star[-1] = 0.0

    # --- 2. KOREKTOR ---
    F_star = 0.5 * u_star ** 2
    u_next = u.copy()
    
    # Odpowiednik: F*[j] - F*[j-1]
    # Indeksy: 1:-1 to środek (j), :-2 to przesunięcie w lewo (j-1)
    flux_bwd = F_star[1:-1] - F_star[:-2]
    diff_term_star = u_star[2:] - 2.0*u_star[1:-1] + u_star[:-2]
    
    u_next[1:-1] = 0.5 * (
        u[1:-1] + u_star[1:-1] 
        - coef * flux_bwd 
        + r * diff_term_star
    )
    
    # Warunki brzegowe
    u_next[0] = 0.0
    u_next[-1] = 0.0
    
    return u_next


def simulate_maccormack(
    N=201,
    L=1.0,
    T_final=0.5,
    mu=0.01,
    CFL=0.4,
    save_every=5,
    results_filename="results_maccormack.txt",
    gif_filename="burgers_maccormack.gif"
):
    # Siatka przestrzenna
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    # Warunek początkowy
    u = initial_condition(x)

    # Stały krok czasu na podstawie maksymalnej początkowej prędkości
    umax0 = np.max(np.abs(u))
    if umax0 < 1e-8:
        umax0 = 1.0
    dt = CFL * dx / umax0
    n_steps = int(T_final / dt) + 1

    # Przygotowanie pliku wynikowego
    with open(results_filename, "w") as f:
        f.write("# x t u\n")

    # Do gifa
    frames = []
    saved_times = []

    umin = np.min(u)
    umax = np.max(u)

    # Pomiar czasu obliczeń
    t0 = time.perf_counter()

    for n in range(n_steps):
        t = n * dt

        if n % save_every == 0 or n == n_steps - 1:
            # zapis do pliku
            with open(results_filename, "a") as f:
                for xi, ui in zip(x, u):
                    f.write(f"{xi:.8f} {t:.8f} {ui:.8f}\n")
                f.write("\n")

            saved_times.append(t)
            umin = min(umin, np.min(u))
            umax = max(umax, np.max(u))

            # tymczasowo zapisz profil, rysunek zrobimy później
            frames.append(u.copy())

        # krok MacCormacka
        u = maccormack_step(u, dt, dx, mu)

    t1 = time.perf_counter()
    print(f"Czas obliczeń (symulacja + zapis do pliku): {t1 - t0:.4f} s")

    # Tworzenie gifa
    images = []
    for u_frame, t in zip(frames, saved_times):
        fig, ax = plt.subplots()
        ax.plot(x, u_frame, lw=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"MacCormack Burgers, t = {t:.3f}")
        ax.set_xlim(0.0, L)
        ax.set_ylim(umin - 0.1 * abs(umin), umax + 0.1 * abs(umax))
        fig.tight_layout()
        fig.canvas.draw()

        w_fig, h_fig = fig.canvas.get_width_height()
        buf = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        img = buf.reshape((h_fig, w_fig, 3))
        images.append(img)
        plt.close(fig)

    imageio.mimsave(gif_filename, images, fps=10)
    print(f"Wyniki zapisane w {results_filename}")
    print(f"Gif zapisany jako {gif_filename}")


if __name__ == "__main__":
    simulate_maccormack()