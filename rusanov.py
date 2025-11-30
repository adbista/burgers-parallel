import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio

def build_matrix_A(N, mu, dt, dx):
    """
    Buduje macierz A dla schematu implicit Rusanova (sekcja 2.1.1).

    Równanie krokowe:
        A u^{n+1} = b

    Współczynnik:
        r = μ Δt / Δx^2

    Elementy macierzy dla punktów wewnętrznych i = 1..N-2:
        A_ij = 1 + 2 r          dla i = j      (diagonala)
        A_ij = -r               dla |i - j| = 1 (pod i nad diagonala)
    """
    r = mu * dt / (dx ** 2)
    Nint = N - 2
    A = np.zeros((Nint, Nint))

    diag = 1.0 + 2.0 * r
    off = -r

    # diagonala: A_ii = 1 + 2 r
    np.fill_diagonal(A, diag)

    # pod i nad diagonala: A_{i,i±1} = -r
    for i in range(Nint - 1):
        A[i, i + 1] = off
        A[i + 1, i] = off

    return A


def rusanov_step(u, A, dt, dx, mu, w):
    """
    Jeden krok czasowy metody Rusanova dla równania Burgersa.

    Z PDF (2.1.1):

        A u^{n+1} = b

        b_i = u_i^n
              - 0.5 * (Δt / Δx) * (F_{i+1}^n - F_i^n)
              + 0.5 * w * (Δt / Δx) * (φ_{i+1}^n - φ_{i-1}^n)

    gdzie

        F_i^n  = 0.5 * (u_i^n)^2
        φ_i^n  = 0.5 * (u_i^n + u_{i-1}^n) * (u_i^n - u_{i-1}^n)

    W kodzie używamy tego samego indeksu `i` dla wszystkich członów.
    """
    N = len(u)
    Nint = N - 2

    # F_i^n = 0.5 * (u_i^n)^2
    F = 0.5 * u ** 2

    # φ_i^n = 0.5 * (u_i^n + u_{i-1}^n) * (u_i^n - u_{i-1}^n)
    phi = np.zeros_like(u)
    phi[1:] = 0.5 * (u[1:] + u[:-1]) * (u[1:] - u[:-1])

    b = np.zeros(Nint)
    coeff = 0.5 * dt / dx          # 0.5 * Δt / Δx
    coeff_phi = 0.5 * w * dt / dx  # 0.5 * w * Δt / Δx

    # b_i – tylko punkty wewnętrzne i = 1..N-2
    for i in range(1, N - 1):
        # (F_{i+1}^n - F_i^n)
        conv_term = F[i + 1] - F[i]

        # (φ_{i+1}^n - φ_{i-1}^n)
        phi_right = phi[i + 1] if i + 1 < N else 0.0
        phi_left = phi[i - 1] if i - 1 >= 0 else 0.0
        visc_art_term = phi_right - phi_left

        b[i - 1] = (
            u[i]
            - coeff * conv_term
            + coeff_phi * visc_art_term
        )

    # Rozwiązanie układu A u_int^{n+1} = b (punkty wewnętrzne)
    u_int_next = np.linalg.solve(A, b)

    # Składamy pełny wektor u^{n+1} z warunkami brzegowymi Dirichleta
    u_next = np.zeros_like(u)
    u_next[0] = 0.0          # u_0^{n+1}
    u_next[-1] = 0.0         # u_{N-1}^{n+1}
    u_next[1:-1] = u_int_next

    return u_next


def initial_condition(x):
    """
    Warunek początkowy u(x,0) = u_0(x):

        u_0(x) = 2  dla 0.1 < x < 0.4
        u_0(x) = 0  w pozostałych punktach

    Skok w profilu prędkości generuje narastający stromy front
    (numeryczny odpowiednik fali uderzeniowej).
    """
    u0 = np.zeros_like(x)
    u0[(x > 0.1) & (x < 0.4)] = 2.0
    return u0

# def initial_condition(x, mu=0.1):
#     # Wzór (1.9) z PDF (strona 8)
#     # u0(x) = (2 * mu * pi * sin(pi * x)) / (a + cos(pi * x))
    
#     a = 1.1 # Przykładowa wartość (autor pisze "for a > 1")
#     return (2 * mu * np.pi * np.sin(np.pi * x)) / (a + np.cos(np.pi * x))



def simulate_and_save(
    N=201,
    L=1.0,
    T_final=0.5,
    mu=0.01,
    w=0.5,
    cfl=0.4,
    save_every=10,
    results_filename="results_rusanov.txt",
    gif_filename="burgers_rusanov.gif"
):
    """
    Główna pętla czasowa:

      1. Dyskretyzacja dziedziny: x_i, Δx
      2. Dobór kroku czasu Δt z warunku CFL
      3. Zbudowanie macierzy A
      4. Ustawienie warunku początkowego u^0
      5. Dla n = 0..n_steps:
           a) zapis u^n do pliku results_rusanov.txt (co save_every kroków)
           b) obliczenie b_i dokładnie z wzoru z 2.1.1
           c) rozwiązanie A u^{n+1} = b
    """
    x = np.linspace(0.0, L, N)
    dx = x[1] - x[0]

    # prosty warunek CFL przy założeniu max |u| ≈ 2
    umax = 2.0
    dt = cfl * dx / max(umax, 1e-8)
    n_steps = int(T_final / dt) + 1

    # macierz A (stała w czasie)
    A = build_matrix_A(N, mu, dt, dx)

    # u^0
    u = initial_condition(x)

    saved_profiles = []
    saved_times = []

    with open(results_filename, "w") as f:
        f.write("# x t u\n")

    umin = np.min(u)
    umax_val = np.max(u)

    for n in range(n_steps):
        t = n * dt

        # zapis wybranych kroków czasowych
        if n % save_every == 0 or n == n_steps - 1:
            saved_profiles.append(u.copy())
            saved_times.append(t)

            with open(results_filename, "a") as f:
                for xi, ui in zip(x, u):
                    f.write(f"{xi:.8f} {t:.8f} {ui:.8f}\n")
                f.write("\n")

            umin = min(umin, np.min(u))
            umax_val = max(umax_val, np.max(u))

        # krok Rusanova: A u^{n+1} = b (z b_i jak w PDF)
        u = rusanov_step(u, A, dt, dx, mu, w)

    # tworzenie gifa z zapisanych profili
    images = []
    for u_frame, t in zip(saved_profiles, saved_times):
        fig, ax = plt.subplots()
        ax.plot(x, u_frame, lw=2)
        ax.set_xlabel("x")
        ax.set_ylabel("u(x,t)")
        ax.set_title(f"Rusanov Burgers, t = {t:.3f}")
        ax.set_xlim(0.0, L)
        ax.set_ylim(umin - 0.1 * abs(umin), umax_val + 0.1 * abs(umax_val))

        fig.tight_layout()
        fig.canvas.draw()
        w_fig, h_fig = fig.canvas.get_width_height()
        image = np.frombuffer(fig.canvas.tostring_rgb(), dtype="uint8")
        image = image.reshape((h_fig, w_fig, 3))
        images.append(image)
        plt.close(fig)

    imageio.mimsave(gif_filename, images, fps=10)
    print(f"Zapisano profile do {results_filename}")
    print(f"Zapisano gifa do {gif_filename}")


if __name__ == "__main__":
    simulate_and_save()
