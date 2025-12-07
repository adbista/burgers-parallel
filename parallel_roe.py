#!/usr/bin/env python3

from mpi4py import MPI
import numpy as np
import sys


def initial_condition(x):
    u0 = np.zeros_like(x)
    u0[(x > 0.1) & (x < 0.4)] = 2.0
    return u0


def decompose_1d(N, size, rank):
    base = N // size
    rem = N % size
    start = rank * base + min(rank, rem)
    end = (rank + 1) * base + min(rank + 1, rem)
    return start, end


def apply_dirichlet(u_local, global_start, N):
    local_n = u_local.shape[0] - 2
    if local_n <= 0:
        return
    if global_start == 0:
        u_local[1] = 0.0
    if global_start + local_n - 1 == N - 1:
        u_local[local_n] = 0.0


def exchange_halo(comm, u_local, left, right):
    local_n = u_local.shape[0] - 2
    if local_n <= 0:
        return

    send_left = np.array(u_local[1], dtype=np.float64)
    send_right = np.array(u_local[local_n], dtype=np.float64)
    recv_left = np.empty(1, dtype=np.float64)
    recv_right = np.empty(1, dtype=np.float64)

    comm.Sendrecv(sendbuf=send_left, dest=left,
                  recvbuf=recv_right, source=right)
    comm.Sendrecv(sendbuf=send_right, dest=right,
                  recvbuf=recv_left, source=left)

    if left != MPI.PROC_NULL:
        u_local[0] = recv_left[0]
    else:
        u_local[0] = 0.0

    if right != MPI.PROC_NULL:
        u_local[local_n + 1] = recv_right[0]
    else:
        u_local[local_n + 1] = 0.0


def compute_time_step(u_local, dx, mu, CFL, comm):
    interior = u_local[1:-1]
    umax_local = np.max(np.abs(interior)) if interior.size else 0.0
    umax = comm.allreduce(umax_local, op=MPI.MAX) if comm is not None else umax_local
    if umax < 1e-10:
        umax = 1.0

    dt_conv = CFL * dx / umax
    if mu > 0.0:
        dt_diff = 0.5 * dx * dx / mu
        return min(dt_conv, dt_diff)
    return dt_conv


def reconstruct_global(N, size, local_data_list):
    u_global = np.empty(N, dtype=np.float64)
    base = N // size
    rem = N % size
    for r in range(size):
        start = r * base + min(r, rem)
        end = (r + 1) * base + min(r + 1, rem)
        u_global[start:end] = local_data_list[r]
    return u_global


def dump_solution(comm, rank, size, u_local, x_global, t, results_filename):
    local_data = u_local[1:-1].copy()
    gathered = comm.gather(local_data, root=0)
    if rank == 0:
        u_global = reconstruct_global(x_global.size, size, gathered)
        with open(results_filename, "a") as f:
            for xi, ui in zip(x_global, u_global):
                f.write(f"{xi:.8f} {t:.8f} {ui:.8f}\n")
            f.write("\n")


def roe_step_local(u_local, dt, dx, mu, global_start, N):
    local_n = u_local.shape[0] - 2
    if local_n <= 0:
        return u_local

    apply_dirichlet(u_local, global_start, N)

    F = 0.5 * u_local ** 2
    phi = np.zeros_like(u_local)
    phi[1:] = 0.5 * np.abs(u_local[1:] + u_local[:-1]) * (u_local[1:] - u_local[:-1])

    r = mu * dt / (dx ** 2)
    coef = 0.5 * dt / dx

    conv = F[2:] - F[1:-1] - phi[2:] + phi[1:-1]
    diff = u_local[2:] - 2.0 * u_local[1:-1] + u_local[:-2]

    u_next = u_local.copy()
    u_next[1:-1] = u_local[1:-1] - coef * conv + r * diff

    apply_dirichlet(u_next, global_start, N)
    return u_next


def simulate_roe_parallel(
    N=201,
    L=1.0,
    T_final=0.5,
    mu=0.01,
    CFL=0.4,
    save_every=5,
    results_filename="results_roe.txt"
):
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    x_global = np.linspace(0.0, L, N)
    dx = x_global[1] - x_global[0]
    u0_global = initial_condition(x_global)

    if rank == 0:
        print("Roe Burgers 1D MPI")
        print(f"N = {N}, L = {L}, T_final = {T_final}, mu = {mu}")
        print(f"Procesy MPI = {size}")
        with open(results_filename, "w") as f:
            f.write("# x t u\n")

    global_start, global_end = decompose_1d(N, size, rank)
    local_n = global_end - global_start

    u_local = np.zeros(local_n + 2, dtype=np.float64)
    u_local[1:-1] = u0_global[global_start:global_end]
    apply_dirichlet(u_local, global_start, N)

    left = rank - 1 if rank > 0 else MPI.PROC_NULL
    right = rank + 1 if rank < size - 1 else MPI.PROC_NULL

    dt = compute_time_step(u_local, dx, mu, CFL, comm)
    if rank == 0:
        print(f"dx = {dx:.6e}, początkowy dt = {dt:.6e}")

    comm.Barrier()
    t0 = MPI.Wtime()

    t = 0.0
    step = 0
    dump_solution(comm, rank, size, u_local, x_global, t, results_filename)
    eps_t = 1e-12

    while t < T_final - eps_t:
        exchange_halo(comm, u_local, left, right)

        dt = compute_time_step(u_local, dx, mu, CFL, comm)
        dt = min(dt, T_final - t)

        u_local = roe_step_local(u_local, dt, dx, mu, global_start, N)

        if not np.all(np.isfinite(u_local)):
            raise FloatingPointError(
                "Wykryto wartości NaN/Inf w rozwiązaniu. Zmniejsz CFL lub sprawdź parametry."
            )

        t += dt
        step += 1

        if step % save_every == 0 or t >= T_final - eps_t:
            dump_solution(comm, rank, size, u_local, x_global, t, results_filename)

    comm.Barrier()
    t1 = MPI.Wtime()
    if rank == 0:
        print(f"Czas obliczeń (symulacja + zapis): {t1 - t0:.4f} s")


if __name__ == "__main__":
    if len(sys.argv) >= 2:
        N = int(sys.argv[1])
    else:
        N = 201

    if len(sys.argv) >= 3:
        T_final = float(sys.argv[2])
    else:
        T_final = 0.5

    simulate_roe_parallel(N=N, T_final=T_final)
