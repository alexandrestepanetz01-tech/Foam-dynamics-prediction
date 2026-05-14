import numpy as np
import math
import time
import os
import multiprocessing as mp
import termistyle as ts


def update_neighbor_list(rx, ry, N, Lx, Ly, list_max, r_neighbor):

    neighbor_list = - np.ones((N, list_max), dtype=int)
    count_neighbor = np.zeros(N, dtype=int)

    for i in range(N):
        for j in range(N):
            if j != i:

                dx = rx[i] - rx[j]
                dy = ry[i] - ry[j]

                if dx > 0.5 * Lx:
                    dx -= Lx
                if dx < -0.5 * Lx:
                    dx += Lx
                if dy > 0.5 * Ly:
                    dy -= Ly
                if dy < -0.5 * Ly:
                    dy += Ly

                rij = math.sqrt(dx * dx + dy * dy)

                if rij < r_neighbor:
                    neighbor_list[i, count_neighbor[i]] = j
                    count_neighbor[i] += 1

    return neighbor_list, count_neighbor


def cal_force_neighbor(alpha, rx, ry, N, Lx, Ly, neighbor_list, count_neighbor, sigm):

    fx = np.zeros(N)
    fy = np.zeros(N)

    for i in range(N):
        for k in range(count_neighbor[i]):
            j = neighbor_list[i, k]

            dx = rx[i] - rx[j]
            dy = ry[i] - ry[j]

            if dx > 0.5 * Lx:
                dx -= Lx
            elif dx < -0.5 * Lx:
                dx += Lx

            if dy > 0.5 * Ly:
                dy -= Ly
            elif dy < -0.5 * Ly:
                dy += Ly

            rij = math.sqrt(dx * dx + dy * dy)
            sigm_ij = 0.5 * (sigm[i] + sigm[j])

            if rij < sigm_ij:
                f = ((1.0 - (rij / sigm_ij)) ** (alpha - 1.0)) / (rij * sigm_ij)
                fx[i] += dx * f
                fy[i] += dy * f

    return fx, fy


def main(N, Lx, Ly, total_step, output_step, r_neighbor, dt, timestep, sigm_obs, trajectory, data, cnf, list_max, alpha,
         fx_ext, K, w, d):
    """
    Generates 10 files. The first with the initial positions at MDsteps=0, the one at MDsteps=1 and others at
    MDsteps=output_step depending on the number of total simulation steps.

    """

    print(ts.YELLOW + f"Computing cnf {cnf}..." + ts.RESET)
    cnf_start = time.time()

    rx = data[N:, 0]
    ry = data[N:, 1]
    rx_before = data[:N, 0]
    ry_before = data[:N, 1]
    sigm = data[:N, 2]

    initial_config_ref = np.column_stack((rx, ry))

    n_traj = 11
    """"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
    d = d
    for traj_id in range(n_traj):
    # for traj_id in [11]:
        try:
            print(f"Chaos trajectory {traj_id}")

            traj_start = time.time()

            if traj_id == 0:
                working_config = initial_config_ref.copy()
            else:
                perturbation = np.random.uniform(-d / 2.0, d / 2.0, size=(N, 2))
                CM_shift = np.mean(perturbation, axis=0)

                working_config = initial_config_ref + perturbation - CM_shift
                working_config[:, 0] %= Lx

            rx = working_config[:, 0].copy()
            ry = working_config[:, 1].copy()

            neighbor_list, count_neighbor = update_neighbor_list(rx, ry, N, Lx, Ly, list_max, r_neighbor)

            dir_name = f"Chaos_MDstep{total_step}_timestep{timestep}_d{d}"
            os.makedirs(dir_name, exist_ok=True)
            dir_name_trajectory = f"configuration_{cnf}"
            output_dir = os.path.join(dir_name, dir_name_trajectory)
            os.makedirs(output_dir, exist_ok=True)

            file_name = f'configuration_{cnf}_trajectory_{traj_id}.txt'
            with open(os.path.join(output_dir, file_name), "w") as traj:

                if trajectory == 1:
                    traj.write(f"{0.0:.6f}\t{N}\t{Lx:.10f}\t{Ly:.10f}\n")
                    snapshot = np.column_stack((
                        initial_config_ref[:, 0],
                        initial_config_ref[:, 1],
                        np.zeros(N),
                        sigm
                    ))

                    np.savetxt(traj, snapshot, fmt="%.16e")

                for MDstep in range(total_step):

                    dx = rx - rx_before
                    dy = ry - ry_before

                    dx[dx > 0.5 * Lx] -= Lx
                    dx[dx < -0.5 * Lx] += Lx

                    dy[dy > 0.5 * Ly] -= Ly
                    dy[dy < -0.5 * Ly] += Ly

                    dr = np.sqrt(dx ** 2 + dy ** 2)
                    dr_max = np.max(dr)

                    sigm_max = 1.0
                    dr_th = (r_neighbor - sigm_max) / 2.0

                    if dr_max > dr_th:
                        print(f"UPDATE: dr_max={dr_max:f}, dr_th={dr_th:f}")
                        neighbor_list, count_neighbor = update_neighbor_list(rx, ry, N, Lx, Ly, list_max, r_neighbor)
                        rx_before = rx.copy()
                        ry_before = ry.copy()

                    fx, fy = cal_force_neighbor(alpha, rx, ry, N, Lx, Ly, neighbor_list, count_neighbor, sigm)

                    rx += (fx + fx_ext) * dt
                    ry += fy * dt

                    F_L_wall = K * (w - ry[ry < w])
                    ry[ry < w] += F_L_wall * dt

                    F_R_wall = -K * (ry[ry > (Ly - w)] - (Ly - w))
                    ry[ry > (Ly - w)] += F_R_wall * dt

                    r = np.sqrt((rx - Lx / 2) ** 2 + (ry - Ly / 2) ** 2)

                    F_x_obs = np.zeros_like(rx)
                    F_y_obs = np.zeros_like(ry)

                    F_x_obs[r < sigm_obs / 2] = -K * (r[r < sigm_obs / 2] - sigm_obs / 2) * (rx[r < sigm_obs / 2] - Lx / 2) / r[r < sigm_obs / 2]
                    F_y_obs[r < sigm_obs / 2] = -K * (r[r < sigm_obs / 2] - sigm_obs / 2) * (ry[r < sigm_obs / 2] - Ly / 2) / r[r < sigm_obs / 2]
                    rx[r < sigm_obs / 2] += F_x_obs[r < sigm_obs / 2] * dt
                    ry[r < sigm_obs / 2] += F_y_obs[r < sigm_obs / 2] * dt

                    rx %= Lx
                    rx %= Lx

                    if not np.all(np.isfinite(rx)) or not np.all(np.isfinite(ry)):
                        raise RuntimeError(f"NaN/inf detected: cnf={cnf}, traj={traj_id}, MDstep={MDstep}")

                    if trajectory == 1 and MDstep % output_step == 0:
                        traj.write(f"{MDstep * dt:.6f}\t{N}\t{Lx:.10f}\t{Ly:.10f}\n")
                        snapshot = np.column_stack((
                            rx,
                            ry,
                            np.zeros(N),
                            sigm
                        ))

                        np.savetxt(traj, snapshot, fmt="%.16e")
        except Exception as e:
            print(f"ERROR cnf={cnf}, traj={traj_id}: {repr(e)}")
            raise

        traj_duration = time.time() - traj_start
        print(f"Trajectory {traj_id} done in {traj_duration/60:.2f} mins")
    cnf_duration = time.time() - cnf_start
    print(ts.YELLOW + f"Cnf {cnf} done in {cnf_duration/60:.2f} mins" + ts.RESET)


def mlp_main(d, cnfs):

    N = 900
    alpha = 2.5
    phi = 1.2
    delta = 0.1
    gamma = 3
    w = 0.5
    sigm_obs = 5.0
    K = 10.0
    fx_ext = 0.001
    dt = 0.1
    run = 1
    r_neighbor = 3.0
    trajectory = 1
    output_step = 3000
    total_step = 6001

    list_max = 500

    n_processes = 10

    file = "output1000_timestep=30.txt"

    simu_start = time.time()

    with open(file, 'r') as f:
        headers = f.readline().split()

        Lx = float(headers[2])
        Ly = float(headers[3])
        timestep = float(int(dt * output_step))

        print(f"N={N}")
        print(f"Timestep={timestep}")
        print(f"Lx={Lx:.10f}, Ly={Ly:.10f}\n")

    data = np.loadtxt(file, skiprows=1)

    ###   Specific confs   ###
    with mp.Pool(1) as pool:
        pool.starmap(main, [(N, Lx, Ly, total_step, output_step, r_neighbor, dt, timestep, sigm_obs, trajectory,
                                    data[(c-1)*N:(c+1)*N], c, list_max, alpha, fx_ext, K, w, d)
                                    for c in cnfs])

    ##   Multiprocessing   ###
    # for k in range(60, 100):
    #
    #     with mp.Pool(n_processes) as pool:
    #         pool.starmap(main, [(N, Lx, Ly, total_step, output_step, r_neighbor, dt, timestep, sigm_obs, trajectory,
    #                             data[(k*10+c)*N:(k*10+c+2)*N], k*10+c, list_max, alpha, fx_ext, K, w)
    #                             for c in range(10)])


    simu_duration = time.time() - simu_start
    print(f"Simulation took {simu_duration / 60:.2f} mins")

