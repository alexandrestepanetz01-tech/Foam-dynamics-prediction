from os import times

import numpy as np
from scipy.signal import find_peaks
import os
import time


def compute_rdf(data, N, box_size, dr=0.1, r_max=None):
    positions = data[:, :2]
    Lx, Ly = box_size[0], box_size[1]

    if r_max is None:
        r_max = np.min((Lx, Ly)) / 2.0

    nbins = int(r_max / dr)
    r = np.linspace(dr, r_max, nbins)
    g_r = np.zeros(nbins)
    norm = np.pi * ((r + dr) ** 2 - r ** 2) * (N / (Lx * Ly))

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < r_max:
                bin_idx = int(dist / dr)
                if bin_idx < nbins:
                    g_r[bin_idx] += 2
    g_r /= (N * norm)
    return r, g_r


def find_first_minimum(r, g_r, smoothing_window=5):
    g_r_smooth = np.convolve(g_r, np.ones(smoothing_window) / smoothing_window, mode='same')
    peaks, _ = find_peaks(g_r_smooth)
    if len(peaks) > 0:
        first_peak = peaks[0]
        for i in range(first_peak, len(g_r_smooth) - 1):
            if g_r_smooth[i] < g_r_smooth[i - 1] and g_r_smooth[i] < g_r_smooth[i + 1]:
                return r[i]
    return r[len(r) // 2]


def compute_neighbors(X, Lx, Ly, cutoff):
    N = len(X)
    neighbors = {i: set() for i in range(N)}
    for i in range(N):
        for j in range(i + 1, N):
            dx = X[i, 0] - X[j, 0]
            dy = X[i, 1] - X[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist = np.sqrt(dx ** 2 + dy ** 2)
            if dist < cutoff:
                neighbors[i].add(j)
                neighbors[j].add(i)
    return neighbors


def compute_dmin2(X0, X1, Lx, Ly, cutoff=1.2):
    N = X0.shape[0]
    dmin2_values = np.full(N, np.nan)
    neighbors = compute_neighbors(X0, Lx, Ly, cutoff)

    for i in range(N):
        neighbors_idx = list(neighbors[i])
        if len(neighbors_idx) < 3:
            dmin2_values[i] = 0.0
            continue

        R = X0[neighbors_idx, :2] - X0[i, :2]
        R[:, 0] -= Lx * np.round(R[:, 0] / Lx)
        R[:, 1] -= Ly * np.round(R[:, 1] / Ly)

        R_prime = X1[neighbors_idx, :2] - X1[i, :2]
        R_prime[:, 0] -= Lx * np.round(R_prime[:, 0] / Lx)
        R_prime[:, 1] -= Ly * np.round(R_prime[:, 1] / Ly)

        R_T = R.T
        R_init_R = R_T @ R
        if np.linalg.det(R_init_R) < 1e-10:
            dmin2_values[i] = 0.0
            continue

        inv_R_init_R = np.linalg.inv(R_init_R)
        J = inv_R_init_R @ (R_T @ R_prime)
        residuals = R_prime - R @ J
        dmin2_values[i] = np.mean(np.sum(residuals ** 2, axis=1))

    return dmin2_values


def compute_neighbor_lists(data, N, box_size, r_min=None, r_max=None):
    positions = data[:, :2]
    Lx, Ly = box_size

    if r_min is None or r_max is None:
        r, g_r = compute_rdf(data, N, box_size)
        r_min = 1.2
        r_max = 1.4

    neighbors_rmin = {i: set() for i in range(N)}
    neighbors_rmax = {i: set() for i in range(N)}

    for i in range(N):
        for j in range(i + 1, N):
            dx = positions[i, 0] - positions[j, 0]
            dy = positions[i, 1] - positions[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist = np.sqrt(dx ** 2 + dy ** 2)

            if dist < r_max:
                neighbors_rmax[i].add(j)
                neighbors_rmax[j].add(i)
                if dist < r_min:
                    neighbors_rmin[i].add(j)
                    neighbors_rmin[j].add(i)
    return neighbors_rmin, neighbors_rmax


def detect_timestep_events(neighbors_init_rmin, neighbors_init_rmax,
                     neighbors_timestep_rmin, neighbors_timestep_rmax,
                     data_t, data_t1, box_size, cutoffs, threshold=1.0):
    r_min, r_max = cutoffs[0], cutoffs[1]
    t1_events = []
    t1_particles = set()
    Lx, Ly = box_size
    N = len(data_t)
    dist_t = np.zeros((N, N))
    dist_t1 = np.zeros((N, N))
    for i in range(N):
        for j in range(i + 1, N):
            dx = data_t[i, 0] - data_t[j, 0]
            dy = data_t[i, 1] - data_t[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist_t[i, j] = dist_t[j, i] = np.sqrt(dx ** 2 + dy ** 2)
            dx = data_t1[i, 0] - data_t1[j, 0]
            dy = data_t1[i, 1] - data_t1[j, 1]
            dx -= Lx * np.round(dx / Lx)
            dy -= Ly * np.round(dy / Ly)
            dist_t1[i, j] = dist_t1[j, i] = np.sqrt(dx ** 2 + dy ** 2)
    for i in range(N):
        displacement = np.abs(data_t[i, :2] - data_t1[i, :2])
        displacement = np.minimum(displacement, box_size - displacement)
        if np.linalg.norm(displacement) > threshold:
            continue
        broken = set()
        for j in neighbors_init_rmin[i]:
            if dist_t1[i, j] > r_max:
                broken.add(j)
        formed = set()
        for j in neighbors_timestep_rmin[i]:
            if j not in neighbors_init_rmax[i] or dist_t[i, j] > r_max:
                formed.add(j)
        if broken or formed:
            t1_events.append((i, broken, formed))
            t1_particles.update([i] + list(broken) + list(formed))
    return t1_events, t1_particles


def main(d, cnfs):
    directory = f"Chaos_MDstep6001_timestep300.0_d{d}"

    for k in cnfs:

        print(f"Computing cnf {k}...")
        start = time.time()

        trajectory_dir = f"configuration_{k}"
        for l in range(11):
        # for l in [11]:
            data_dir = os.path.join(directory, trajectory_dir, f"configs{l}_txt")

            data = np.loadtxt(os.path.join(directory, trajectory_dir, f'configuration_{k}_trajectory_0.txt'))

            N = int(data[0][1])
            timestep = int(data[2*(N+1)][0])
            Lx, Ly = data[0][2], data[0][3]
            box_size = [Lx, Ly]
            N_configs = int(data.shape[0] / (N + 1))

            print("Number of particles     : ", N)
            print("Number of configurations: ", N_configs)
            print("Timestep                : ", timestep)
            print("Box size                : ", f"({Lx}, {Ly})")

            output_file = os.path.join(directory, trajectory_dir, f"output_timestep{l}={timestep}.txt")
            if os.path.exists(output_file):
                os.remove(output_file)

            with open(output_file, 'w') as f:
                f.write(f"{timestep:.6f}\t{N}\t{Lx:.10f}\t{Ly:.10f}\n")

            for file_num in range(N_configs - 2):

                file_init_config = os.path.join(data_dir, f"data_{file_num + file_num**2:04d}.txt")
                file_timestep_config = os.path.join(data_dir, f"data_{file_num + 2:04d}.txt")

                try:
                    data_init_config = np.loadtxt(file_init_config, skiprows=1)
                    data_timestep_config = np.loadtxt(file_timestep_config, skiprows=1)
                except:
                    print(f"Could not load files {file_init_config} or {file_timestep_config}, skipping...")
                    continue

                print(f"Processing files {file_num} and {file_num + 1}...")

                r, g_r = compute_rdf(data_init_config, N, box_size, dr=0.1, r_max=None)
                first_min = find_first_minimum(r, g_r)
                r_min, r_max = first_min - 0.1, first_min + 0.1
                cutoffs = [r_min, r_max]
                # print(f"r_min = {r_min}   and r_max = {r_max}")

                neighbors_init_rmin, neighbors_init_rmax = compute_neighbor_lists(data_init_config, N, box_size, r_min, r_max)
                neighbors_timestep_rmin, neighbors_timestep_rmax = compute_neighbor_lists(data_timestep_config, N, box_size, r_min, r_max)

                t1_events, t1_particles = detect_timestep_events(
                    neighbors_init_rmin, neighbors_init_rmax,
                    neighbors_timestep_rmin, neighbors_timestep_rmax,
                    data_init_config, data_timestep_config, box_size, cutoffs
                )

                dmin2_values = compute_dmin2(data_init_config[:, :2], data_timestep_config[:, :2], Lx, Ly, cutoffs[1])

                T1_flag_array = np.zeros(len(data_init_config), dtype=int)
                for i in t1_particles:
                    T1_flag_array[i] = 1

                output_data = np.column_stack((data_init_config[:, 0], data_init_config[:, 1], data_init_config[:, 2], T1_flag_array, dmin2_values))

                with open(output_file, 'a') as f:
                    np.savetxt(f, output_data, fmt="%.6f")

            duration = time.time() - start
            print(f"Cnf {k} done in {duration:.2f} s")
