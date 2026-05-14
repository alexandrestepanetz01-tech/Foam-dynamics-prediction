import numpy as np
import os


def write_config_txt(data, N, N_configs, directory, k, l):
    if not os.path.exists(f"{directory}/configs{l}_txt"):
        os.makedirs(f"{directory}/configs{l}_txt")

    for i in range(N_configs):
        headers = data[i * (N + 1)]
        config = data[i * (N + 1) + 1:(i + 1) * (N + 1)]

        # create .txt for calculations (T1, d²min ...)
        with open(f"{directory}/configs{l}_txt/data_{i:04d}.txt", "w") as f:
            for header in headers:
                f.write(f"{header} ")
            f.write("\n")
            for particle in config:
                for k, param in enumerate(particle):
                    if k != 2:
                        f.write(f"{param} ")
                f.write("\n")

    print(f".txt files saved in {directory}/configs{l}_txt")


def write_config_dat(data, N, N_configs, Lx, Ly, directory, k, l):
    if not os.path.exists(f"{directory}/configs{l}_dat"):
        os.makedirs(f"{directory}/configs{l}_dat")
    output_dir = f"{directory}/configs{l}_dat"

    for i in range(N_configs):
        headers = data[i * (N + 1)]
        config = data[i * (N + 1) + 1:(i + 1) * (N + 1)]
        time = headers[0]

        output_file = os.path.join(output_dir, f"run1_mov_data_{i:04d}.dat")

        with open(output_file, "w") as f:
            f.write("ITEM: TIMESTEP\n")
            f.write(f"{time}\n")
            f.write("ITEM: NUMBER OF ATOMS\n")
            f.write(f"{N}\n")
            f.write("ITEM: BOX BOUNDS pp pp pp\n")
            f.write(f"0.0 {Lx}\n")
            f.write(f"0.0 {Ly}\n")
            f.write("0.0 1.0\n")
            f.write("ITEM: ATOMS id x y z c_strs[1]\n")

            for atom_id, particle in enumerate(config, start=1):
                x, y, z, sigma = particle
                f.write(f"{atom_id} {x:.16e} {y:.16e} {z:.16e} {sigma:.16e}\n")

    print(f".dat files saved in {output_dir}")


def main(d, cnfs):
    directory = f"Chaos_MDstep6001_timestep300.0_d{d}"
    for k in cnfs:
        for l in range(11):
        # for l in [10]:
            trajectory_dir = f"configuration_{k}"
            trajectory_file = f"configuration_{k}_trajectory_{l}.txt"

            data = np.loadtxt(os.path.join(directory, trajectory_dir, trajectory_file))

            N = int(data[0][1])
            timestep = int(data[2*(N+1)][0])
            Lx, Ly = data[0][2], data[0][3]
            N_configs = int(data.shape[0] / (N + 1))

            print("Number of particles     : ", N)
            print("Number of configurations: ", N_configs)
            print("Timestep                : ", timestep)
            print("Box size                : ", f"({Lx}, {Ly}) \n")

            write_config_txt(data, N, N_configs, os.path.join(directory, trajectory_dir), k, l)
            write_config_dat(data, N, N_configs, Lx, Ly, os.path.join(directory, trajectory_dir), k, l)
