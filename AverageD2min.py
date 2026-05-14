import os
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable


def plotting_configuration(cnf, y, data_matrix, box_size, vmin, vmax, name, output_dir):

    vx_max = box_size[0]
    vy_max = box_size[1]

    fig, ax = plt.subplots(dpi=600)

    ax.set_xlim(-1, vx_max + 1)
    ax.set_ylim(-1, vy_max + 1)

    j = 45
    sizes = data_matrix[:, 2] * j

    ax.set_aspect(1)

    scatter = ax.scatter(
        data_matrix[:, 0],
        data_matrix[:, 1],
        s=sizes,
        c=y,
        cmap="coolwarm",
        edgecolor="k",
        linewidth=0.05,
        vmin=vmin,
        vmax=vmax
    )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    colorbar = plt.colorbar(scatter, cax=cax)

    colorbar.outline.set_edgecolor("black")
    colorbar.outline.set_linewidth(1)
    colorbar.ax.tick_params(labelsize=12)

    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.tick_params(axis="both", which="both", length=0)

    for spine in ax.spines.values():
        spine.set_edgecolor("black")
        spine.set_linewidth(1)

    os.makedirs(output_dir, exist_ok=True)

    plt.savefig(
        os.path.join(output_dir, f"Simu_{cnf:04d}_{name}.png"),
        bbox_inches="tight"
    )

    plt.close()


def main(d, cnfs):
    directory = f"Chaos_MDstep6001_timestep300.0_d{d}"

    # cnfs = np.arange(1, 1000)
    for k in cnfs:
        print(f"Computing cnf {k}...")

        trajectory_dir = f"configuration_{k}"

        D2min = np.zeros((900, 10))
        for l in range(1, 11):
            filepath = os.path.join(directory, trajectory_dir, f"output_timestep{l}=300.txt")
            with open(filepath) as f:
                headers = f.readline().split()

                timestep = int(float(headers[0]))
                N = int(headers[1])
                Lx = float(headers[2])
                Ly = float(headers[3])

            data = np.loadtxt(filepath, skiprows=1)[:N]
            box_size = [Lx, Ly]
            print("Number of particles     : ", N)
            print("Timestep                : ", timestep)
            print("Box size                : ", f"({Lx}, {Ly})")


            y = data[:, -1]
            y_min = np.min(y[y > 0])
            y = np.where(y < y_min, y_min, y)
            y = np.log10(y)

            D2min[:, l-1] = y

        mean_D2min = np.mean(D2min, axis=1)
        print(mean_D2min.shape)
        new_data = np.zeros_like(data)
        new_data[:, :4] = data[:, :4]
        new_data[:, 4] = mean_D2min

        output_dir = os.path.join(directory, trajectory_dir)
        np.savetxt(output_dir + "Output_averageD2min", new_data)

        vmin = -6
        vmax = 0
        plotting_configuration(k, mean_D2min, data, box_size, vmin, vmax, "D2min", output_dir)
