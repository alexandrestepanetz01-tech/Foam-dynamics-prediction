import time
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable

from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression


def plotting_configuration(cnf:list, y:np.ndarray, data_matrix:np.ndarray, vmin:float, vmax:float, name:str):

    # define the box limits
    vx_max = data_matrix[0, 3]
    vy_max = data_matrix[0, 4]

    fig, ax = plt.subplots(dpi=600)

    # set box limits
    ax.set_xlim(-1, vx_max + 1)
    ax.set_ylim(-1, vy_max + 1)

    # adjust particle sizes
    j = 45
    sizes = data_matrix[:, 2] * j

    # draw particles
    ax.set_aspect(1)
    scatter = ax.scatter(
        data_matrix[:, 0],
        data_matrix[:, 1],
        s=sizes,
        c=y,
        cmap='coolwarm',
        edgecolor='k',
        linewidth=0.05,
        vmin=vmin,
        vmax=vmax
    )

    # create colorbar with same height as the plot
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="4%", pad=0.05)
    colorbar = plt.colorbar(scatter, cax=cax)

    # colorbar styling
    colorbar.outline.set_edgecolor('black')
    colorbar.outline.set_linewidth(1)
    colorbar.ax.tick_params(labelsize=12)

    # remove tick labels
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)

    # spines styling
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # save figure
    file_name = f'Images/LR/t{timestep}'
    os.makedirs(file_name, exist_ok=True)
    plt.savefig(f'{file_name}/Simu_{cnf}_{name}.png', bbox_inches='tight')
    plt.close()


def regression(data:np.ndarray):
    x_train = data[:800 * 900, 5:327]
    y_train = data[:800 * 900, -1]
    x_test = data[800 * 900:, 5:327]
    y_test = data[800 * 900:, -1]

    # Normalize the input data
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = LinearRegression()
    model.fit(x_train_norm, y_train)

    pearson_vector = np.zeros(200)
    for i in range(200):
        x_test_i = x_test_norm[i * 900:(i+1) * 900]
        y_test_i = y_test[i * 900:(i+1) * 900]

        y_pred_i = model.predict(x_test_i)

        pearson_vector[i] = pearsonr(y_test_i, y_pred_i)[0]

        vmin = 0
        vmax = -6

        # calculate and plot pearson coefficient
        # plt.figure(573 * i)
        # plt.scatter(y_test_i, y_pred_i)
        # plt.xlabel('y_test')
        # plt.ylabel('y_pred')
        # plt.title(f'pearson correlation coefficient: {round(pearson_vector[i], 3)}')
        # plt.savefig(
        #     f'{folder_name}/Pearson_{i}.png',
        #     bbox_inches='tight')
        # plt.close()

        plotting_configuration(i, y_test_i, data[(800 + i) * 900:(800 + i+1) * 900, :5], vmin, vmax,
                               f'ground-truth')
        plotting_configuration(i, y_pred_i, data[(800 + i) * 900:(800 + i+1) * 900, :5], vmin, vmax,
                               f'prediction')

    print(np.mean(pearson_vector))


start = time.time()

HERE = Path(__file__).resolve().parent
processed_data = HERE.parent / "processing_data"

for timestep in [5, 10, 20, 30, 60, 90, 120, 150]:
    data = np.loadtxt(processed_data / "ML_CG_900x1000x329_t30_l2_102025")

    # log transformation of the target
    y = data[:, -1]
    y_min = np.min(y[y != 0])
    logy = np.where(y < y_min, y_min, y)
    logy = np.log10(logy)
    data[:, -1] = logy

    regression(data)

end = time.time()
print('Duration: ', end-start)
