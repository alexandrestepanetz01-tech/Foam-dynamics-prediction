import time
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import f1_score, accuracy_score, recall_score, precision_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.utils.class_weight import compute_class_weight


def open_file(name):
    # open the data file as pandas table
    with open(f'{name}', 'r') as file:
        data = np.loadtxt(file, dtype=float)
    return data


def weights():
    for i in [5, 10, 20, 30, 60, 90, 120, 150]:
        data = np.loadtxt(f'output_timestep={i}.txt', skiprows=1)
        y_train = data[:, -2]

        classes = np.unique(y_train)

        weights = compute_class_weight(
            class_weight="balanced",
            classes=classes,
            y=y_train
        )
        print(f'timestep {i}:')
        print(dict(zip(classes, weights)))


def plotting_configuration(cnf, y, data_matrix, vmin, vmax, name):

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

    # remove tick labels
    ax.xaxis.set_tick_params(labelbottom=False)
    ax.yaxis.set_tick_params(labelleft=False)
    ax.tick_params(axis='both', which='both', length=0)

    # spines styling
    for spine in ax.spines.values():
        spine.set_edgecolor('black')
        spine.set_linewidth(1)

    # save figure
    file_name = f'Images/LgR/t{timestep}'
    os.makedirs(file_name, exist_ok=True)
    plt.savefig(f'{file_name}/Simu_{cnf}_{name}.png', bbox_inches='tight')
    plt.close()


def regression(data):
    x_train = data[:800 * 900, 5:327]
    y_train = data[:800 * 900, -2]
    x_test = data[800 * 900:, 5:327]
    y_test = data[800 * 900:, -2]

    # Normalize the input data
    scaler = StandardScaler()
    x_train_norm = scaler.fit_transform(x_train)
    x_test_norm = scaler.transform(x_test)

    model = LogisticRegression(class_weight={0:1, 1:6}, C=np.inf, max_iter=5000)
    model.fit(x_train_norm, y_train)

    f1_vector = np.zeros(200)
    accuracy_vector = np.zeros(200)
    recall_vector = np.zeros(200)
    precision_vector = np.zeros(200)

    for i in range(200):
    # for i in [62, 175, 192]:
        x_test_i = x_test_norm[i * 900:(i+1) * 900]
        y_test_i = y_test[i * 900:(i+1) * 900]

        y_pred_i = model.predict(x_test_i)
        prob_i = model.predict_proba(x_test_i)[:, 1]

        f1_vector[i] = f1_score(y_test_i, y_pred_i, zero_division=0)
        accuracy_vector[i] = accuracy_score(y_test_i, y_pred_i)
        recall_vector[i] = recall_score(y_test_i, y_pred_i, zero_division=0)
        precision_vector[i] = precision_score(y_test_i, y_pred_i, zero_division=0)

        vmin = min(np.min(y_test), np.min(y_pred_i))
        vmax = max(np.max(y_test), np.max(y_pred_i))

        plotting_configuration(i, y_test_i, data[(800 + i) * 900:(800 + i+1) * 900, :5], vmin, vmax,
                               'ground-truth')
        plotting_configuration(i, y_pred_i, data[(800 + i) * 900:(800 + i+1) * 900, :5], vmin, vmax,
                               "prediction")
        plotting_configuration(i, prob_i, data[(800 + i) * 900:(800 + i+1) * 900, :5], vmin, vmax,
                               "prob prediction")

    print('F1: ', np.mean(f1_vector))
    # print('Accuracy: ', np.mean(accuracy_vector))
    # print('Recall: ', np.mean(recall_vector))
    # print('Precision: ', np.mean(precision_vector))


start = time.time()


HERE = Path(__file__).resolve().parent
processed_data = HERE.parent / "processing_data"

for timestep in [5, 10, 20, 30, 60, 90, 120, 150]:
    data = open_file(processed_data / 'ML_CG_900x1000x329_t30_l2_102025')

    regression(data)


end = time.time()
print('time 2: ', end-start)
