import numpy as np
from pathlib import Path
from tqdm import tqdm


def open_data(name):
    with open(f'{name}', 'r') as file:
        matrix = np.loadtxt(file, dtype=float)
        print(np.shape(matrix))

    return matrix

def NN_Rc(i, matrix, R_c, Lx):
    cnf_data = matrix
    x_i, y_i = cnf_data[i, :2]
    # calculate the distances between particle i and its neighbors
    dx = cnf_data[:, 0] - x_i
    dy = cnf_data[:, 1] - y_i
    # take into account the periodic boundary condition along x
    dx = dx - np.round(dx / Lx) * Lx
    index = np.arange(0, 900)
    updated_matrix = np.vstack([dx, dy, cnf_data[:, 2], index]).T
    # take only particles inside cutoff radius and exclude particle i itself
    dist_squared = dx ** 2 + dy ** 2
    indices = np.where((dist_squared <= R_c ** 2))[0]
    NN_matrix = updated_matrix[indices, :4]

    return NN_matrix

def coarse_grain(matrix, R_c, Lx):
    matrix_original = matrix
    matrix_coarse_grained = matrix_original.copy()
    matrix_coarse_grained[:, 5:281] = 0
    for i in tqdm(range(0, 900), desc="Processing"):
        NN = NN_Rc(i, matrix, R_c, Lx)
        Sum = np.zeros(276)
        Sum1 = 0
        for j in range(len(NN)):
            Sum1 += np.exp(-(np.sqrt(NN[j, 0] ** 2 + NN[j, 1] ** 2) / R_c))
            ind = int(NN[j, 3])
            Sum = Sum + matrix_original[ind, 5:281] * np.exp(-(np.sqrt(NN[j, 0] ** 2 + NN[j, 1] ** 2) / R_c))
        matrix_coarse_grained[i, 5:281] = Sum / Sum1

    return matrix_coarse_grained

for timestep in [30]:
    matrix = open_data(f'ML_900x1000x329_t{timestep}_102025')
    CG_matrix = np.zeros_like(matrix)
    l = [2]
    Lx = 46.7192005534
    Ly = 15.5730668511
    num_config = 1000

    for R_c in l:
        name = f'ML_CG_900x1000x329_t{timestep}_l{R_c}_102025'
        for c in range(num_config):
            CG_matrix[c*900:(c+1)*900] = coarse_grain(matrix[c*900:(c+1)*900], R_c, Lx)
        np.savetxt(f"{name}", CG_matrix)


