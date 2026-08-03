import numpy as np
import multiprocessing as mp

from pathlib import Path
import math
import time

"""
Crate from simulation data X_i = (xi, yi, sigma_i, T1_i, D²min_i) a data matrix with structural descriptors used for 
prediction part:

==> X_i = (xi, yi, sigma_i, Lx, Ly, G_i, Psy_i, alpha_i, beta_i, T1_i, D²min_i)
"""

def open_data(name, diff_species):
    with open(f'{name}.txt', 'r') as file:
        first_line = file.readline().strip()
        # extract the num of particle and box size form the header
        n, Lx, Ly = map(float, first_line.split())
        matrix = np.loadtxt(file, dtype=float)
        n = int(n)

    # to differentiate types of particle in the case of polydispersity
    if diff_species:
        third = int(n / 3)

        # loop over the configurations
        for i in range(int(np.shape(matrix)[0] / n)):
            sigma = matrix[i*n:(i+1)*n, 2]
            # define the third party of smallest, medium and biggest particles as species 1, 2 and 3 respectively
            sorted_indices = np.argsort(sigma)
            idx_small = sorted_indices[:third]
            idx_mid = sorted_indices[third: 2 * third]
            idx_large = sorted_indices[2 * third:]
            sigma[idx_small] = 1
            sigma[idx_mid] = 2
            sigma[idx_large] = 3
            # replace the current sizes by the species
            matrix[i*n:(i+1)*n, 2] = sigma

    return matrix, n, Lx, Ly

def NN_Rc(i, c, matrix, R_c, Lx):
    cnf_data = matrix[c*900:(c+1)*900]
    x_i, y_i = cnf_data[i, :2]
    # calculate the distances between particle i and its neighbors
    dx = cnf_data[:, 0] - x_i
    dy = cnf_data[:, 1] - y_i
    # take into account the periodic boundary condition along x
    dx = dx - np.round(dx / Lx) * Lx
    updated_matrix = np.vstack([dx, dy, cnf_data[:, 2]]).T
    # take only particles inside cutoff radius and exclude particle i itself
    dist_squared = dx ** 2 + dy ** 2
    indices = np.where((dist_squared > 0) & (dist_squared <= R_c ** 2))[0]
    NN_matrix = updated_matrix[indices, :3]
    return NN_matrix

def radial_descriptor(c, matrix, R_c, n, Lx):
    G = np.zeros((n, 48))
    l = 0.1

    # loop over all the particles
    for i in range(n):
        # neighbor network
        NN_matrix = NN_Rc(i, c, matrix, R_c, Lx)
        mu_values = np.arange(0.3, 5.1, 0.1)
        G_i = np.zeros((1, np.size(mu_values)))
        m = 0

        # calculate G_i for different values of the parameter
        for mu in mu_values:
            Sum = 0

            # loop over all the neighbors
            for j in range(0, NN_matrix.shape[0]):
                # distance between particle i and j
                r_ij = math.sqrt(NN_matrix[j, 0] ** 2 + NN_matrix[j, 1] ** 2)
                # cutoff function
                fc_ij = 0.5 * (math.cos(np.pi * r_ij / R_c) + 1)
                # sum over all the neighbors j
                Sum += math.exp(-((r_ij - mu) ** 2 / l ** 2)) * fc_ij

            G_i[:, m] = Sum
            m = m + 1

        G[i, :] = G_i
    return G

def angular_descriptor(c, matrix, R_c, n, Lx):
    list_constants = [(14.633, 1, -1), (14.633, 1, 1), (14.638, 2, -1), (14.638, 2, 1), (2.554, 1, -1), (2.554, 1, 1),
                      (2.554, 2, -1), (2.554, 2, 1), (1.648, 1, 1), (1.648, 2, 1), (1.204, 1, 1), (1.204, 2, 1),
                      (1.204, 4, 1), (1.204, 16, 1), (0.933, 1, 1), (0.933, 2, 1), (0.933, 4, 1), (0.933, 16, 1),
                      (0.695, 1, 1), (0.695, 2, 1), (0.695, 4, 1), (0.695, 16, 1)]

    phi = np.zeros((n, 22))

    # loop over all the particles i
    for i in range(0, n):
        phi_i = np.zeros((1, len(list_constants)))
        m = 0

        # calculate phi_i for different values of the parameters xi, zeta and lambda
        for p in range(0, len(list_constants)):
            # neighbor network
            NN = NN_Rc(i, c, matrix, R_c, Lx)
            param_list = list_constants[p]
            xi = param_list[0]
            zeta = param_list[1]
            lamda = param_list[2]
            constant = 2 ** (1 - zeta)
            Sum = 0
            # loop over all the neighbor couples
            for j in range(NN.shape[0]):
                for k in range(NN.shape[0]):
                    if j != k:
                        # distance between particle k and j
                        r_kj = math.sqrt((NN[k, 0] - NN[j, 0]) ** 2 + (NN[k, 1] - NN[j, 1]) ** 2)
                        if r_kj <= R_c:
                            # distance between particle i and j, i and k
                            r_ij = math.sqrt(NN[j, 0] ** 2 + NN[j, 1] ** 2)
                            r_ik = math.sqrt(NN[k, 0] ** 2 + NN[k, 1] ** 2)
                            # calculate the cos of the angle in i of the triangle jik
                            cos_theta = (NN[j, 0] * NN[k, 0] + NN[j, 1] * NN[k, 1]) / (r_ij * r_ik)
                            # cutoff functions
                            fc_ij = 0.5 * (math.cos(np.pi * r_ij / R_c) + 1)
                            fc_ik = 0.5 * (math.cos(np.pi * r_ik / R_c) + 1)
                            fc_jk = 0.5 * (math.cos(np.pi * r_kj / R_c) + 1)
                            # sum over all the neighbors j and k
                            Sum += math.exp(-((r_ij**2 + r_ik**2 + r_kj**2) / xi ** 2)) * (
                                        (1 + lamda * cos_theta) ** zeta) * fc_ij * fc_ik * fc_jk

            phi_i[:, m] = Sum * constant
            m = m + 1

        phi[i, :] = phi_i
    return phi

def structural_descriptors(c, matrix, R_c, n, Lx):
    G = radial_descriptor(c, matrix, R_c, n, Lx)
    Psy = angular_descriptor(c, matrix, R_c, n, Lx)
    return G, Psy

def d_walls_obs(data, Lx, Ly):
    param = [0.5, 1, 2, 5, 10]
    l_vec = [0.5, 1, 2, 5]
    m_vec = [0, 1, 2, 3, 4]
    x = data[:, 0]
    y = data[:, 1]

    # distances between particle i and wall
    r_iw = np.where(y > Ly / 2, Ly - y, y)
    # distances between particle i and obstacle
    r_io = np.sqrt((x - Lx / 2) ** 2 + (y - Ly / 2) ** 2)

    alpha = np.zeros((np.shape(data)[0], len(param)))
    beta = np.zeros_like(alpha)

    theta1 = np.arccos((x - Lx / 2) / r_io)
    theta2 = np.arccos((y - Ly / 2) / r_io)

    gamma = np.zeros((np.shape(data)[0], 36))

    for k, l in enumerate(l_vec):

        for j, m in enumerate(m_vec):
            gamma[:,k*9 + j] = np.cos(m*theta1)*np.exp(-r_io / l)
            if j != 0:
                gamma[:, k*9 +j + 4] = np.cos(m*theta2)*np.exp(-r_io / l)

    # calculation of alpha and beta with varying the parameter l
    for i, l in enumerate(param):
        alpha[:, i] = np.exp(-r_iw / l)
        beta[:, i] = np.exp(-r_io / l)

    return alpha, beta, gamma

def radial_descriptor_diff_species(c, matrix, R_c, n, Lx):
    G = np.zeros((n, 48 * 3))
    l = 0.1

    # loop over all the particles
    for i in range(0, n):
        # neighbor network
        NN_matrix = NN_Rc(i, c, matrix, R_c, Lx)
        mu = np.arange(0.3, 5.1, 0.1)
        G_i = np.zeros((1, mu.shape[0] * 3))
        m = 0

        # loop to take into account species 1, 2 and 3
        for p_type in range(1, 4):
            p = p_type

            # calculate G_i for different values of the parameter
            for k in mu:
                Sum = 0

                # loop over all the neighbors
                for j in range(0, NN_matrix.shape[0]):
                    if NN_matrix[j, 2] == p:
                        # distance between particle i and j
                        r_ij = math.sqrt(NN_matrix[j, 0] ** 2 + NN_matrix[j, 1] ** 2)
                        # cutoff function
                        fc_ij = 0.5 * (math.cos(np.pi * r_ij / R_c) + 1)
                        # sum over all the neighbors j
                        Sum += math.exp(-((r_ij - k) ** 2 / l ** 2)) * fc_ij

                G_i[0, m] = Sum
                m = m + 1

        G[i, :] = G_i
    return G

def angular_descriptor_diff_species(c, matrix, R_c, n, Lx):
    list_constants = [(14.633, 1, -1), (14.633, 1, 1), (14.638, 2, -1), (14.638, 2, 1), (2.554, 1, -1), (2.554, 1, 1),
                      (2.554, 2, -1), (2.554, 2, 1), (1.648, 1, 1), (1.648, 2, 1), (1.204, 1, 1), (1.204, 2, 1),
                      (1.204, 4, 1), (1.204, 16, 1), (0.933, 1, 1), (0.933, 2, 1), (0.933, 4, 1), (0.933, 16, 1),
                      (0.695, 1, 1), (0.695, 2, 1), (0.695, 4, 1), (0.695, 16, 1)]

    phi = np.zeros((n, 22 * 6))

    # loop over all the particles i
    for i in range(0, n):
        NN_matrix_1 = NN_Rc(i, c, matrix, R_c, Lx)
        phi_i = np.zeros((1, 6 * len(list_constants)))
        m = 0

        # loop to take into account species
        for p1_type in range(1, 4):
            p1 = p1_type
            for p2_type in range(p1, 4):
                p2 = p2_type

                # calculate phi_i for different values of the parameters xi, zeta and lambda
                for p in range(0, len(list_constants)):
                    param_list = list_constants[p]
                    xi = param_list[0]
                    zeta = param_list[1]
                    lamda = param_list[2]
                    constant = 2 ** (1 - zeta)
                    Sum = 0
                    NN_matrix_2 = NN_Rc(i, c, matrix, R_c, Lx)

                    # loop over all the neighbor couples
                    for j in range(0, NN_matrix_2.shape[0]):
                        if NN_matrix_1[j, 2] == p1:
                            for k in range(0, NN_matrix_2.shape[0]):
                                if j != k and NN_matrix_1[k, 2] == p2:
                                    # distance between particle k and j
                                    r_kj = math.sqrt((NN_matrix_2[k, 0] - NN_matrix_2[j, 0]) ** 2 + (NN_matrix_2[k, 1] - NN_matrix_2[j, 1]) ** 2)
                                    if r_kj <= R_c:
                                        # distance between particle i and j, i and k
                                        r_ij = math.sqrt(NN_matrix_2[j, 0] ** 2 + NN_matrix_2[j, 1] ** 2)
                                        r_ik = math.sqrt(NN_matrix_2[k, 0] ** 2 + NN_matrix_2[k, 1] ** 2)
                                        # calculate the cos of the angle in i of the triangle jik
                                        cos_theta = (NN_matrix_2[j, 0] * NN_matrix_2[k, 0] + NN_matrix_2[j, 1] * NN_matrix_2[k, 1]) / (r_ij * r_ik)
                                        # cutoff functions
                                        fc_ij = 0.5 * (math.cos(np.pi * r_ij / R_c) + 1)
                                        fc_ik = 0.5 * (math.cos(np.pi * r_ik / R_c) + 1)
                                        fc_jk = 0.5 * (math.cos(np.pi * r_kj / R_c) + 1)
                                        # sum over all the neighbors j and k
                                        Sum += math.exp(-((r_ij**2 + r_ik**2 + r_kj**2) / xi ** 2)) * (
                                                (1 + lamda * cos_theta) ** zeta) * fc_ij * fc_ik * fc_jk

                    phi_i[0, m] = Sum * constant
                    m = m + 1

        phi[i, :] = phi_i
    return phi

def structural_descriptors_diff_species(c, matrix, R_c, n, Lx):
    G = radial_descriptor_diff_species(c, matrix, R_c, n, Lx)
    Psy = angular_descriptor_diff_species(c, matrix, R_c, n, Lx)
    return G, Psy

def calc_descriptors(timestep, diff_species):
    HERE = Path(__file__).resolve().parent
    raw_data = HERE.parent / "raw_data"
    num_config = 1000
    R_c = 5.0

    if diff_species:
        matrix, n, Lx, Ly = open_data(raw_data / f'output_timestep={timestep}', diff_species)

        data = np.zeros((n * num_config, 329))

        # loop over 10 packs of 10 configurations
        for k in range(num_config // 10):

            # parallelization of calculation for 10 configurations on 10 cores
            with mp.Pool(10) as pool:
                results = pool.starmap(structural_descriptors_diff_species,
                                       [(c + 10 * k, matrix, R_c, n, Lx) for c in range(10)])

            # put results in the data matrix
            for c, (G, Psy) in enumerate(results):
                data[(c + 10 * k) * 900:(c + 10 * k + 1) * 900, 5:149] = G
                data[(c + 10 * k) * 900:(c + 10 * k + 1) * 900, 149:281] = Psy

            print(k)

        # fill the data matrix with other features and previous data
        alpha, beta, gamma = d_walls_obs(matrix, Lx, Ly)
        data[:, :3] = matrix[:, :3]
        data[:, 3:5] = Lx, Ly
        data[:, 281:286] = alpha
        data[:, 286:291] = beta
        data[:, 291:327] = gamma
        data[:, 327] = matrix[:, 3]
        data[:, 328] = matrix[:, 4]

        name = f'ML_900x{num_config}x329_t{timestep}_102025'


    else:
        matrix, n, Lx, Ly = open_data(raw_data / f'output_timestep={timestep}', diff_species)

        data = np.zeros((n * num_config, 123))

        # loop over all the configurations
        for k in range(num_config // 10):

            # parallelization of calculation for 10 configurations on 10 cores
            with mp.Pool(10) as pool:
                results = pool.starmap(structural_descriptors, [(c + 10 * k, matrix, R_c, n, Lx) for c in range(10)])

            # put results in the data matrix
            for c, (G, Psy) in enumerate(results):
                data[(c + 10 * k) * 900:(c + 10 * k + 1) * 900, 5:53] = G
                data[(c + 10 * k) * 900:(c + 10 * k + 1) * 900, 53:75] = Psy

            print(k)

        # fill the data matrix with other features and previous data
        alpha, beta, gamma = d_walls_obs(matrix, Lx, Ly)
        data[:, :3] = matrix[:, :3]
        data[:, 3:5] = Lx, Ly
        data[:, 75:80] = alpha
        data[:, 80:85] = beta
        data[:, 85:121] = gamma
        data[:, 121] = matrix[:, 3]
        data[:, 122] = matrix[:, 4]

        name = f'ML_900x{num_config}x123_t{timestep}_102025'

    np.savetxt(f"{name}", data)


start = time.time()

timestep = 30
diff_species = True
for diff_species in [True, False]:
    calc_descriptors(timestep, diff_species)
    end = time.time()
    print('time (s): ', end - start)

