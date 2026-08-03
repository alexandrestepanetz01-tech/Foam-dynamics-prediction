import matplotlib.pyplot as plt
import numpy as np

def plot_LR_perf():
    '''
    Plot the linear and neural network regression performance curves.
    '''
    data = np.loadtxt("LR_perf")[:, :7]

    t = data[0]
    D2min = data[1]
    log_D2min = data[2]
    three_comp = data[3]
    obs_wall = data[4]
    CG = data[5]
    NN = data[6]

    plt.figure(1, (10, 8))

    l1, = plt.plot(t, D2min,
                   color='purple',
                   label="D²min",
                   marker='s',
                   markersize=10)

    l2, = plt.plot(t, log_D2min,
                   color='green',
                   label="log(D²min)",
                   marker='*',
                   markersize=10)

    l3, = plt.plot(t, three_comp,
                   color='blue',
                   label="Three component descriptor",
                   marker='^',
                   markersize=10)

    l4, = plt.plot(t, obs_wall,
                   color='black',
                   label="Obstacle/wall descriptor",
                   marker='o',
                   markersize=10)

    l5, = plt.plot(t, CG,
                   color='yellow',
                   label="Coarse-grained descriptor",
                   marker='x',
                   markersize=10)

    l6, = plt.plot(t, NN,
                   color='red',
                   label="Neural network",
                   marker='d',
                   markersize=10)


    plt.xlabel(r'$\Delta t$', fontsize=25)
    plt.ylabel('Pearson coefficient', fontsize=25)
    plt.xlim(0, 1220)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    ax = plt.gca()

    # Légende du milieu droit
    leg1 = ax.legend(
        handles=[l1, l2],
        loc='lower right',
        bbox_to_anchor=(0.35, 0),
        fontsize=20
    )

    # Garder cette légende
    ax.add_artist(leg1)

    # Deuxième légende
    ax.legend(
        handles=[l3, l4, l5, l6],
        loc='lower right',
        fontsize=20
    )

    plt.grid()
    plt.show()


def plot_LgR_perf():
    '''
    Plot the logistic regression performance curves.
    '''
    data = np.loadtxt("LGR_perf")[:, :7]

    t = data[0]
    T1 = data[1]
    three_comp = data[2]
    obs_wall = data[3]
    CG = data[4]

    plt.figure(2, (10, 8))

    plt.plot(t, T1, color='purple', label="One component descriptor", marker='s', markersize=10)
    plt.plot(t, three_comp, color='blue', label="Three component descriptor", marker='^', markersize=10)
    plt.plot(t, obs_wall, color='black', label="Obstacle/wall descriptor", marker='o', markersize=10)
    plt.plot(t, CG, color='green', label="Coarse-grained descriptor", marker='x', markersize=10)

    plt.xlabel(r'$\Delta t$', fontsize=25)
    plt.ylabel('F1 score', fontsize=25)
    plt.xlim(0, 1220)
    plt.ylim(0, 1)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='upper left', fontsize=20)
    plt.grid()
    plt.show()


def plot_LC(file):
    '''
    Plot the learning curves corresponding to the file given.
    :param file: string, the file name with corresponding data
    '''

    data = np.loadtxt(file)

    if file[:3] == 'Log': score_name = "F1 score"
    else: score_name = 'Pearson coefficient'

    x = data[:, 0]
    train = data[:, 1]
    validation = data[:, 2]

    plt.figure(10, (10, 8))

    plt.plot(x, train, color='green', label='train')
    plt.plot(x, validation, color='blue', label='validation')

    plt.xlim(0, 750)
    plt.ylim(0, 1)

    plt.xlabel('Number of configurations', fontsize=30)
    plt.ylabel(score_name, fontsize=30)
    plt.xticks(fontsize=25)
    plt.yticks(fontsize=25)

    plt.legend(loc='lower right', fontsize=25)
    plt.grid()
    plt.show()


def plot_logistic_scores():
    '''
    Plot the logistic regression performances (accuracy, precision, recall, f1 score) with respect to the delta t.
    '''

    data = np.loadtxt('Logistic_scores')[:, :7]

    x = data[0]
    accuracy = data[1]
    precision = data[2]
    recall = data[3]
    f1 = data[4]

    plt.figure(10, (10, 8))

    plt.plot(x, accuracy, color='green', label='Accuracy', marker='s', markersize=10)
    plt.plot(x, precision, color='blue', label='Precision', marker='^', markersize=10)
    plt.plot(x, recall, color='black', label='Recall', marker='o', markersize=10)
    plt.plot(x, f1, color='red', label='F1', marker='x', markersize=10)

    plt.xlim(0, 1220)
    plt.ylim(0, 1)

    plt.xlabel(r'$\Delta t$', fontsize=25)
    plt.ylabel('Performance metrics', fontsize=25)
    plt.xticks(fontsize=20)
    plt.yticks(fontsize=20)

    plt.legend(loc='lower right', fontsize=20)
    plt.grid()
    plt.show()



plot_LR_perf()
plot_LgR_perf()
plot_LC('Linear_regression_LC')
plot_LC('Logistic_regression_LC')
plot_LC('Neural_network_LC')
plot_logistic_scores()
