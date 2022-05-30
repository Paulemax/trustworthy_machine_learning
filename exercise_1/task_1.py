import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm


def plot_function(X: np.ndarray, f):
    # 
    X_1, X_2 = np.meshgrid(X[0], X[1])
    Y: np.ndarray = f(X_1, X_2)

    fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
    surf = ax.plot_surface(X_1, X_2, Y, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    ax.set_xlabel("sensitivity")
    ax.set_ylabel("prevelance")

    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()


def task_one():
    X = np.vstack((np.arange(0, 1, 0.01), np.arange(0, 1, 0.01)))

    spes = np.arange(0, 1, 0.1)
    print(spes)
    for spe in spes:
        ppv = lambda x, y: (x * y) / (x * y + (1 - spe) * (1 - y))
        npv = lambda sen, pre: (sen * (1- pre)) / (sen * (1 - pre) + (1 - spe) * pre)

        plot_function(X, ppv)
        plot_function(X, npv)


if __name__ == '__main__':
    task_one() 