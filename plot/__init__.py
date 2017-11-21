from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

__init__ = ["plot3D"]

# obtained from: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
def plot3D(x, y, z, z_lim, labels):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, z_lim)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    ax.set_xlabel(labels[0])
    ax.set_ylabel(labels[1])
    ax.set_zlabel(labels[2])

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()




def plot_multiple_curves(curves, points, labels):
    fig = plt.figure()
    plots = []
    print(labels)
    for i in range(len(curves)):
        plots.append(plt.plot(curves[i][0], curves[i][1], label=labels[i])[0])
    plt.legend(handles=plots)
    plt.scatter(points[0], points[1], s=0.5)
    plt.show()


def data_statistics(data):
    d = data.T
    dim = len(d)
    m1 = np.zeros(dim)
    m2 = np.zeros(dim)
    for i in range(dim):
        m1[i] = np.min(d[i])
        m2[i] = np.max(d[i])
    return (m1, m2)