from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter
import numpy as np

__init__ = ["plot3D"]

# obtained from: https://matplotlib.org/mpl_toolkits/mplot3d/tutorial.html
def plot3D(x, y, z, z_lim):

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    surf = ax.plot_surface(x, y, z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, z_lim)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

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