'''Radio towers in a domain.
'''
from argparse import ArgumentParser
import numpy as np
import matplotlib.pyplot as plt


def step(x, y, pos, supp=0.5):
    z = x + 1j*y
    z_abs = abs(z - pos)
    cond = z_abs <= supp
    return np.where(cond, 1, 0)


def alpha(x, y, pos, supp):
    z = x + 1j*y
    z_abs = abs(z - pos)
    cond = z_abs <= supp
    return np.where(cond, np.exp(-1/z_abs), 0)


def gaussian(x, y, pos, supp=0.05):
    z = x + 1j*y
    z_abs = abs(z - pos)
    x = z_abs/supp
    cond = z_abs <= supp
    return np.where(cond, np.exp(-x*x), 0)


def get_holes(x, y, holes, size):
    cond = False
    for _ in range(holes):
        x0, y0 = np.random.choice(x.flat), np.random.choice(y.flat)
        cond = ((abs(x-x0) < size) & (abs(y-y0) < size)) | cond
    return cond


def get_towers(x, y, no_of_towers, hole_info):
    choices = np.vstack(np.where(~hole_info))
    pick = np.random.choice(choices.shape[1], size=no_of_towers)
    tx, ty = choices[:, pick]
    return np.vstack([x[tx, ty], y[tx, ty]])


def make_colored_vec(transmission):
    no_of_towers, nx, ny = transmission.shape
    k = np.zeros([3, nx, ny])
    for i in range(no_of_towers//3):
        for j in range(3):
            k[j] += transmission[i*3 + j]
    # Normalize
    k[:] /= k.max()
    # Make the shape right for plotting
    k = np.transpose(k, (1, 2, 0))
    return k


def plot(towers, k):
    plt.scatter(*towers, s=50, marker='x', c='w')
    plt.imshow(k, extent=[0, 1, 1, 0])
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("Tower transmission on a grid.")
    plt.savefig('transmission_grid.png', dpi=300)
    plt.show()


def compute_transmission(x, y, no_of_towers, supp, holes, hole_width, dx, fname):
    transmission = np.zeros([no_of_towers, *x.shape])
    if fname == 'gaussian':
        func = gaussian
    elif fname == 'step':
        func = step
    elif fname == 'alpha':
        func = alpha

    if holes > 0:
        size = hole_width * dx
        hole_info = get_holes(x, y, holes, size)
        towers = get_towers(x, y, no_of_towers, hole_info)

    # Apply the condition
    for i in range(no_of_towers):
        transmission[i] = func(x, y, np.complex(*towers[:, i]), supp=supp)
    transmission[:, hole_info] = 0.0
    return transmission, towers


def main(n, supp, no_of_towers=12, func='gaussian', seed=111,
         holes=0, hole_width=20):
    dx = 1.0/n
    supp = supp * dx
    assert hole_width < 0.9 * n, "Hole width is larger than 90% domain size."
    np.random.seed(seed)
    _x = np.arange(0, 1, dx)
    _y = np.arange(0, 1, dx/2)
    x, y = np.meshgrid(_x, _y)

    transmission, towers = compute_transmission(x, y, no_of_towers, supp, holes,
                                                hole_width, dx, func)
    k = make_colored_vec(transmission)
    plot(towers, k)


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--n', action='store', type=float, dest='n',
                   default=0.005,
                   help='No. of grid points in the domain.')
    p.add_argument('--towers', action='store', type=int, dest='no_of_towers',
                   default=12, help='Number of towers in the domain.')
    p.add_argument('--holes', action='store', type=int, dest='no_of_holes',
                   default=0, help='Number of holes in the domain.')
    p.add_argument('--hw', action='store', type=float, dest='h_width',
                   default=0, help='Width of each hole.')
    p.add_argument('--supp', action='store', type=float, dest='supp',
                   default=10, help='Tower transmission function support size.')
    p.add_argument('--seed', action='store', type=int, dest='seed',
                   default=2, help='Seed for the random no. generator.')
    p.add_argument(
        "--function", action="store", type=str, dest='func',
        default='gaussian',
        choices=['gaussian', 'step', 'alpha'],
        help="Choice of function for the towers."
    )
    p.add_argument(
        '--plot', action='store_true', dest='plot',
        default=False, help='Show plots at the end of simulation.'
    )
    o = p.parse_args()
    main(n=o.n, supp=o.supp, no_of_towers=o.no_of_towers,
         seed=o.seed, holes=o.no_of_holes, hole_width=o.h_width, func=o.func)
