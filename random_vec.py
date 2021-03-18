'''Random vector drawn from unit sphere in `d` dimensions.
'''
from argparse import ArgumentParser
from time import time
import numpy as np
from numpy import cos, sin
from numpy.random import default_rng
import matplotlib.pyplot as plt
import pandas as pd
from mpl_toolkits.mplot3d import Axes3D
import seaborn as sns
from matplotlib import rcParams


rcParams.update({'figure.autolayout': True})


def generate_samples(n_samples, dim, rng=default_rng(), sphere=True):
    samples = rng.standard_normal([n_samples, dim])
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    if not sphere:
        samples *= pow(rng.random_sample(dim), 1.0/dim)
    return samples


def compute_norm(vec):
    return np.linalg.norm(vec)


def compute_angle(v1, v2):
    return np.arccos(np.round(np.dot(v1, v2), 8))


def angle_dim(n_samples, dim):
    start = time()
    samples = generate_samples(n_samples, dim)
    print(f"Time taken to draw the samples: {time() - start:.4}s.")
    ang = np.zeros((n_samples, n_samples))
    for i in range(n_samples):
        xi = samples[i]
        for j in range(n_samples):
            xj = samples[j]
            ang[i, j] = compute_angle(xi, xj)
    return samples, ang


def moore_penrose_inverse(A):
    return np.linalg.inv(np.dot(A.T, A))@A.T


def projection(A, vec):
    return A@np.linalg.inv(np.dot(A.T, A))@A.T@vec


def gram_schmidt(vecs):
    Q, R = np.linalg.qr(vecs)
    return Q, R


def uniform_dist(samples=20):
    return 2 * np.random.rand(samples, 2) - 1


def get_ortho2(dim, subspace_dim, nsamples):
    A = np.zeros((nsamples, subspace_dim, dim))
    msg = "Subspace dimension is greater than dimension \
    of space it is contained in."
    assert subspace_dim <= dim, msg

    # Generate random samples in a space of dimension=dim.
    for i in range(nsamples):
        samples = generate_samples(dim, dim, rng=default_rng(), sphere=True)
        Q, R = gram_schmidt(samples)
        D = np.diag(R)
        D = D/np.abs(D)
        ph = np.diag(D)
        Q = Q@ph@Q
        print(np.linalg.det(Q))
        A[i, :, :] = Q[:subspace_dim]
    # std_basis = np.zeros((dim, subspace_dim))
    # std_basis1 = np.zeros((dim, subspace_dim))
    # std_basis2 = np.zeros((dim, subspace_dim))
    # for i in range(subspace_dim):
    #     std_basis[i, i] = 1.0
    # std_basis1[0, 1] = 1.0
    # std_basis1[1, 0] = 1.0
    # std_basis1[2, 2] = -1.0
    # std_basis2[0, 2] = 1.0
    # std_basis2[1, 1] = 1.0
    # std_basis2[2, 0] = 1.0
    # A[0, :, :] = std_basis.T
    # A[1, :, :] = std_basis1.T
    # A[2, :, :] = std_basis2.T
    return A


def get_ortho(dim, subspace_dim, samples):
    from scipy.stats import special_ortho_group as SO
    A = np.zeros((samples, subspace_dim, dim))
    for i in range(samples):
        Q = SO.rvs(dim)
        print(np.linalg.det(Q))
        A[i, :, :] = Q[:subspace_dim]
    # std_basis = np.zeros((dim, subspace_dim))
    # std_basis1 = np.zeros((dim, subspace_dim))
    # std_basis2 = np.zeros((dim, subspace_dim))
    # for i in range(subspace_dim):
    #     std_basis[i, i] = 1.0
    # std_basis1[0, 1] = 1.0
    # std_basis1[1, 0] = 1.0
    # std_basis1[2, 2] = -1.0
    # std_basis2[0, 1] = 1.0
    # std_basis2[1, 0] = 1.0
    # std_basis2[2, 2] = 1.0
    # A[0, :, :] = std_basis.T
    # A[1, :, :] = std_basis1.T
    # A[2, :, :] = std_basis2.T
    return A


def random_function(x, y, p, q):
    norm2 = (x - p)**2 + (y - q)**2
    return np.exp(-4*norm2)


def map_to_higher_dim(x, y, m):
    f = np.zeros((len(x), len(m)))
    for i, (xi, yi) in enumerate(zip(x, y)):
        f[i] = random_function(xi, yi, m[:, 0], m[:, 1])
    return f


def initialize_square(nx, holes=False):
    side = 0.5
    dx = 5/nx
    y, x = np.mgrid[0:side:nx*1j, 0:side:nx*1j]
    if holes:
        cond = (((x-0.5*side)**2 + (y-0.5*side)**2 < dx**2) |
                ((x-0.1*side)**2 + (y-0.1*side)**2 < dx**2) |
                ((x-0.1*side)**2 + (y-0.9*side)**2 < dx**2) |
                ((x-0.9*side)**2 + (y-0.1*side)**2 < dx**2) |
                ((x-0.9*side)**2 + (y-0.9*side)**2 < dx**2))
        x, y = x[~cond], y[~cond]
    x, y = x.ravel(), y.ravel()
    return x, y, 'square'


def initialize_eight(nx):
    rad = 1.0
    theta = np.linspace(0, 2*np.pi, nx, endpoint=False)
    x, y = rad * sin(theta), rad * sin(theta) * cos(theta)
    x, y = x.ravel(), y.ravel()
    return x, y, 'eight'


def initialize_circle(nx):
    rad = 1.0
    theta = np.linspace(0, 2*np.pi, nx, endpoint=False)
    x, y = rad * cos(theta), rad * sin(theta)
    x, y = x.ravel(), y.ravel()
    return x, y, 'circle'


def initialize_rose(nx, k=1):
    # rad = 0.1
    rad = 0.1
    theta = np.linspace(0, 10*np.pi, nx, endpoint=False)
    r = rad * cos(k*theta)
    x, y = r * cos(theta), r * sin(theta)
    x, y = x.ravel(), y.ravel()
    return x, y, f'{k}-rose'


def main():
    base_dim = 3
    dim = 20
    n_subspaces = 4
    x, y, name = initialize_rose(nx=1000, k=2/5)
    # x, y, name = initialize_square(nx=100, holes=True)
    m = uniform_dist(dim)
    F = map_to_higher_dim(x, y, m)
    A = get_ortho2(dim, base_dim, n_subspaces)
    pt = np.zeros((n_subspaces, len(x), base_dim))
    std_basis = np.zeros((dim, base_dim))
    for i in range(base_dim):
        std_basis[i, i] = 1.0
    for i in range(n_subspaces):
        a = A[i].T
        for j, f in enumerate(F):
            proj = projection(a, f)
            pt[i, j, :] = a.T@proj

    subspace = np.arange(len(x)*n_subspaces, dtype=int)//len(x)
    data = np.c_[pt.reshape(-1, pt.shape[-1]), subspace]
    df = pd.DataFrame(data, columns=['x-axis', 'y-axis', 'z-axis', 'subspace'])

    sns.relplot(data=df, x='x-axis', y='y-axis', col='subspace', col_wrap=4,
                height=3, hue='z-axis', palette='viridis_r',
                kind='scatter', size='z-axis', linewidth=0, sizes=(2, 2),
                facet_kws={'sharey': False, 'sharex': False})
    plt.savefig(name + '-xy.png', bbox_inches='tight')

    sns.relplot(data=df, x='x-axis', y='z-axis', col='subspace', col_wrap=4,
                height=3, hue='y-axis', palette='viridis_r',
                kind='scatter', size='y-axis', linewidth=0, sizes=(2, 2),
                facet_kws={'sharey': False, 'sharex': False})
    plt.savefig(name + '-xz.png', bbox_inches='tight')

    # fig = plt.figure(figsize=plt.figaspect(0.5))
    # for i in range(n_subspaces):
    #     ax = fig.add_subplot(2, 2, i+1, projection='3d')
    #     ax.scatter(pt[i, :, 0], pt[i, :, 1], pt[i, :, 2], s=1)
    # plt.savefig(name + '.png')
    # plt.show()

    # from mayavi import mlab
    # for i in range(n_subspaces):
    #     pts = pt[i].T
    #     pts[0] += i
    #     mlab.points3d(*pts, scale_factor=0.001)
    # mlab.show()


if __name__ == '__main__':
    parse = ArgumentParser()
    parse.add_argument('--dim', action='store', type=int, dest='dim',
                   default=1000,
                   help='Dimension of the space on which the vector is \
                   sampled. Default dimension is 1000.')
    parse.add_argument('--size', action='store', type=int, dest='n_samples',
                   default=12, help='No of samples to be drawn.')
    parse.add_argument('--subspace-dim', action='store', type=int,
                   dest='subspace_dim', default=100,
                   help='Dimension of the subspace of the original space \
                   (default: 100).')
    parse.add_argument('--bins', action='store', type=int, dest='bins',
                   default=12, help='No. of bins of the histogram.')
    parse.add_argument(
        '--plot', action='store_true', dest='plot',
        default=True, help='Show plots at the end of simulation.'
    )
    o = parse.parse_args()
    vecs, angles = angle_dim(o.n_samples, o.dim)

    if o.plot:
        # plt.hist(angles.ravel(), bins=o.bins)
        plt.show()
    main()
