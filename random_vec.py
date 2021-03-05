'''Random vector drawn from unit sphere in `d` dimensions.
'''
from argparse import ArgumentParser
from time import time
import numpy as np
from numpy.random import default_rng
import matplotlib.pyplot as plt


def generate_samples1(no_of_samples, dim, rng=default_rng()):
    x = np.zeros([no_of_samples, dim])
    mean = np.zeros(dim)
    cov = np.identity(dim)
    for i in range(no_of_samples):
        sample = rng.multivariate_normal(mean, cov)
        x[i][:] = sample
        norm[i] = np.linalg.norm(sample)
    return sample


def generate_samples(no_of_samples, dim, rng=default_rng(), sphere=True):
    samples = rng.standard_normal([no_of_samples, dim])
    samples /= np.linalg.norm(samples, axis=1, keepdims=True)
    if not sphere:
        samples *= pow(rng.random_sample(dim), 1.0/dim)
    return samples


def compute_norm(vec):
    return np.linalg.norm(vec)


def compute_angle(v1, v2):
    return np.arccos(np.round(np.dot(v1, v2), 8))


def angle_dim(no_of_samples, dim):
    start = time()
    samples = generate_samples(no_of_samples, dim)
    print(f"Time taken to draw the samples: {time() - start:.4}s.")
    ang = np.zeros((no_of_samples, no_of_samples))
    for i in range(no_of_samples):
        xi = samples[i]
        for j in range(no_of_samples):
            xj = samples[j]
            ang[i, j] = compute_angle(xi, xj)
    return samples, ang


def independent_vec(samples, angles, tol=1e-5):
    i, j = np.where(np.abs(angles - np.pi*0.5) < tol)
    idx = np.column_stack((i, j))
    sort_idx = np.sort(idx, axis=1)
    unique_idx = np.unique(sort_idx, axis=0)


def projection(A):
    return A@np.linalg.inv(np.dot(A.T, A))@A.T


def construct_projection(vecs):
    return np.array(vecs)


def main(dim, subspace_dim):
    pass


if __name__ == '__main__':
    p = ArgumentParser()
    p.add_argument('--dim', action='store', type=int, dest='dim',
                   default=1000,
                   help='Dimension of the space on which the vector is \
                   sampled. Default dimension is 1000.')
    p.add_argument('--size', action='store', type=int, dest='no_of_samples',
                   default=12, help='No of samples to be drawn.')
    p.add_argument('--subspace-dim', action='store', type=int,
                   dest='subspace_dim', default=100,
                   help='Dimension of the subspace of the original space \
                   (default: 100).')
    p.add_argument('--bins', action='store', type=int, dest='bins',
                   default=12, help='No. of bins of the histogram.')
    p.add_argument(
        '--plot', action='store_true', dest='plot',
        default=True, help='Show plots at the end of simulation.'
    )
    o = p.parse_args()
    samples, angles = angle_dim(o.no_of_samples, o.dim)

    if o.plot:
        plt.hist(angles.ravel(), bins=o.bins)
        plt.xlim(1.45, 1.65)
        plt.show()
