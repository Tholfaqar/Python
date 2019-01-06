import numpy as np
import scipy.sparse as sps
from scipy.spatial import Delaunay
import matplotlib.pyplot as plt
import matplotlib.tri as tri 
from scipy.sparse.linalg import spsolve


def all_edges(t,p):
    # Find all unique edges in the triangulation t (nt-by-3 array)
    # Second output is indices to the boundary edges.
    edges = np.vstack((t[:,[0,1]], t[:,[1,2]], t[:,[2,0]]))
    key = edges.min(axis=1) + edges.max(axis=1) * p.shape[0]
    _, ix, counts = np.unique(key, return_index=True, return_counts=True)
    return edges[ix, :], np.where(counts == 1)

def all_edges_for_p2(t2, p2):
    # Find all unique edges in the triangulation t (nt-by-6 array)
    # Second output is indices to the boundary edges.
    edges = np.vstack([
        t2[:, [0, 3]],
        t2[:, [3, 1]],
        t2[:, [1, 4]],
        t2[:, [4, 2]],
        t2[:, [2, 5]],
        t2[:, [5, 0]],
    ])
    key = edges.min(axis=1) + edges.max(axis=1) * p2.shape[0]
    _, ix, counts = np.unique(key, return_index=True, return_counts=True)
    return edges[ix, :], np.where(counts == 1)

def boundary_nodes(t,p):
    # Find all boundary nodes in the triangulation t
    edges, boundary_indices = all_edges(t,p)
    return np.unique(edges[boundary_indices, :].flatten())

def boundary_nodes_for_p2(t2, p2):
    # Find all boundary nodes in the triangulation t
    edges, boundary_indices = all_edges_for_p2(t2, p2)
    return np.unique(edges[boundary_indices, :].flatten())

def boundary_edges(t,p):
    # Find all boundary edges in the triangulation t
    edges, boundary_indices = all_edges(t,p)
    return edges[boundary_indices, :]

def tplot(p, t, u=None):
    # Plot triangular mesh p, t, or contour plot if solution u is provided
    plt.clf()
    plt.axis('equal')
    if u is None:
        plt.tripcolor(p[:,0], p[:,1], t, 0*t[:,0], cmap='Set3',
                      edgecolors='k', linewidth=1)
    else:
        plt.tricontourf(p[:,0], p[:,1], t, u, 20)
        plt.colorbar()
    plt.draw()

def unique_rows(A, return_index=False, return_inverse=False):
    """
    Similar to MATLAB's unique(A, 'rows'), this returns B, I, J
    where B is the unique rows of A and I and J satisfy
    A = B[J,:] and B = A[I,:]
    Returns I if return_index is True
    Returns J if return_inverse is True
    """
    A = np.require(A, requirements='C')
    assert A.ndim == 2, "array must be 2-dim'l"

    orig_dtype = A.dtype
    ncolumns = A.shape[1]
    dtype = np.dtype((np.character, orig_dtype.itemsize*ncolumns))
    B, I, J = np.unique(A.view(dtype),
                        return_index=True,
                        return_inverse=True)

    B = B.view(orig_dtype).reshape((-1, ncolumns), order='C')

    # There must be a better way to do this:
    if (return_index):
        if (return_inverse):
            return B, I, J
        else:
            return B, I
    else:
        if (return_inverse):
            return B, J
        else:
            return B