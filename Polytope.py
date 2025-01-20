import numpy as np
from scipy.linalg import block_diag

class Polytope():
    def __init__(self, H, h):
        """Define a polytope in H-representation: all x such that Hx <= h.

        Parameters
        ----------
        H: ndarray, shape (n, dim)
            Matrix of hyperplane normal vectors as rows, n is the number of planes
            and dim is the dimension of the space in which the planes are embedded.
        h: ndarray, shape (n,)
            Vector of n values.
        
        Attributes
        ----------
        H: ndarray, shape (n, dim)
        h: ndarray, shape (n, 1)
        n: int
            number of planes
        dim: int
            dimension of embedding space
        """
        #print(H.shape, h.shape)
        assert H.shape[0] == h.shape[0]
        self.H = H
        self.h = h.reshape(h.shape[0],1)
        self.shape = np.shape(self.H)
        
    def cart(self, p1):
        """
        Parameters
        ----------
        p1: Polytope

        Returns
        -------
        Polytope
            Cartesian product of the polytopes self and p1.
        """
        H01 = block_diag(self.H, p1.H)
        h01 = np.block([[self.h],
                        [p1.h  ]])
        assert H01.shape[0] == h01.shape[0]
        return Polytope(H01, h01)
    
    def cartpower(self, n):
        """
        Parameters
        ----------
        n: int

        Returns
        -------
        Polytope
            The nth cartesian power of the polytope self.
        """
        assert self.H.shape[0] == self.h.shape[0]
        Hn = np.kron(np.eye(n), self.H)
        hn = np.kron(np.ones([n,1]), self.h)
        assert Hn.shape[0] == hn.shape[0]
        return Polytope(Hn, hn)
    
def H_cube(c, r):
    """Define a hypercube of dimension len(c)
    Parameters
    ----------
    c: ndarray, shape (len(c),)
        Center of hypercube
    r: float or ndarray, shape (len(c),)
        Half side lengths of the hypercube in each dimension
        [r_1, ... , r_len(c)]

    +-----------+
    |           |
    |     c     |2*r_2
    |           |
    +-----------+
        2*r_1

    Return
    ------
    Polytope
        len(c) dimensional hypercube with center c and half side lengths r.
    """
    n = np.size(c)
    if np.size(r) == 1:
        r = n*[r]
    H = np.kron( np.eye(n), np.array([[1], [-1]]) )
    h = np.zeros(2*n)
    for i in range(n):
        h[2*i:2*(i+1)] = [c[i] + r[i], -(c[i] - r[i])]
    assert H.shape[0] == h.shape[0]
    return Polytope(H, h)

def cart_H_cube(center_times, radius_times):
    """
    Paramters
    ---------
    center_times: list
        list of centers
    radius_times: list
        list of radii
    Returns
    -------
    Polytope
        cartesian product of H_cubes with centers in center_times and radii in radius_times
    """
    assert len(center_times) == len(radius_times)
    for i in range(len(center_times)):
        if i == 0:
            H = H_cube(center_times[0], radius_times[0])
        else:
            H = H.cart(H_cube(center_times[i], radius_times[i]))
    return H
