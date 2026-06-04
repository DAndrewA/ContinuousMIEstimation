#from . import _MIxnyn
import ctypes
import numpy as np
import os
from dataclasses import dataclass
from typing import Self, Optional

so_files = [f for f in os.listdir(os.path.dirname(__file__)) if f[-3:] == ".so"]
print(f"continuousmiestimation: Available compiled .so files are {so_files}")
if len(so_files) == 0:
    raise FileNotFoundError("No compiled _MIxnyn.*.so file exists")
elif len(so_files) > 1:
    print("continuousmiestimation: WARN: using last available .so file by alphabetical (prioritises python version)")
so_file = sorted(so_files)[-1]

fpath_so = os.path.join(
    os.path.dirname(__file__),
    so_file
)
LIB = ctypes.CDLL(fpath_so)

LIB.MIxnyn.argtypes = [
    ctypes.POINTER(ctypes.POINTER(ctypes.c_double)),  # double **x
    ctypes.c_int,                                      # int dimx
    ctypes.c_int,                                      # int dimy
    ctypes.c_int,                                      # int K
    ctypes.c_int,                                      # int N
    ctypes.POINTER(ctypes.c_double)                   # double *MI
]
LIB.MIxnyn.restype = None


@dataclass(frozen=True, kw_only=True)
class MI:
    """Value holding the mutual inforamtion computation result
    ATTRIBUTES:
        value: float
            The computed mutual information value
        std: Optional[float]
            If the variance of the value is estimated, this contains the square root of the variance. Otherwise None.
        n_samples: int
            The number of smaples used in computing the value
        K: int
            The k-th nearest neighbour parameter used in the KSG estimator
        M: Optional[int]
            If the variance is estimated, the degrees of freedom of the estimated B value
        n_splits: Optional[int]
            If the variance is estimated, the number of non-overlapping subsets for which the mutual information is computed and between which the standard deviation is estimated.
    """
    value: float
    std: Optional[float]
    n_samples: int
    K: int
    M: Optional[int]
    n_splits: Optional[int]


class MIEstimator:
    """Class holding parameters for the KSG mutual information estimator.

    ATTRIDBUTES:
        K: int
            k-th nearest neighbour parameter used in the KSG estimator

        M: int
            The number of times a standard deviation of mutual information values over non-overlapping subsets of the data is computed.
            This is the degrees of freedom in the estimation of the maximum likelihood of the chi^2 distributed value B that is scaled to estimate the standard deviation of the mutual information estimate.

        n_splits: int
            The number of non-overlapping subsets of the data (X Y) that are generated to estimate a standard deviation of the mutual inforamtion.
            
        RNG: Optional[np.random.Generator]
            An optional numpy random number Generator for producing subsets of the original data.
            If None, a default generator will be created.
    
    METHODS:
        estimate(X: np.ndarray, Y: np.ndarray, *, n_samples: int) -> MIEstimate
            Given samples X and Y of equal length in the first dimension i.e.( X(n_samples,d_X), Y(n_samples, d_Y)),
            compute the mutual information between the two sets of samples using the KSG estiator defined by Self's
            parameter values.
    """
    def __init__(self, *, K: int, M: int, n_splits: int, RNG: Optional[np.random.Generator] = None):
        assert isinstance(K, int), f"{type(K)=} must be an int"
        assert isinstance(M, int), f"{type(M)=} must be an int"
        assert isinstance(n_splits, int), f"{type(n_splits)=} must be an int"
        assert isinstance(RNG, np.random.Generator | None), f"{type(RNG)=} must be np.random.Generator or None"

        self.K = K
        self.M = M
        self.n_splits = n_splits
        if RNG is None:
            RNG = np.random.default_rng()
        self.RNG = RNG

    @property
    def parameters(self) -> dict[str, int]:
        """Return a dictionary of the KSG algorithm parameters.
        NOTE: this does not include the RNG value.
        """
        return dict(
            K = self.K,
            M = self.M,
            n_splits = self.n_splits,
        )

    def estimate(self, X: np.ndarray, Y: np.ndarray, *, n_samples: int) -> MI:
        """
        INPUTS:
            X: (nx, n_samples) array_like
                An ordered array of n_samples, nx-dimensional samples drawn from distribution X. 

            Y: (ny, n_samples) array_like
                An ordered array of n_samples, ny-dimensional samples drawn from distribution Y.

            n_samples: int
                The number of samples between which the mutual information is calculated.
                This is explicitly passed in to ensure that the function is called on inputs of the correct shape.
        """
        mi_value = call_MI_xnyn(X=X, Y=Y, K=self.K, n_samples=n_samples)
        return MI(
            value = mi_value,
            std = None,
            n_samples = n_samples,
            K = self.K,
            M = None,
            n_splits = None,
        )

    def estimate_with_uncertainty(self, X, Y, *, n_samples: int):
        mi_value = call_MI_xnyn(X=X, Y=Y, K=self.K, n_samples=n_samples)
        # M times:
        #     generate n_splits non-overlapping subsets of the input data
        #     compute the mi value of the non-overlapping subsets
        #     compute the standard deviation between the mi values of the subsets
        # Then, estimate the ML value of the B scalar.
        # Then scale B_ML to estimate the variance of the mi estimate
        sigma_i = list()
        n_i = list()
        for _ in range(self.M):
            MI_for_current_iteration = list()
            for shuffled_indices in np.array_split(
                    self.RNG.permutation(n_samples),
                    indices_or_sections = self.n_splits,
                ):
                shuffled_X = X[:,shuffled_indices]
                shuffled_Y = Y[:,shuffled_indices]
                new_n_samples = len(shuffled_indices)
                MI_for_current_iteration.append(
                    call_MI_xnyn(
                        X = shuffled_X,
                        Y = shuffled_Y,
                        K = self.K,
                        n_samples = new_n_samples,
                    )
                )
            sigma_i.append(np.std(MI_for_current_iteration))
            n_i.append(n_splits)

        sigma_i = np.asarray(sigma_i)
        n_i = np.asarray(n_i)
        # computation is Eq. (8)/N from (10.1103/PhysRevE.100.022404). Communications with Holmes shows that this is the correct formulation of the formula, as 1. the chi2 distribution has an additional factor of 1/2 in the exponentiation, and 2. x~sigma_i is poorly defined, so a probability density based on a value a_i sigma_i^2 / B needs to be used, making use of the Jacobian |d(a_i sigma_i^2/B)/d(sigma_i^2)|
        a_i_div_N = (n_i - 1) / n_i
        k_i = n_i - 1
        std_MI = np.sqrt(
            np.sum(
                a_i_div_N * np.power(sigma_i, 2)
            )
            / np.sum(k_i)
        )
        return MI(
            value = mi_value,
            std = std_MI,
            n_samples = n_samples,
            K = self.K,
            M = self.M,
            n_splits = self.n_splits,
        )


def call_MI_xnyn(X: np.ndarray, Y:np.ndarray, *, K:int, n_samples: int) -> float:
    """A safe function to call LIB.MI_xnyn, that ensures certain assertions on the data are met before the computation can proceed"""
    dimx, Nx = [int(v) for v in X.shape]
    dimy, Ny = [int(v) for v in Y.shape]

    assert Nx == Ny, f"Number of samples in X and Y are unqual: ({Nx=}) != ({Ny=})"
    assert Nx == n_samples, f"Number of samples given is not the expected number of samples, {Nx} != ({n_samples=})"

    assert isinstance(K,int), f"{K=} must be integer"
    assert isinstance(n_samples, int), f"{n_samples=} must be integer"
    assert isinstance(dimx, int)
    assert isinstance(dimy, int)

    return _unsafe_call_MI_xnyn(
        X=X,
        Y=Y,
        K=K
    )


def _unsafe_call_MI_xnyn(X: np.ndarray, Y: np.ndarray, K: int) -> float:
    """Function that calls LIB.MIxnyn without any assertions on the input data"""
    dimx, N = [int(v) for v in X.shape]
    dimy, _ = [int(v) for v in Y.shape]

    input_x = np.ascontiguousarray(
        np.concatenate((X,Y),axis=0),
        dtype=np.float64
    )

    MI = ctypes.c_double(0.0)

    x_ptrs = (ctypes.POINTER(ctypes.c_double) * input_x.shape[0])()
    for i in range(input_x.shape[0]):
        x_ptrs[i] = input_x[i].ctypes.data_as(ctypes.POINTER(ctypes.c_double))

    LIB.MIxnyn(
        x_ptrs,
        dimx,
        dimy,
        K,
        N,
        ctypes.byref(MI)
    )
    return MI.value
