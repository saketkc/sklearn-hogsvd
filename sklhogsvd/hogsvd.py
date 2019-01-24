"""
This is a module to perform higher order generalized singular value decomposition.
"""
import numpy as np
from np.linalg import eig
from np.linalg import inv as invert_matrix
from np.linalg import norm
from scipy.linalg import sqrtm
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted


class HigherOrderGSVD(BaseEstimator, TransformerMixin):
    """ Perform higher order generalized singular value decomposition.


    Parameters
    ----------
    demo_param : str, default='demo'
        A parameter used for demonstation of how to pass and store paramters.

    Attributes
    ----------
    n_features_ : int
        The number of features of the data passed to :meth:`fit`.
    """

    def __init__(self):
        self.U = None
        self.sigmas = None
        self.V = None

    @staticmethod
    def _mat_inner_prod(X):
        """Return matrix inner product that can be applied to 3darray"""
        return np.vectorize(lambda x: x.T.dot(x))(X)

    @staticmethod
    def _fit_S(X):
        X = [check_array(x, accept_sparse=True) for x in X]
        N = len(X)
        data_shape = X[0].shape
        A = [x.T.dot(x) for x in X]
        A_inv = [invert_matrix(a) for a in A]
        S = np.zeros((data_shape[1], data_shape[1]))
        for i in range(N):
            for j in range(i + 1, N):
                S = S + (A[i].dot(A_inv[j]) + A[j].dot(A_inv[i]))
        S = S / (N * (N - 1))
        return S

    @staticmethod
    def _fit_B(X, V):
        X = [check_array(x, accept_sparse=True) for x in X]
        V_inv = invert_matrix(V)
        B = [V_inv.dot(x.T).T for x in X]
        return B

    @staticmethod
    def _eigen_decompostion(X):
        X = check_array(X, accept_sparse=True)
        eigen_values, V = eig(X)
        # Ensure they are sorted
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        V = V[:, idx]
        return eigen_values, V

    @staticmethod
    def _fit_U_Sigma(B):
        B = [check_array(b, accept_sparse=True) for b in B]
        sigmas = np.array([norm(b, axis=0) for b in B])
        U = [b / sigma for b, sigma in zip(B, sigmas)]
        return sigmas, U

    def fit(self, X, y=None):
        """A reference implementation of a fitting function for a transformer.

        Parameters
        ----------
        X : array-like, shape (n_samples, (n_rows_i, n_cols)
            List of training input samples. Eah input element has
            the same numbe of columns but can have unequal number of rows.

        Returns
        -------
        self : object
            Returns self.
        """
        X = [check_array(x, accept_sparse=True) for x in X]

        self.n_features_ = X[0].shape[1]

        # Step 1: Calculate normalized S
        S = self._fit_S(X)

        # Step 2: Eigen decomposition of S
        # SV = V\Lambda
        # V= (v_1, v_2, ..., v_n)
        # ||v||_k =1, but V is not necesarrily orthonormal
        eigen_values, V = self._eigen_decompostion(S)

        # Step 3: Calculate B matrices
        # V_iB_i.T = D_i.T
        B = self._fit_B(X, V)

        # Step 4: Calculate U, sigma
        # B= U\Sigma
        sigmas, U = self._fit_U_Sigma(B)

        UTU = [u.T.dot(u) for u in U]
        UTU_inv = [invert_matrix(utu) for utu in UTU]
        self.U = U
        self.U_ortho1 = [u.dot(utu_inv) for u, utu_inv in zip(U, UTU_inv)]
        self.U_ortho2 = [u.dot(invert_matrix(sqrtm(utu))) for u, utu in zip(U, UTU)]
        self.sigmas = sigmas
        self.eigen_values = eigen_values
        self.V = V
        return self

    def transform(self, X, transform_type="uinv"):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.
        transform_type: {univ, ortho}

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, "n_features_")

        # Input validation
        X = [check_array(x, accept_sparse=True) for x in X]

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X[0].shape[1] != self.n_features_:
            raise ValueError(
                "Shape of input is different from what was seen" "in `fit`"
            )
        if transform_type == "uinv":
            projection_mat = self.U
        elif transform_type == "ortho1":
            projection_mat = self.U_ortho1
        elif transform_type == "ortho2":
            projection_mat = self.U_ortho2
        else:
            raise ValueError("{} not a valid transform_type".format(transform_type))

        X_transformed = [x.T.dot(p) for x, p in zip(X, projection_mat)]
        return X_transformed
