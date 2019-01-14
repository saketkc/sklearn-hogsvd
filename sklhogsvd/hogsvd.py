"""
This is a module to perform higher order generalized singular value decomposition.
"""
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_X_y, check_array, check_is_fitted

#from numpy.linalg import multi_dot


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
        return np.vectorize(lambda x: x.T.dot(x), signature='(m,n)->(n,n)')(X)

    @staticmethod
    def _fit_S(X):
        X = check_array(
            X, ensure_min_samples=2, allow_nd=True, accept_sparse=True)
        N = X.shape[0]
        data_shape = X[0].shape
        A = HigherOrderGSVD._mat_inner_prod(X)
        A_inv = np.linalg.inv(A)
        S = np.zeros((data_shape[1], data_shape[1]))
        for i in range(N):
            for j in range(i + 1, N):
                S = S + (np.dot(A[i], A_inv[j]) + np.dot(A[j], A_inv[i]))
        S = S / (N * (N - 1))
        return S

    @staticmethod
    def _fit_B(X, V):
        X = check_array(
            X, ensure_min_samples=2, allow_nd=True, accept_sparse=True)
        V_inv = np.linalg.inv(V)
        get_b = np.vectorize(
            lambda vinv, x: np.dot(vinv, x.T).T,
            signature='(n,n),(m,n)->(m,n)')
        B = get_b(V_inv, X)
        return B

    @staticmethod
    def _eigen_decompostion(X):
        X = check_array(X, accept_sparse=True)
        eigen_values, V = np.linalg.eig(X)
        # Ensure they are sorted
        idx = eigen_values.argsort()[::-1]
        eigen_values = eigen_values[idx]
        V = V[:, idx]
        return eigen_values, V

    @staticmethod
    def _fit_U_Sigma(B):
        B = check_array(
            B, ensure_min_samples=2, allow_nd=True, accept_sparse=True)
        get_sigma = np.vectorize(
            lambda b_i: np.linalg.norm(b_i, axis=0), signature='(m,n)->(n)')
        sigmas = get_sigma(B)
        U = np.divide(B, sigmas[:, None, :])
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
        X = check_array(
            X, ensure_min_samples=2, allow_nd=True, accept_sparse=True)

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

        self.U = U
        utu = self._mat_inner_prod(U)
        utu_inv = np.linalg.inv(utu)
        self.U_ortho = np.dot(U, utu_inv)
        self.sigmas = sigmas
        self.V = V
        return self

    def transform(self, X):
        """ A reference implementation of a transform function.

        Parameters
        ----------
        X : {array-like, sparse-matrix}, shape (n_samples, n_features)
            The input samples.

        Returns
        -------
        X_transformed : array, shape (n_samples, n_features)
            The array containing the element-wise square roots of the values
            in ``X``.
        """
        # Check is fit had been called
        check_is_fitted(self, 'n_features_')

        # Input validation
        X = check_array(X, allow_nd=True, accept_sparse=True)

        # Check that the input is of the same shape as the one passed
        # during fit.
        if X[0].shape[1] != self.n_features_:
            raise ValueError('Shape of input is different from what was seen'
                             'in `fit`')
        transform = self.U_ortho
        X_transformed = np.dot(X.T, self.U)
        return X_transformed
