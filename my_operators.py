"""Implementation of some observables

The stringcorrelation operator and the ferromagnetic correlation operator are implemented. There is also a slow
implementation to check, if I did any mistakes. I do not recommend to use the slow version of the operator.
To use the operators easily, you can use the method get_operator in helping_functions.py.
To implement the observables, I mainly used the code of NetKets BoseHubbard hamiltonian.

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch, tqdm, matplotlib

This file contains the following classes:

    * StringCorrelation
    * FerroCorrelationZ

This file contains the following functions:

    * FerroCorrelationZ_slow
    * StringCorrelation_slow
"""
import netket as nk
from netket.operator._abstract_operator import AbstractOperator

import numpy as _np
from numba import jit


class StringCorrelation(AbstractOperator):
    r"""
    The string-correlation-operator going from site j to site k.
    """

    def __init__(self, hilbert, j, k):
        r"""
        Constructs a new stringcorrelationoperator.

            Args:
               hilbert (netket.hilbert.Boson): Hilbert space the operator acts on.
               j (int): The Operator starts at site j.
               k (int) : The Operator ends at site k.

        """
        self._j = j
        self._k = k
        self._hilbert = hilbert
        self._n_sites = hilbert.size
        self._section = hilbert.size + 1
        #self._edges = _np.asarray(hilbert.graph.edges())
        self._edges = None
        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._n_sites

    @staticmethod
    @jit(nopython=True)
    def n_conn(x, out):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.

            Returns:
                array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.int32)

        out.fill(x.shape[1] + 1)

        return out

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (array): An array of shape (hilbert.size) containing the quantum numbers x.

            Returns:
                matrix: The connected states x' of shape (N_connected,hilbert.size)
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        return self._flattened_kernel(
            x.reshape((1, -1)), _np.ones(1), self._edges, self._j, self._k,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(x, sections, edges, j, k):
        n_sites = x.shape[1]
        n_conn = n_sites + 1
        # n_conn = 1

        x_prime = _np.empty((x.shape[0] * n_conn, n_sites))
        mels = _np.empty(x.shape[0] * n_conn)

        diag_ind = 0

        for i in range(x.shape[0]):

            mels[diag_ind] = 1.0
            if (x[i, j] >= 1):
                mels[diag_ind] *= 1.
            elif (x[i, j] <= -1):
                mels[diag_ind] *= -1.
            else:
                mels[diag_ind] *= 0.

            for i2 in range(j + 1, k):
                if (x[i, i2] >= 1):
                    mels[diag_ind] *= -1.
                elif (x[i, i2] <= -1):
                    mels[diag_ind] *= -1.
                else:
                    mels[diag_ind] *= 1.

            if (x[i, k] >= 1):
                mels[diag_ind] *= 1.
            elif (x[i, k] <= -1):
                mels[diag_ind] *= -1.
            else:
                mels[diag_ind] *= 0.

            odiag_ind = 1 + diag_ind

            mels[odiag_ind: (odiag_ind + n_sites)].fill(0.)

            x_prime[diag_ind: (diag_ind + n_conn)] = _np.copy(x[i])

            # for j2 in range(n_sites):
            #    x_prime[j2 + odiag_ind][j2] *= -1.0

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                sections (array): An array of size (batch_size) useful to unflatten
                            the output of this function.
                            See numpy.split for the meaning of sections.
                pad (bool): no effect here

            Returns:
                matrix: The connected states x', flattened together in a single matrix.
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._flattened_kernel(x, sections, self._edges, self._j, self._k)


class FerroCorrelationZ(AbstractOperator):
    r"""
    The ferromagnetic correlation-operator between site j and k. Spins in z-direction.
    """

    def __init__(self, hilbert, j, k):
        r"""
        Constructs a new ferromagnetic correlation operator

        Args:
           hilbert (netket.hilbert.Boson): Hilbert space the operator acts on.
           j (int): The first spin is at site j.
           k (int): The last spin is at site k.

        """
        self._j = j
        self._k = k
        self._hilbert = hilbert
        self._n_sites = hilbert.size
        self._section = hilbert.size + 1
        #self._edges = _np.asarray(hilbert.graph.edges())
        self._edges = None
        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._n_sites

    @staticmethod
    @jit(nopython=True)
    def n_conn(x, out):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.

            Returns:
                array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.int32)

        out.fill(x.shape[1] + 1)

        return out

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (array): An array of shape (hilbert.size) containing the quantum numbers x.

            Returns:
                matrix: The connected states x' of shape (N_connected,hilbert.size)
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        return self._flattened_kernel(
            x.reshape((1, -1)), _np.ones(1), self._edges, self._j, self._k,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(x, sections, edges, j, k):
        n_sites = x.shape[1]
        n_conn = n_sites + 1
        # n_conn = 1

        x_prime = _np.empty((x.shape[0] * n_conn, n_sites))
        mels = _np.empty(x.shape[0] * n_conn)

        diag_ind = 0

        for i in range(x.shape[0]):

            mels[diag_ind] = 1
            if(x[i, j] >= 1):
                mels[diag_ind] = mels[diag_ind] * 1
            elif(x[i, j] <= -1):
                mels[diag_ind] = mels[diag_ind] * -1
            else:
                mels[diag_ind] = 0

            if (x[i, k] >= 1):
                mels[diag_ind] = mels[diag_ind] * 1
            elif (x[i, k] <= -1):
                mels[diag_ind] = mels[diag_ind] * -1
            else:
                mels[diag_ind] = 0


            # This might be wrong, because x in [+2, -2] instead of [+1, -1] ?
            #mels[diag_ind] = x[i, j] * x[i, k]

            odiag_ind = 1 + diag_ind

            mels[odiag_ind: (odiag_ind + n_sites)].fill(0.)

            x_prime[diag_ind: (diag_ind + n_conn)] = _np.copy(x[i])

            # for j2 in range(n_sites):
            #    x_prime[j2 + odiag_ind][j2] *= -1.0

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                sections (array): An array of size (batch_size) useful to unflatten
                            the output of this function.
                            See numpy.split for the meaning of sections.
                pad (bool): no effect here

            Returns:
                matrix: The connected states x', flattened together in a single matrix.
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._flattened_kernel(x, sections, self._edges, self._j, self._k)


class S_Z_squared(AbstractOperator):
    r"""
    Measures the squared Spin operator in z direction at site j
    """

    def __init__(self, hilbert, j):
        r"""
        Constructs a new squared Spin operator in z direction at site j

        Args:
           hilbert (netket.hilbert.Boson): Hilbert space the operator acts on.
           j (int): The site where the operator is measured.

        """
        self._j = j
        self._hilbert = hilbert
        self._n_sites = hilbert.size
        self._section = hilbert.size + 1
        #self._edges = _np.asarray(hilbert.graph.edges())
        self._edges = None
        super().__init__()

    @property
    def hilbert(self):
        r"""AbstractHilbert: The hilbert space associated to this operator."""
        return self._hilbert

    @property
    def size(self):
        return self._n_sites

    @staticmethod
    @jit(nopython=True)
    def n_conn(x, out):
        r"""Return the number of states connected to x.

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                out (array): If None an output array is allocated.

            Returns:
                array: The number of connected states x' for each x[i].

        """
        if out is None:
            out = _np.empty(x.shape[0], dtype=_np.int32)

        out.fill(x.shape[1] + 1)

        return out

    def get_conn(self, x):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (array): An array of shape (hilbert.size) containing the quantum numbers x.

            Returns:
                matrix: The connected states x' of shape (N_connected,hilbert.size)
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """
        return self._flattened_kernel(
            x.reshape((1, -1)), _np.ones(1), self._edges, self._j,
        )

    @staticmethod
    @jit(nopython=True)
    def _flattened_kernel(x, sections, edges, j):
        n_sites = x.shape[1]
        n_conn = n_sites + 1
        # n_conn = 1

        x_prime = _np.empty((x.shape[0] * n_conn, n_sites))
        mels = _np.empty(x.shape[0] * n_conn)

        diag_ind = 0

        for i in range(x.shape[0]):

            mels[diag_ind] = 1
            if(x[i, j] >= 1):
                mels[diag_ind] = 1
            elif(x[i, j] <= -1):
                mels[diag_ind] = 1
            else:
                mels[diag_ind] = 0



            # This might be wrong, because x in [+2, -2] instead of [+1, -1] ?
            #mels[diag_ind] = x[i, j] * x[i, k]

            odiag_ind = 1 + diag_ind

            mels[odiag_ind: (odiag_ind + n_sites)].fill(0.)

            x_prime[diag_ind: (diag_ind + n_conn)] = _np.copy(x[i])

            # for j2 in range(n_sites):
            #    x_prime[j2 + odiag_ind][j2] *= -1.0

            diag_ind += n_conn

            sections[i] = diag_ind

        return x_prime, mels

    def get_conn_flattened(self, x, sections, pad=False):
        r"""Finds the connected elements of the Operator. Starting
            from a given quantum number x, it finds all other quantum numbers x' such
            that the matrix element :math:`O(x,x')` is different from zero. In general there
            will be several different connected states x' satisfying this
            condition, and they are denoted here :math:`x'(k)`, for :math:`k=0,1...N_{\mathrm{connected}}`.

            This is a batched version, where x is a matrix of shape (batch_size,hilbert.size).

            Args:
                x (matrix): A matrix of shape (batch_size,hilbert.size) containing
                            the batch of quantum numbers x.
                sections (array): An array of size (batch_size) useful to unflatten
                            the output of this function.
                            See numpy.split for the meaning of sections.
                pad (bool): no effect here

            Returns:
                matrix: The connected states x', flattened together in a single matrix.
                array: An array containing the matrix elements :math:`O(x,x')` associated to each x'.

        """

        return self._flattened_kernel(x, sections, self._edges, self._j)



#copied from my old code. Maybe I can make the code look nicer. But I only needed it to compare it to the fast version.
def FerroCorrelationZ_slow(hilbert, j, k):
    r"""
        Slow implementation of the FerroCorrelationZ operator. I do not recommend to use this.
        """
    hi = hilbert
    # We need to specify the local operators as a matrix acting on a local Hilbert space
    sf = []
    sites = []
    sigmaz = _np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    bigfatone = _np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    helper = []
    mszs = _np.kron(sigmaz, bigfatone)
    if ((k-j) == 1):
        mszs = _np.kron(sigmaz, sigmaz)
        helper = [j, k]
    else:
        for i in range(j+1, k - 1):
            mszs = _np.kron(mszs, bigfatone)
        mszs = _np.kron(mszs, sigmaz)
        for i in range(j, k+1):
            helper.append(i)
    sf.append((mszs).tolist())
    print(k, ' ', j, ' ', k-j)
    print(helper)
    print('Size of Observable:')
    print(mszs.shape)
    sites.append(helper)
    string_correlation_function = nk.operator.LocalOperator(hi, sf, sites)
    return string_correlation_function


def StringCorrelation_slow(hilbert, j, k):
    r"""
            Slow implementation of the StringCorrelation operator. I do not recommend to use this.
            """
    hi = hilbert
    # We need to specify the local operators as a matrix acting on a local Hilbert space
    sf = []
    sites = []
    sigmaz = _np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    expSz = _np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
    helper = []
    mszs = _np.kron(sigmaz, expSz)
    if ((k - j) == 1):
        mszs = _np.kron(sigmaz, sigmaz)
        helper = [j, k]
    else:
        for i in range(j + 1, k - 1):
            mszs = _np.kron(mszs, expSz)
        mszs = _np.kron(mszs, sigmaz)
        for i in range(j, k + 1):
            helper.append(i)
    sf.append((mszs).tolist())
    print(k, ' ', j, ' ', k - j)
    print(helper)
    print('Size of Observable:')
    print(mszs.shape)
    sites.append(helper)
    string_correlation_function = nk.operator.LocalOperator(hi, sf, sites)
    return string_correlation_function