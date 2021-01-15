"""File with useful functions

I added some helpful functions that I use in my other files.

This file contains the following functions:

    * Lanczos
    * get_operator
    * create_path
    * create_machinefile
    * test_operator_startingpoint
    * test_operator_both_sides
    * power_method
"""
import netket as nk
import scipy
import numpy as np
import my_operators as operators
import os
import sys


#If possible, returns exact energy. Else, it returns None!
def Lanczos(hamilton, L):
    """Lanczos method to calculate the ground state energy of a hamiltonian.
        It only works for L<= 12. If L is larger, it returns None to prevent memory overflow.

        Args:
            hamiltonian (netket.hamiltonian) : hamiltonian
            L (int) : Lattice size

        Returns:
            exact_ens (float) : exact ground state energy
                                                """
    if L <= 12:
        exact_ens = scipy.sparse.linalg.eigsh(hamilton.to_sparse(), k=1, which='SA', return_eigenvectors=False)
        print("Exact energy is : ", exact_ens[0])
        return exact_ens
    else:
        print('Too big for exact diagonalization')
        return None


#ToDo if symmetric=True works fine, make it standard and remove parameter symmetric
#symmetric=True works only for 'FerroCorr'g
#allowed inputs for operator: 'FerroCorr', 'StringCorr', None
def get_operator(hilbert, L, operator = None, symmetric = True):
    """Method to get the observable, that is measured.
        It works with FerroCorrelationZ and StringCorrelation. I recommend to use symmetric = False.
        symmetric=True works only for 'FerroCorrelationZ'. I did not implement it for StringCorrelation.

            Args:
                hilbert (netket.hilbert) : hilbert space
                L (int) : Lattice size
                operator (str) : allowed inputs are 'FerroCorr' and 'StringCorr'
                symmetric (bool) : If True, the observable is measured symmetric to the center of the lattice.
                                    If False, the measurement is started at the 0-th site.

            Returns:
                observables (dict) : dictionary with names of observable (str) and observables (netket.operator)
                                                    """
    observables = {}
    if(operator == 'FerroCorr'):
        if(symmetric==False):
            for i in range(1, L):
                observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=0, k=i)
                name_fast = 'Ferro_correlation_function' + str(i)
                observables[name_fast] = observ_fast
        else:
            for i in range(1, int(L/2.) + L%2):
                observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=int(L/2.)-i, k=int(L/2.)+i)
                name_fast = 'Symmetric_Ferro_correlation_function' + str(int(2*i)) #because k-j=2*i
                observables[name_fast] = observ_fast
    elif(operator == 'StringCorr'):
        for i in range(1, L):
            observ_fast = operators.StringCorrelation(hilbert=hilbert, j=0, k=i)
            name_fast = 'String_correlation_function' + str(i)
            observables[name_fast] = observ_fast
    elif(operator == 'FerroCorr_slow'):
        for i in range(2, np.minimum(L + 1, 9)):
            observ = operators.FerroCorrelationZ_slow(hilbert, l = i)
            name = 'Ferro_correlation_function_slow' + str(i-1)
            observables[name] = observ
    return observables


def create_path(dataname, path='run'):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)
    return '/'.join((path, dataname))


def create_machinefile(machine_name, L, alpha, dataname, use_sr):
    with open(''.join((dataname, '.machine')), 'w') as f:
        f.write(''.join((machine_name, '\n')))
        f.write(''.join(('L = ', str(L), '\n')))
        f.write(''.join(('Alpha = ', str(alpha), '\n')))
        f.write(''.join(('diag_shift = ', str(use_sr), '\n')))




def test_operator_startingpoint(hilbert, L, fast=True):
    """Method to compare the effect of the starting site of the observable FerroCorrelationZ.
            I recommend to use fast=True to use the fast implementation of FerroCorrelationZ.

                Args:
                    hilbert (netket.hilbert) : hilbert space
                    L (int) : Lattice size
                    fast (bool) : Use True

                Returns:
                    observables (dict) : dictionary with names of observable (str) and observables (netket.operator)
                                                        """
    observables = {}
    if(fast == True):
        for start, j in enumerate([1, 2, 3, 4, 5, int(L/5.), int(L/4.), int(L/3.), int(L/2.), int(3 * L/2.)]):
            for k in range(j+1, L):
                observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=j, k=k)
                name_fast = ''.join((str(j), 'Ferro_correlation_function', str(k - j)))
                observables[name_fast] = observ_fast
    else:
        for start, j in enumerate([1, 2, 3, 4, 5, int(L/5.), int(L/4.), int(L/2.), int(3 * L/2.)]):
            for k in range(j+1, np.minimum(j + 8, L)):
                observ_fast = operators.FerroCorrelationZ_slow(hilbert, j, k)
                name_fast = ''.join((str(j), 'Ferro_correlation_function', str(k - j)))
                observables[name_fast] = observ_fast
    return observables

def test_operator_both_sides(hilbert, L):
    """Method to check if the result is symmetric.
        Starting point is the center of the lattice. It goes from the center to the left
        and from the center to the right.

                    Args:
                        hilbert (netket.hilbert) : hilbert space
                        L (int) : Lattice size

                    Returns:
                        observables (dict) : dictionary with names of observable (str) and observables (netket.operator)
                                                            """
    observables = {}
    for i in range(1, int(L/2.)):
        observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=int(L/2.), k=int(L/2.) + i)
        name_fast = 'Ferro_correlation_function' + str(i)
        observ_fast_mirrored = operators.FerroCorrelationZ(hilbert=hilbert, j=int(L / 2.), k=int(L / 2.) - i)
        name_fast_mirrored = 'Ferro_correlation_function_mirrored' + str(i)
        observables[name_fast] = observ_fast
        observables[name_fast_mirrored] = observ_fast_mirrored
    return observables


#hamiltonian has to be a sparse matrix
def power_method(hamiltonian, L, eigenvalue_lanczos):
    """Calculates the gound state vector.
                At the moment it works only if the ground state energy is the dominant eigenvalue.
                It needs a lot of RAM. Therefore, use it only for small lattices.

                    Args:
                        hamiltonian (scipy.sparse) : use to_dense of netket.hamiltonian object.
                        L (int) : Lattice size
                        eigenvalue_lanczos (float) : exact ground state energy

                    Returns:
                        x (numpy.array) : ground state of the hamiltonian.
                                                            """

    #generate starting vector
    #ToDo find out, why I need a real vector!!!
    x = np.random.random_sample(3**L) - 0.5 #+ np.random.random_sample(3**L) * 1j - 0.5j
    x = x / np.linalg.norm(x)
    #find the eigenvector
    for i in range(1, 50000):
        x = hamiltonian.dot(x)
        eigval = np.linalg.norm(x)
        if (i % 250 == 0):
            print('guess of eigenvalue', eigval)
            sys.stdout.flush()
        x = x / np.linalg.norm(x)
        if (np.abs(np.abs(eigenvalue_lanczos) - np.abs(eigval)) < 0.000001):
            print('Found solution', eigval)
            sys.stdout.flush()
            print('Needed steps:', i)
            sys.stdout.flush()
            return x
    print('Did not find a proper solution. Found', eigval)
    sys.stdout.flush()
    return x


