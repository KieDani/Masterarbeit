import netket as nk
import scipy
import numpy as np
import my_operators as operators
import os


#TODO check, why transformed Hamiltonian does not work!
#If possible, returns exact energy. Else, it returns None!
def Lanczos(hamilton, L):
    if L <= 12:
        exact_ens = scipy.sparse.linalg.eigsh(hamilton.to_sparse(), k=1, which='SA', return_eigenvectors=False)
        print("Exact energy is : ", exact_ens[0])
        return exact_ens
    else:
        print('Too big for exact diagonalization')
        return None


#allowed inputs for operator: 'FerroCorr', 'StringCorr', None
def get_operator(hilbert, L, operator = None):
    observables = {}
    if(operator == 'FerroCorr'):
        for i in range(1, L):
            observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=0, k=i)
            name_fast = 'Ferro_correlation_function' + str(i)
            observables[name_fast] = observ_fast
    elif(operator == 'StringCorr'):
        for i in range(1, L):
            observ_fast = operators.StringCorrelation(hilbert=hilbert, l=i)
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
        f.write(''.join(('Use_machine = ', str(use_sr), '\n')))




def test_operator_startingpoint(hilbert, L, fast=True):
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
