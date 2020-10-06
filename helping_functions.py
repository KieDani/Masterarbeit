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
            #gs.add_observable(observ_fast, name_fast)
    elif(operator == 'StringCorr'):
        for i in range(1, L):
            observ_fast = operators.StringCorrelation(hilbert=hilbert, l=i)
            name_fast = 'String_correlation_function' + str(i)
            observables[name_fast] = observ_fast
            #gs.add_observable(observ_fast, name_fast)
    return observables


def create_path(dataname, path='run'):
    try:
        os.makedirs(path)
    except OSError:
        print("Creation of the directory %s failed" % path)
    else:
        print("Successfully created the directory %s" % path)
    return '/'.join((path, dataname))
