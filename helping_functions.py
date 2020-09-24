import netket as nk
import scipy
import numpy as np
import my_operators as operators


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
def add_operator(gs, hilbert, L, operator = None):
    if(operator == 'FerroCorr'):
        for i in range(1, L):
            observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=0, k=i)
            name_fast = 'Ferro_correlation_function_fast' + str(i)
            gs.add_observable(observ_fast, name_fast)
            return gs
    elif(operator == 'StringCorr'):
        for i in range(1, L):
            observ_fast = operators.FerroCorrelationZ(hilbert=hilbert, j=0, k=i)
            name_fast = 'Ferro_correlation_function_fast' + str(i)
            gs.add_observable(observ_fast, name_fast)
            return gs
    else:
        return gs
