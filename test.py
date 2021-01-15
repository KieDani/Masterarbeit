"""Easy usage of Netket + my custom code

This script allows the user to train a machine using the run-method. A machine can also be loaded with the load method.
In the load method an observable is evaluated additionally.
Furthermore, exact results can be computed with the method exact.
It is recommended to use few samples plus many iterations with the run method and many samples plus few iterations with the load method.
The supported machines are defined in my_machines.py, the supported hamiltonians are defined in my_models.py, and the observables are defined in my_operators.py.
To use this file, you have to import it and use its functions.

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch, tqdm

This file contains the following functions:

    * run
    * load
    * exact
"""
import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time
import numpy as np
import scipy as sp
import sys
import jax






__L__ = 50
__number_samples__ = 700
__number_iterations__ = 500
__alpha__ = 4

#use Gd, if Sr == None; otherwise, sr is the diag_shift
def run(L=__L__, alpha=__alpha__, sr = None, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', hamiltonian_name = 'transformed_Heisenberg', n_samples = __number_samples__, n_iterations = __number_iterations__):
    """Method to train a machine.

    A hamiltonian and sampler can be chosen. The machine is defined and trained for the hamiltonian.

    Args:
        L (int) : The number of sites of the lattice
        alpha (int) : A factor to define the size of different machines
        sr (float) : The parameter for stochastic reconfiguration method. If it is None, stochastic reconfiguration is not used
        dataname (str) : The dataname. If None, an automatic dataname is chosen
        path (str) : The directory, where the results are saved. If None, the directory is 'run'
        machine_name (str) A string to choose the machine. Possible inputs: See get_machine in my_machines.py
        sampler (str) : A string to choose the sampler: Recommended: 'Local' (this works with every machine)
        hamiltonian_name (str) : A string to choose the hamiltonian. Possible inputs: see get_hamiltonian in my_models.py
        n_samples (int) : The number of samples used in every iteration step
        n_iterations (int) : The number of iterations (training steps)
            """
    ha, hi, g = models.get_hamiltonian(hamiltonian_name, L)
    print('uses', hamiltonian_name, 'hamiltonian')
    sys.stdout.flush()
    generate_machine = machines.get_machine(machine_name)
    ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha, optimizer='Adamax', lr=0.005, sampler=sampler)

    if(sr == None):
        gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples)
    else:
        sr = nk.optimizer.SR(ma, diag_shift=sr)
        gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples, sr=sr)

    #observables = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr')
    if(dataname == None):
        dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname, path=path)
    print('')
    functions.create_machinefile(machine_name, L, alpha, dataname, sr)
    start = time.time()
    gs.run(n_iter=int(n_iterations), out=dataname)#, obs=observables)
    end = time.time()
    with open(''.join((dataname, '.time')), 'w') as reader:
        reader.write(str(end - start))
    print('Time', end - start)
    sys.stdout.flush()


#ensure, that the machine is the same as used before!
def load(L=__L__, alpha=__alpha__, sr = None, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', hamiltonian_name = 'transformed_Heisenberg', n_samples =10000, n_iterations = 20):
    """Method to load a pretrained machine and measure some observables.

        A hamiltonian and sampler can be chosen. The machine is defined and trained for the hamiltonian.

    Args:
        L (int) : The number of sites of the lattice
        alpha (int) : A factor to define the size of different machines
        sr (float) : The parameter for stochastic reconfiguration method. If it is None, stochastic reconfiguration is not used
        dataname (str) : The dataname. If None, an automatic dataname is chosen
        path (str) : The directory, where the results are saved. If None, the directory is 'run'
        machine_name (str) A string to choose the machine. Possible inputs: See get_machine in my_machines.py
        sampler (str) : A string to choose the sampler: Recommended: 'Local' (this works with every machine)
        hamiltonian_name (str) : A string to choose the hamiltonian. Possible inputs: see get_hamiltonian in my_models.py
        n_samples (int) : The number of samples used in every iteration step
        n_iterations (int) : The number of iterations (training steps)

                """
    if (dataname == None):
        dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname, path=path)
    ha, hi, g = models.get_hamiltonian(hamiltonian_name, L)
    print('uses', hamiltonian_name, 'hamiltonian')
    sys.stdout.flush()
    print('load the machine: ', dataname)
    generate_machine = machines.get_machine(machine_name)
    ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
    ma.load(''.join((dataname, '.wf')))
    op, sa = machines.load_machine(machine=ma, hamiltonian=ha, optimizer='Adamax', lr=0.001, sampler=sampler)
    #observables = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr', symmetric=True)
    observables = {**functions.get_operator(hilbert=hi, L=L, operator='FerroCorr', symmetric=False), **functions.get_operator(hilbert=hi, L=L, operator='FerroCorr', symmetric=True)}

    print('Estimated results:')
    if(sr == None):
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples)#, n_discard=5000)
    else:
        sr = nk.optimizer.SR(ma, diag_shift=sr)
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples, sr=sr)#, n_discard=5000)

    functions.create_machinefile(machine_name, L, alpha, dataname, sr)
    start = time.time()
    gs2.run(n_iter=n_iterations, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
    end = time.time()
    with open(''.join((dataname, '_estimate', '.time')), 'w') as reader:
        reader.write(str(end - start))
    print(gs2.estimate(observables))
    print('Time', end - start)
    sys.stdout.flush()


def exact(L = __L__, symmetric = True, dataname = None, path = 'run', hamiltonian_name = 'transformed_Heisenberg'):
    """Method to solve a hamiltonian exactly.

        A hamiltonian can be chosen. The energy is evaluated using the lanczos method
        and a observable is evaluated with the power method.
        Use this only for small lattices, because it needs an awful lot of RAM.

    Args:
        L (int): The number of sites of the lattice
        symmetric (bool) :
            If True, the evaluated observable is symmetric to the center of the lattice.
            If false, it starts at one end of the lattice.
        dataname (str) : The dataname. If None, an automatic dataname is chosen
        path (str) : The directory, where the results are saved. If None, the directory is 'run'
        hamiltonian_name (str) : A string to choose the hamiltonian. Possible inputs: 'transformed_Heisenberg', 'original_Heisenberg'

                """
    ha, hi, g = models.get_hamiltonian(hamiltonian_name, L)
    print('uses', hamiltonian_name, 'hamiltonian')
    sys.stdout.flush()

    w, v_tmp = sp.sparse.linalg.eigsh(ha.to_sparse(), k=1, which='SR', return_eigenvectors=True)
    #w, v_tmp = sp.linalg.eigh(ha.to_dense())
    print(v_tmp.shape)
    print('Energy:', w[0], 'Lattice size:', L)
    sys.stdout.flush()
    # v = np.empty(3**L, dtype=np.complex128)
    # print(v.shape)
    # for index, i in enumerate(v_tmp[:, 0]):
    #     v[index] = i

    v = functions.power_method(ha.to_sparse(), L, w[0])

    if(symmetric == True):
        results = np.empty(int(L/2) -1 + L%2, dtype=np.float64)
        for index, i in enumerate(range(1, int(L / 2.) + L%2)):
            if (hamiltonian_name == 'transformed_Heisenberg' or hamiltonian_name == 'transformed_AKLT'):
                observable = operators.FerroCorrelationZ(hilbert=hi, j=int(L / 2.) - i, k=int(L / 2.) + i).to_sparse()
            else:
                print('Not implemented yet. Do not use the symmetric operator!')
                sys.stdout.flush()
                observable = None
            result_l = observable.dot(v).dot(v).real
            results[index] = result_l
    else:
        results = np.empty(L-1, dtype=np.float64)
        for index, i in enumerate(range(1, L)):
            if (hamiltonian_name == 'transformed_Heisenberg' or hamiltonian_name == 'transformed_AKLT'):
                observable = operators.FerroCorrelationZ(hilbert=hi, j=0, k=i).to_sparse()
            else:
                observable = operators.StringCorrelation(hilbert=hi, j=0, k=i).to_sparse()
            #print(observable.shape)
            #print(v.shape)
            #result_l = np.dot(np.dot(v, observable), v).real
            result_l = observable.dot(v).dot(v).real
            #print(result_l, '; ', result_l2)
            results[index] = result_l
    if(dataname == None):
        dataname = ''.join(('L', str(L), '_exact'))
    dataname = functions.create_path(dataname, path=path)
    dataname = ''.join((dataname, '.csv'))
    # save to csv file
    np.savetxt(dataname, results, delimiter=';')
    print(results)
    sys.stdout.flush()
    return results


#run(L=5, alpha=10, sr=0.01, path='test_sr', dataname='test_sr', n_samples=300, n_iterations=50, machine_name='JaxFFNN')
#load(L=5, alpha=10, sr=0.01, path='test_sr', dataname='test_sr', n_samples=3000, n_iterations=20, machine_name='JaxFFNN')

#exact(L=10, symmetric=False, transformed=True)

#jax.config.update('jax_disable_jit', True)
#run(L=4, alpha=2, n_samples=300, n_iterations=300, machine_name='JaxFFNN', sampler='VBS')

#exact(L=6, symmetric=False, hamiltonian_name='original_Heisenberg')
#run(L=16, alpha=16, machine_name='JaxDeepConvNN', sampler='Local', hamiltonian_name='transformed_Heisenberg', n_samples=500, n_iterations=300)
#load(L=16, alpha=16, machine_name='JaxDeepConvNN', sampler='Local', hamiltonian_name='transformed_Heisenberg', n_samples=2000, n_iterations=30)




# L=16
# alpha0 = 100
# ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
#
# alpha = alpha0
# print(alpha)
# machine_name = 'JaxRBM'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# compare = ma.n_par
# print(machine_name, ma.n_par)
#
# alpha = 16*alpha0
# print(alpha)
# machine_name = 'JaxSymmRBM'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.94 * alpha0)
# print(alpha)
# machine_name = 'JaxFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.07)
# print(alpha)
# machine_name = 'JaxDeepFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.14*alpha0)
# print(alpha)
# machine_name = 'JaxDeepConvNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.1*alpha0)
# print(alpha)
# machine_name = 'JaxSymmFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.35)
# print(alpha)
# machine_name = 'JaxUnaryFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.11)
# print(alpha)
# machine_name = 'JaxConv3NN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.05)
# print(alpha)
# machine_name = 'JaxResFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.33)
# print(alpha)
# machine_name = 'JaxResConvNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)




# L=30
# alpha0 = 100
# ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
#
# alpha = alpha0
# print(alpha)
# machine_name = 'JaxRBM'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# compare = ma.n_par
# print(machine_name, ma.n_par)
#
# alpha = 30*alpha0
# print(alpha)
# machine_name = 'JaxSymmRBM'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.94 * alpha0)
# print(alpha)
# machine_name = 'JaxFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.07)
# print(alpha)
# machine_name = 'JaxDeepFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.12*alpha0)
# print(alpha)
# machine_name = 'JaxDeepConvNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.1*alpha0)
# print(alpha)
# machine_name = 'JaxSymmFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.34)
# print(alpha)
# machine_name = 'JaxUnaryFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.11)
# print(alpha)
# machine_name = 'JaxConv3NN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.05)
# print(alpha)
# machine_name = 'JaxResFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.61)
# print(alpha)
# machine_name = 'JaxResConvNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)



# L=50
# alpha0 = 100
# ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
#
# alpha = alpha0
# print(alpha)
# machine_name = 'JaxRBM'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# compare = ma.n_par
# print(machine_name, ma.n_par)
#
# alpha = 50*alpha0
# print(alpha)
# machine_name = 'JaxSymmRBM'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.98 * alpha0)
# print(alpha)
# machine_name = 'JaxFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.07)
# print(alpha)
# machine_name = 'JaxDeepFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.11*alpha0)
# print(alpha)
# machine_name = 'JaxDeepConvNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(0.1*alpha0)
# print(alpha)
# machine_name = 'JaxSymmFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.34)
# print(alpha)
# machine_name = 'JaxUnaryFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.10)
# print(alpha)
# machine_name = 'JaxConv3NN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 0.05)
# print(alpha)
# machine_name = 'JaxResFFNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)
#
# alpha = int(alpha0 * 1.02)
# print(alpha)
# machine_name = 'JaxResConvNN'
# generate_machine = machines.get_machine(machine_name)
# ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
# print(machine_name, ma.n_par, compare/ma.n_par)