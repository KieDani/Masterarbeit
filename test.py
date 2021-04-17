"""Easy usage of Netket + my custom code

This script allows the user to train a machine using the run function. A machine can also be loaded with the load function.
To measure an observable, you can use the function measureObservable.
Furthermore, exact results can be computed with the method exact.
It is recommended to use few samples plus many iterations with the run method and many samples plus few iterations with the load method.
The supported machines are defined in my_machines.py, the supported hamiltonians are defined in my_models.py, and the observables are defined in my_operators.py.
To use this file, you have to import it and use its functions.

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch, tqdm, matplotlib

This file contains the following functions:

    * run
    * load
    * measureObservables
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
import csv


__L__ = 50
__number_samples__ = 700
__number_iterations__ = 500
__alpha__ = 4


def run(L=__L__, alpha=__alpha__, sr = None, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', hamiltonian_name = 'transformed_Heisenberg', n_samples = __number_samples__, n_iterations = __number_iterations__):
    """Method to train a machine.

        A hamiltonian and sampler can be chosen. The machine is defined and trained for the hamiltonian.

            Args:
                L (int) : The number of sites of the lattice.
                alpha (int) : A factor to define the size of different machines.
                sr (float) : The parameter for stochastic reconfiguration method. If it is None, stochastic reconfiguration is not used.
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

    if(dataname == None):
        dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname, path=path)
    print('')
    functions.create_machinefile(machine_name, L, alpha, dataname, sr)
    start = time.time()
    gs.run(n_iter=int(n_iterations), out=dataname)
    end = time.time()
    with open(''.join((dataname, '.time')), 'w') as reader:
        reader.write(str(end - start))
    print('Time', end - start)
    sys.stdout.flush()


def load(L=__L__, alpha=__alpha__, sr = None, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', hamiltonian_name = 'transformed_Heisenberg', n_samples =10000, n_iterations = 20):
    """Method to load a pretrained machine and continue the training.

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
    #observables = {**functions.get_operator(hilbert=hi, L=L, operator='FerroCorr', symmetric=False), **functions.get_operator(hilbert=hi, L=L, operator='FerroCorr', symmetric=True)}
    if(sr == None):
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples)#, n_discard=5000)
    else:
        sr = nk.optimizer.SR(ma, diag_shift=sr)
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples, sr=sr)#, n_discard=5000)

    functions.create_machinefile(machine_name, L, alpha, dataname, sr)
    start = time.time()
    #gs2.run(n_iter=n_iterations, out=''.join((dataname, '_load')), obs=observables, write_every=4, save_params_every=4)
    gs2.run(n_iter=n_iterations, out=dataname, write_every=10, save_params_every=10)
    end = time.time()
    with open(''.join((dataname, '.time')), 'a') as reader:
        reader.write(str(end - start))
    print('Time', end - start)
    sys.stdout.flush()


def measureObservable(L=__L__, alpha=__alpha__, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', hamiltonian_name = 'transformed_Heisenberg', n_samples =10000, n_iterations = 20, append = False, operator='FerroCorr'):
    """Method to measure th observables with a trained machine.

            The sampler can be chosen.

                Args:
                    L (int) : The number of sites of the lattice
                    alpha (int) : A factor to define the size of different machines
                    dataname (str) : The dataname. If None, an automatic dataname is chosen
                    path (str) : The directory, where the results are saved. If None, the directory is 'run'
                    machine_name (str) A string to choose the machine. Possible inputs: See get_machine in my_machines.py
                    sampler (str) : A string to choose the sampler: Recommended: 'Local' (this works with every machine)
                    hamiltonian_name (str) : A string to choose the hamiltonian. Possible inputs: see get_hamiltonian in my_models.py
                    n_samples (int) : The number of samples used in every iteration step
                    n_iterations (int) : The number of iterations (training steps)
                    append (bool) : If True, the old .csv file is deleted. If False, the results are appended to the old .csv file
                    operator (str) : allowed inputs are 'FerroCorr', 'StringCorr' and 'S_Z_squared'

                    """
    if (dataname == None):
        dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname, path=path)
    ha, hi, g = models.get_hamiltonian(hamiltonian_name, L)
    generate_machine = machines.get_machine(machine_name)
    ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
    ma.load(''.join((dataname, '.wf')))
    op, sa = machines.load_machine(machine=ma, hamiltonian=ha, optimizer='Adamax', lr=0.001, sampler=sampler)
    observables = functions.get_operator(hilbert=hi, L=L, operator=operator, symmetric=False)
    start = time.time()
    time_per_iteration = 0
    for i in range(n_iterations):
        before = time.time()
        measurement = nk.variational.estimate_expectations(observables, sa, n_samples=n_samples)
        after = time.time()
        time_per_iteration += after - before
        if(i == 0 and append == False):
            w = csv.writer(open(''.join((dataname, '_observables', '.csv')), "w"))
            for key, val in measurement.items():
                w.writerow([key, val])
        else:
            w = csv.writer(open(''.join((dataname, '_observables', '.csv')), "a"))
            for key, val in measurement.items():
                w.writerow([key, val])
        if i%10 == 0:
            time_per_iteration = time_per_iteration / 10
            print('Progress: ', float(i)/n_iterations*100, '%', ';  Time per iteration: ', time_per_iteration)
            sys.stdout.flush()
            time_per_iteration = 0

    end = time.time()
    with open(''.join((dataname, '_observables', '.time')), 'w') as reader:
        reader.write(str(end - start))
    # print(gs2.estimate(observables))
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
    print(v_tmp.shape)
    print('Energy:', w[0], 'Lattice size:', L)
    sys.stdout.flush()

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
            result_l = observable.dot(v).dot(v).real
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
# alpha = int(alpha0 * 0.1)
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
# alpha = int(alpha0 * 0.1)
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
# alpha = int(alpha0 * 0.1)
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



#---------------------------------------------------------------------------------------------------------------------


#Show that the original Heisenberg model and AKLT model can not be solved properly
#run(L=12, alpha=20, machine_name='JaxRBM', sampler='Local', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=200, path='results/problems/RBM')
#run(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=500, path='results/problems/FFNN')
#measureObservable(L=12, alpha=20, machine_name='JaxRBM', sampler='Local', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=20, path='results/problems/RBM', append=False)
#measureObservable(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=100, path='results/problems/FFNN', append=False, operator='StringCorr')
#measureObservable(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=100, path='results/problems/FFNN', append=True, operator='FerroCorr')
#run(L=12, alpha=20, machine_name='JaxRBM', sampler='Local', hamiltonian_name='original_AKLT', n_samples=1000, n_iterations=200, path='results/problemsAKLT/RBM')
#run(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_AKLT', n_samples=1000, n_iterations=500, path='results/problemsAKLT/FFNN')
#measureObservable(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_AKLT', n_samples=1000, n_iterations=100, path='results/problemsAKLT/FFNN', append=False, operator='FerroCorr')
#measureObservable(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_AKLT', n_samples=1000, n_iterations=100, path='results/problemsAKLT/FFNN', append=True, operator='StringCorr')



#Test VBSSampler and InverseSampler
#run(L=12, alpha=20, machine_name='JaxFFNN', sampler='Inverse', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=1000, path='results/InverseSampler2/FFNN')
#measureObservable(L=12, alpha=20, machine_name='JaxFFNN', sampler='Local', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=20, path='results/InverseSampler/FFNN', append=False)
#run(L=12, alpha=20, machine_name='JaxFFNN', sampler='Inverse', hamiltonian_name='original_Heisenberg', n_samples=1000, n_iterations=1000, path='results/VBSSampler2/FFNN')


#measure S_Z_squared for original Hamiltonians
#measureObservable(L=12, path='results/problems/FFNN', machine_name='JaxFFNN', hamiltonian_name='transformed_Heisenberg', n_samples=1000, n_iterations=100, append=True, operator='S_Z_squared')
#measureObservable(L=12, path='results/problemsAKLT/FFNN', machine_name='JaxFFNN', hamiltonian_name='transformed_AKLT', n_samples=1000, n_iterations=100, append=True, operator='S_Z_squared')


#Transformed machine
#jax.config.update('jax_disable_jit', True)
#run(L=4, alpha=2, n_samples=50, n_iterations=100, machine_name='JaxTransformedFFNN', sampler='Local', hamiltonian_name='original_Heisenberg', path='results/TransformedFFNN')