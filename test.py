import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time


__L__ = 50
__number_samples__ = 700
__number_iterations__ = 500
__alpha__ = 4

#use Gd, if Sr == None; otherwise, sr is the diag_shift
def run(L=__L__, alpha=__alpha__, sr = None, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', n_samples = __number_samples__, n_iterations = __number_iterations__):
    ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
    ha_orig, hi_orig, g_orig = models.build_Heisenbergchain_S1(L=L)
    generate_machine = machines.get_machine(machine_name)
    ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha, optimizer='Adamax', lr=0.005, sampler=sampler)

    #TODO: check, why Lanczos does not work for transformed Hamiltonian
    exact_energy = functions.Lanczos(hamilton=ha_orig, L=L)

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
    start = time.time()

    functions.create_machinefile(machine_name, L, alpha, dataname, sr)

    gs.run(n_iter=int(n_iterations), out=dataname)#, obs=observables)

    end = time.time()
    print(end - start)


#ensure, that the machine is the same as used before!
def load(L=__L__, alpha=__alpha__, sr = None, dataname = None, path = 'run', machine_name = 'JaxRBM', sampler = 'Local', n_samples =10000, n_iterations = 20):
    if (dataname == None):
        dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname, path=path)
    ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
    print('load the machine: ', dataname)
    generate_machine = machines.get_machine(machine_name)
    ma, op, sa, machine_name = generate_machine(hilbert=hi, hamiltonian=ha, alpha=alpha)
    ma.load(''.join((dataname, '.wf')))
    op, sa = machines.load_machine(machine=ma, hamiltonian=ha, optimizer='Adamax', lr=0.001, sampler=sampler)
    observables = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr', symmetric=True)

    print('Estimated results:')
    if(sr == None):
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples)#, n_discard=5000)
    else:
        sr = nk.optimizer.SR(ma, diag_shift=sr)
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=n_samples, sr=sr)#, n_discard=5000)

    functions.create_machinefile(machine_name, L, alpha, dataname, sr)
    gs2.run(n_iter=n_iterations, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
    print(gs2.estimate(observables))



#run(L=5, alpha=10, sr=0.01, path='test_sr', dataname='test_sr', n_samples=300, n_iterations=50, machine_name='JaxFFNN')
#load(L=5, alpha=10, sr=0.01, path='test_sr', dataname='test_sr', n_samples=3000, n_iterations=20, machine_name='JaxFFNN')


