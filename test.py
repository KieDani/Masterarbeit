import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time


__L__ = 50
__number_samples__ = 800
__number_iterations__ = 400
__alpha__ = 4

def run(L=__L__, alpha=__alpha__, use_sr = False):
    ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
    ha_orig, hi_orig, g_orig = models.build_Heisenbergchain_S1(L=L)
    ma, op, sa, machine_name = machines.JaxRBM(hilbert=hi, hamiltonian=ha, alpha=alpha, optimizer='Adamax', lr=0.005)

    #TODO: check, why Lanczos does not work for transformed Hamiltonian
    exact_energy = functions.Lanczos(hamilton=ha_orig, L=L)

    if(use_sr == False):
        gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__)#, sr=sr)
    else:
        sr = nk.optimizer.SR(ma, diag_shift=0.1)
        gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__, sr=sr)

    #observables = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr')
    dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname, path='run/startingpoint')
    print('')
    start = time.time()

    functions.create_machinefile(machine_name, L, alpha, dataname)

    gs.run(n_iter=int(__number_iterations__), out=dataname)#, obs=observables)

    # op, sa = machines.load_machine(machine=ma, optimizer='Sgd', lr=0.1)
    # gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__)
    # gs.run(n_iter=int(__number_iterations__/2), out=dataname)  # , obs=observables)

    end = time.time()
    print(end - start)

    # print('Estimated results:')
    # gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=20000)
    # gs2.run(n_iter=20, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
    # print(gs2.estimate(observables))


#ensure, that the machine is the same as used before!
def load(dataname=None , L=__L__, alpha=__alpha__, use_sr = False):
    if (dataname == None):
        dataname = ''.join(('L', str(L)))
        dataname = functions.create_path(dataname, path='run/startingpoint')
    ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
    print('load the machine: ', dataname)
    ma, op, sa, machine_name = machines.JaxRBM(hilbert=hi, hamiltonian=ha, alpha=alpha)
    ma.load(''.join((dataname, '.wf')))
    op, sa = machines.load_machine(machine=ma, hamiltonian=ha, optimizer='Adamax', lr=0.001)
    observables = functions.test_operator_startingpoint(hilbert=hi, L=L)

    print('Estimated results:')
    if(use_sr == False):
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=10000)#, sr=sr)#, n_discard=5000)
    else:
        sr = nk.optimizer.SR(ma, diag_shift=0.1)
        gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=10000, sr=sr)#, n_discard=5000)

    functions.create_machinefile(machine_name, L, alpha, dataname)
    gs2.run(n_iter=20, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
    print(gs2.estimate(observables))



#run(L=12)
#load(L=12)

for l in [10, 20, 30]:
    run(L=l, alpha=8)
    load(L=l, alpha=8)