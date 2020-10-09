import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time


__L__ = 50
__number_samples__ = 1000
__number_iterations__ = 200

def run(L=__L__):
    ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
    ha_orig, hi_orig, g_orig = models.build_Heisenbergchain_S1(L=L)
    ma, op, sa = machines.JaxFFNN(hilbert=hi, alpha=4, optimizer='Sgd', lr=0.1)

    #TODO: check, why Lanczos does not work for transformed Hamiltonian
    exact_energy = functions.Lanczos(hamilton=ha_orig, L=L)

    #TODO automaticaly add or remove SR
    sr = nk.optimizer.SR(ma, diag_shift=0.5)

    gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__)#, sr=sr)
    observables = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr')
    #observables_slow = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr_slow')
    #observables = {**observables, **observables_slow}
    dataname = ''.join(('L', str(L)))
    dataname = functions.create_path(dataname)
    print('')
    start = time.time()
    #TODO remove observables from run-method -> Greater speed! But adjust plot to ..._estim.log file!
    gs.run(n_iter=__number_iterations__, out=dataname)#, obs=observables)
    end = time.time()
    print(end - start)

    # print('Estimated results:')
    # gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=20000)
    # gs2.run(n_iter=20, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
    # print(gs2.estimate(observables))


#ensure, that the machine is the same as used before!
def load(dataname=None , L=__L__):
    if (dataname == None):
        dataname = ''.join(('L', str(L)))
        dataname = functions.create_path(dataname)
    ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=L)
    print('load the machine: ', dataname)
    ma, op, sa = machines.JaxFFNN(hilbert=hi, alpha=4)
    ma.load(''.join((dataname, '.wf')))
    op, sa = machines.load_machine(machine=ma, optimizer='Sgd', lr=0.01)
    observables = functions.get_operator(hilbert=hi, L=L, operator='FerroCorr')

    print('Estimated results:')
    gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=20000, n_discard=5000)
    gs2.run(n_iter=20, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
    print(gs2.estimate(observables))



#run(L=12)
#load(L=12)