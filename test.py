import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time


__L__ = 50
__number_samples__ = 1000
__number_iterations__ = 1000

ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=__L__)
ha_orig, hi_orig, g_orig = models.build_Heisenbergchain_S1(L=__L__)
ma, op, sa = machines.JaxFFNN(hilbert=hi, alpha=4, optimizer='Sgd', lr=0.1)

#TODO: check, why Lanczos does not work for transformed Hamiltonian
exact_energy = functions.Lanczos(hamilton=ha_orig, L=__L__)

#TODO automaticaly add or remove SR
sr = nk.optimizer.SR(ma, diag_shift=0.5)

gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__)#, sr=sr)
observables = functions.get_operator(hilbert=hi, L=__L__, operator='FerroCorr')
#observables_slow = functions.get_operator(hilbert=hi, L=__L__, operator='FerroCorr_slow')
#observables = {**observables, **observables_slow}
dataname = ''.join(('L', str(__L__)))
dataname = functions.create_path(dataname)
print('')
start = time.time()
#TODO remove observables from run-method -> Greater speed! But adjust plot to ..._estim.log file!
gs.run(n_iter=__number_iterations__, out=dataname)#, obs=observables)
end = time.time()
print(end - start)

print('Estimated results:')
gs2 = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=20000)
gs2.run(n_iter=20, out=''.join((dataname, '_estimate')), obs=observables, write_every=4, save_params_every=4)
print(gs2.estimate(observables))