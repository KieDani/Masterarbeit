import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time


__L__ = 10
__number_samples__ = 1000
__number_iterations__ = 250

ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=__L__)
ha_orig, hi_orig, g_orig = models.build_Heisenbergchain_S1(L=__L__)
ma, op, sa = machines.JaxFFNN(hilbert=hi, alpha=2, optimizer='Sgd', lr=0.1)

#TODO: check, why Lanczos does not work for transformed Hamiltonian
exact_energy = functions.Lanczos(hamilton=ha_orig, L=__L__)

#TODO automaticaly add or remove SR
sr = nk.optimizer.SR(ma, diag_shift=0.5)

gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__, sr=sr)
gs = functions.add_operator(gs, hilbert=hi, L=__L__, operator='FerroCorr')
dataname = 'L10'
dataname = functions.create_path(dataname)
print('')
start = time.time()
gs.run(n_iter=__number_iterations__, out=dataname)
end = time.time()
print(end - start)