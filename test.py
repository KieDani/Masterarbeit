import netket as nk
import my_models as models
import my_machines as machines
import my_operators as operators
import helping_functions as functions

import time


__L__ = 4
__number_samples__ = 1000
__number_iterations__ = 150

ha, hi, g = models.build_Heisenbergchain_S1_transformed(L=__L__)
ma, op, sa = machines.JaxRBM(hilbert=hi, alpha=1, optimizer='Sgd', lr=0.1)

#TODO: check, why Lanczos does not work
#exact_energy = functions.Lanczos(hamilton=ha, L=__L__)

#TODO automaticaly add or remove SR
sr = nk.optimizer.SR(ma, diag_shift=0.5)

gs = nk.Vmc(hamiltonian=ha, sampler=sa, optimizer=op, n_samples=__number_samples__, sr=sr)
gs = functions.add_operator(gs, hilbert=hi, L=__L__, operator=None)
dataname = 'test'
start = time.time()
gs.run(n_iter=__number_iterations__, out='test')
end = time.time()
print(end - start)