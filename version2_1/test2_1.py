import netket as nk
#import my_models as models
#import my_machines as machines
#import my_operators as operators
#import helping_functions as functions
import scipy
import numpy as np
import time


__L__ = 8
__number_samples__ = 800
__number_iterations__ = 350



def baue_graph_original(L, dimension=1, TotalSz=False):
    # J1-J2 Model Parameters
    J = [1]#, 0.02]
    # Define custom graph
    edge_colors = []
    if(dimension==1):
        for i in range(L-1):
            edge_colors.append([i, (i+1)%L, 1])
        #edge_colors.append([i, (i+2)%L, 2])
    else:
        #D=2
        for i in range(L):
            for j in range(L):
                edge_colors.append([j*L+i, j*L+(i+1)%L, 1])
                #edge_colors.append([j*L+i, j*L+(i+2)%L, 2])
                edge_colors.append([j*L+i, (j+1)%L*L+i, 1])
            #edge_colors.append([j*L+i, (j+2)%L*L+i, 2])

    #print(edge_colors)

    # Define the netket graph object
    g = nk.graph.CustomGraph(edge_colors)

    # Printing out the graph information
    print('This graph has ' + str(g.n_sites) + ' sites')
    print('with the following set of edges: ' + str(g.edges))

    # Spin 1 based Hilbert Space
    if(TotalSz==True):
        hi = nk.hilbert.Spin(s=1, total_sz=0.0, graph=g)
    else:
        hi = nk.hilbert.Spin(s=1, graph=g)

    # Pauli Matrices for Spin 1
    sigmax = 1./np.sqrt(2)*np.asarray([[0, 1, 0], [1,0,1], [0,1,0]])
    sigmaz = np.asarray([[1,0,0], [0,0,0],[0,0,-1]])
    sigmay = 1./np.sqrt(2)*np.asarray([[0, -1j, 0], [1j, 0, -1j], [0,1j,0]])

    # Bond Operator
    interaction = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)
    #interaction = np.asarray([[-1, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, -1]])

    bond_operator = [
        (J[0] * interaction).tolist(),
        #(J[1] * interaction).tolist(),
    ]
    bond_color = [1]#, 2]

    # Custom Graph Hamiltonian operator
    op = nk.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)

    return op, hi, g

def baue_graph_transformed(L, dimension=1, TotalSz=False):
    # J1-J2 Model Parameters
    J = [1]#, 0.02]
    # Define custom graph
    edge_colors = []
    if(dimension==1):
        for i in range(L-1):
            edge_colors.append([i, (i+1)%L, 1])
        #edge_colors.append([i, (i+2)%L, 2])
    else:
        #D=2
        for i in range(L):
            for j in range(L):
                edge_colors.append([j*L+i, j*L+(i+1)%L, 1])
                #edge_colors.append([j*L+i, j*L+(i+2)%L, 2])
                edge_colors.append([j*L+i, (j+1)%L*L+i, 1])
            #edge_colors.append([j*L+i, (j+2)%L*L+i, 2])

    #print(edge_colors)

    # Define the netket graph object
    g = nk.graph.CustomGraph(edge_colors)

    # Printing out the graph information
    print('This graph has ' + str(g.n_sites) + ' sites')
    print('with the following set of edges: ' + str(g.edges))

    # Spin 1 based Hilbert Space
    if(TotalSz==True):
        hi = nk.hilbert.Spin(s=1, total_sz=0.0, graph=g)
    else:
        hi = nk.hilbert.Spin(s=1, graph=g)

    # Pauli Matrices for Spin 1
    sigmax = 1./np.sqrt(2)*np.asarray([[0, 1, 0], [1,0,1], [0,1,0]])
    sigmaz = np.asarray([[1,0,0], [0,0,0],[0,0,-1]])
    sigmay = 1./np.sqrt(2)*np.asarray([[0, -1j, 0], [1j, 0, -1j], [0,1j,0]])

    # Bond Operator
    #interaction = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)
    interaction = np.asarray([[-1, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0], [0, -1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, -1, 0], [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, -1]])

    bond_operator = [
        (J[0] * interaction).tolist(),
        #(J[1] * interaction).tolist(),
    ]
    bond_color = [1]#, 2]

    # Custom Graph Hamiltonian operator
    op = nk.operator.GraphOperator(hi, bondops=bond_operator, bondops_colors=bond_color)

    return op, hi, g

def MultiRBMansatz(L, dimension=1, number_samples=2000, number_iterations=500, methode='Sr', opti='Sgd', sampl='Hamiltonian', Alpha=2):
	diagShift = 0
	if (methode == 'Sr'):
		iterative = True
		diagShift = 0.1
	else:
		iterative = True
		diagShift = 0.1

	if (sampl == 'Exchange'):
		ha, hi, g = baue_graph_transformed(L, dimension, True)
	else:
		ha, hi, g = baue_graph_transformed(L, dimension)
	# RBM ansatz with alpha=1
	ma = nk.machine.RbmMultiVal(alpha=Alpha, hilbert=hi)

	if (opti == 'Adamax'):
		op = nk.optimizer.AdaMax()
	elif (opti == 'Sgd'):
		op = nk.optimizer.Sgd(learning_rate=0.1)
	else:
		op = nk.optimizer.RmsProp(learning_rate=0.1)

	# Sampler
	if (sampl == 'Local'):
		sa = nk.sampler.MetropolisLocal(machine=ma)
	elif (sampl == 'Exact'):
		sa = nk.sampler.ExactSampler(machine=ma)
	elif (sampl == 'Exchange'):
		sa = nk.sampler.MetropolisExchange(machine=ma, graph=g, d_max=1)
	elif (sampl == 'Hop'):
		sa = nk.sampler.MetropolisHop(machine=ma, d_max=int(L / 2))
	else:
		sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

	ma.init_random_parameters(seed=123, sigma=0.01)

	gs = nk.variational.Vmc(
		hamiltonian=ha,
		sampler=sa,
		optimizer=op,
		n_samples=number_samples,
		diag_shift=diagShift,
		use_iterative=iterative)

	start, end = starte_berechnung(dateiname='startingpoint_2_1/L='+str(L), number_iterations=number_iterations, stepSize=1, gs=gs,
								   hilbert=hi, L=L)

	ma.save('startingpoint_2_1/L='+str(L)+'.save')

	if nk.MPI.rank() == 0:
		print('###Multi Value  RBM calculation')
		print('Has', ma.n_par, 'parameters')
		print('The multi value RBM calculation took', end - start, 'seconds')


def baue_StringCorr(hilbert, l):
	hi = hilbert
	# We need to specify the local operators as a matrix acting on a local Hilbert space
	sf = []
	sites = []
	sigmaz = np.asarray([[1,0,0], [0,0,0],[0,0,-1]])
	expsigmaz = np.asarray([[-1,0,0], [0,1,0],[0,0,-1]])
	mszs = np.kron(sigmaz,expsigmaz)
	for i in range(1,l-2):
		mszs = np.kron(mszs, expsigmaz)
	mszs = np.kron(mszs, sigmaz)
	sf.append((mszs).tolist())
	print('Ausmaße der Observable:')
	print(mszs.shape)
	helper = []
	for i in range (0,l):
		helper.append(i)
	#printmpi('Zugehörige Gitterplätze:')
	#printmpi(helper)
	sites.append(helper)
	string_correlation_function = nk.operator.LocalOperator(hi, sf, sites)
	print('Zugehörige Gitterplätze:')
	print(string_correlation_function.acting_on)

	return string_correlation_function


def FerroCorrelationZ_slow(hilbert, j, k):
    hi = hilbert
    # We need to specify the local operators as a matrix acting on a local Hilbert space
    sf = []
    sites = []
    sigmaz = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    bigfatone = np.asarray([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    helper = []
    mszs = np.kron(sigmaz, bigfatone)
    if ((k-j) == 1):
        mszs = np.kron(sigmaz, sigmaz)
        helper = [j, k]
    else:
        for i in range(j+1, k - 1):
            mszs = np.kron(mszs, bigfatone)
        mszs = np.kron(mszs, sigmaz)
        for i in range(j, k+1):
            helper.append(i)
    sf.append((mszs).tolist())
    print(k, ' ', j, ' ', k-j)
    print(helper)
    print('Size of Observable:')
    print(mszs.shape)
    sites.append(helper)
    string_correlation_function = nk.operator.LocalOperator(hi, sf, sites)
    return string_correlation_function


#Hier kann man praktischerweise dann Observablen hinzufügen
def starte_berechnung(dateiname, number_iterations, stepSize, gs, hilbert, L, operators=False):
	if(operators==True):
		for start, j in enumerate([1, 2, 3, 4, 5, int(L / 5.), int(L / 4.), int(L / 2.), int(3 * L / 2.)]):
			for k in range(j + 1, np.minimum(j + 8, L)):
				observ = FerroCorrelationZ_slow(hilbert=hilbert, j=j, k=k)
				name = ''.join((str(j), 'Ferro_correlation_function', str(k - j)))
				gs.add_observable(observ, name)
	start = time.time()
	gs.run(output_prefix=dateiname, n_iter=number_iterations, step_size=stepSize)
	end = time.time()
	return start, end


def lade_RBM(dateiname, L, alpha):
	ha, hi, g = baue_graph_transformed(L=L)
	ma = nk.machine.RbmMultiVal(alpha=alpha, hilbert=hi)
	ma.load(filename=dateiname)

	iterative = True
	diagShift = 0.1

	ha, hi, g = baue_graph_transformed(L, 1)

	op = nk.optimizer.Sgd(learning_rate=0.1)


	sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=ha)

	ma.init_random_parameters(seed=123, sigma=0.01)

	gs = nk.variational.Vmc(
		hamiltonian=ha,
		sampler=sa,
		optimizer=op,
		n_samples=5000,
		diag_shift=diagShift,
		use_iterative=iterative)

	start, end = starte_berechnung(dateiname='startingpoint_2_1/L=' + str(L) + '_estimate', number_iterations=20,
								   stepSize=1, gs=gs,
								   hilbert=hi, L=L, operators=True)






# l=10
# MultiRBMansatz(L=l, number_samples=100, number_iterations=51, Alpha=8)
# lade_RBM(dateiname='startingpoint_2_1/L='+str(l)+'.save', L=l, alpha= 8)

for l in [30, 35, 40, 45, 50, 55, 60]:
	ha, hi, g = baue_graph_original(l, 1)
	ha2, hi2, g2 = baue_graph_transformed(l, 1)

	print('\n')

	if(l <= 12):
		print('Now we do the original Hamiltonian')
		start = time.time()
		exact_result = nk.exact.lanczos_ed(ha, first_n=1, compute_eigenvectors=False)
		exact_gs_energy = exact_result.eigenvalues[0]
		end = time.time()
		print('The exact ground-state energy is E0= ' + str(exact_gs_energy))
		print('Dauer exakte Diagonalisierung: ' + str(end - start))

		print('\n')
		print('Now we do the transformed Hamiltonian')
		start = time.time()
		exact_result = nk.exact.lanczos_ed(ha2, first_n=1, compute_eigenvectors=False)
		exact_gs_energy = exact_result.eigenvalues[0]
		end = time.time()
		print('The exact ground-state energy is E0= ' + str(exact_gs_energy))
		print('Dauer exakte Diagonalisierung: ' + str(end - start))

	print('\n')


	MultiRBMansatz(L=l, number_samples=__number_samples__, number_iterations=__number_iterations__, Alpha=8)
	lade_RBM(dateiname='startingpoint_2_1/L=' + str(l) + '.save', L=l, alpha=8)
