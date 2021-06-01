from dmrgpy import spinchain

L=6


spins = ["S=1" for i in range(L)] # S=1 chain
sc = spinchain.Spin_Chain(spins) # create spin chain object
h = 0 # initialize Hamiltonian
for i in range(len(spins)-1):
  h = h + sc.Sx[i]*sc.Sx[i+1]
  h = h + sc.Sy[i]*sc.Sy[i+1]
  h = h + sc.Sz[i]*sc.Sz[i+1]
sc.set_hamiltonian(h)
es = sc.get_excited(n=2,mode="DMRG")
gap = es[1]-es[0] # compute gap
print("Gap of the Haldane chain",gap)
print('GS-Energy', es[0])
#sc.get_dynamical_correlator(name=(sc.Sz[0],sc.Sz[0]))

cs = [sc.vev(sc.Sz[0]*sc.Sz[i]).real for i in range(L)]
print(cs)