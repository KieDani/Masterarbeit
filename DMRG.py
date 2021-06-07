from dmrgpy import spinchain
import dmrgpy.multioperator as mo

L=40


#original Heisenberg
spins = ["S=1" for i in range(L)] # S=1 chain
sc = spinchain.Spin_Chain(spins) # create spin chain object
h = 0 # initialize Hamiltonian
for i in range(len(spins)-1):
  h = h + sc.Sx[i]*sc.Sx[i+1]
  h = h + sc.Sy[i]*sc.Sy[i+1]
  h = h + sc.Sz[i]*sc.Sz[i+1]
sc.set_hamiltonian(h)


#transformed Heisenberg
spins = ["S=1" for i in range(L)] # S=1 chain
sc2 = spinchain.Spin_Chain(spins) # create spin chain object
h = 0 # initialize Hamiltonian
for i in range(len(spins)-1):
  expSx = -2* sc2.Sx[i+1]*sc2.Sx[i+1] + mo.obj2MO(1)
  expSz = -2 * sc2.Sz[i] * sc2.Sz[i] + mo.obj2MO(1)
  h = h + sc2.Sx[i]*sc2.Sx[i+1]
  h = h + sc2.Sy[i]*expSz*expSx*sc2.Sy[i+1]
  h = h + sc2.Sz[i]*sc2.Sz[i+1]
sc2.set_hamiltonian(h)


print(sc)
print(sc.Sx)
print(mo.write(sc.Sz[1], 'dummy.txt'))

print(sc2)
print(sc2.Sx)
print(mo.write(sc2.Sz[1], 'dummy.txt'))


es = sc.get_excited(n=2,mode="DMRG")
gap = es[1]-es[0] # compute gap
print("Gap of the Haldane chain",gap)
print('GS-Energy', es[0])


es2 = sc2.get_excited(n=2,mode="DMRG")
gap = es2[1]-es2[0] # compute gap
print("Gap of the Haldane chain",gap)
print('GS-Energy', es2[0])

#sc.get_dynamical_correlator(name=(sc.Sz[0],sc.Sz[0]))

cs = [sc.vev(sc.Sz[0]*sc.Sz[i]).real for i in range(L)]
print(cs)

cs2 = [sc2.vev(sc2.Sz[0]*sc2.Sz[i]).real for i in range(L)]
print(cs2)