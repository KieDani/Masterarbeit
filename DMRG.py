from dmrgpy import spinchain
import dmrgpy.multioperator as mo
import numpy as np
import os

__L__=6


def get_original_Haldanechain(L = __L__):
  #original Heisenberg
  spins = ["S=1" for i in range(L)] # S=1 chain
  sc = spinchain.Spin_Chain(spins) # create spin chain object
  h = 0 # initialize Hamiltonian
  for i in range(len(spins)-1):
    h = h + sc.Sx[i]*sc.Sx[i+1]
    h = h + sc.Sy[i]*sc.Sy[i+1]
    h = h + sc.Sz[i]*sc.Sz[i+1]
  sc.set_hamiltonian(h)
  return sc


def get_transformed_Haldanechain(L=__L__):
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
  return sc2


sc = get_transformed_Haldanechain()
es = sc.get_excited(n=1,mode="DMRG")
print(es)
#gap = es[1]-es[0] # compute gap
#print("Gap of the Haldane chain",gap)
#print('GS-Energy', es[0])
print('GS-Energy', es)

path = 'run/DMRG'
dataname = ''.join(('DMRG_Energy', str(__L__), '.csv'))
try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)
dataname = '/'.join((path, dataname))

# save to csv file
np.savetxt(dataname, np.asarray([es]), delimiter=';')


cs = [sc.vev(sc.Sz[0]*sc.Sz[i]).real for i in range(__L__)]
print(cs)

dataname = ''.join(('DMRG_', str(__L__), '.csv'))

try:
  os.makedirs(path)
except OSError:
  print("Creation of the directory %s failed" % path)
else:
  print("Successfully created the directory %s" % path)
dataname = '/'.join((path, dataname))

# save to csv file
np.savetxt(dataname, np.asarray(cs), delimiter=';')

