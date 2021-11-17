from dmrgpy import spinchain
import dmrgpy.multioperator as mo
import numpy as np
import os

__L__=6


def get_original_Haldanechain(L = __L__):
  #original Heisenberg
  spins = ["S=1" for i in range(L)] # S=1 chain
  sc = spinchain.Spin_Chain(spins) # create spin chain object
  H = 0 # initialize Hamiltonian
  for i in range(len(spins)-1):
    H += + sc.Sx[i]*sc.Sx[i+1]
    H += + sc.Sy[i]*sc.Sy[i+1]
    H += + sc.Sz[i]*sc.Sz[i+1]
  sc.set_hamiltonian(H)
  return sc


def get_transformed_Haldanechain(L=__L__):
  #transformed Heisenberg
  spins = ["S=1" for i in range(L)] # S=1 chain
  sc2 = spinchain.Spin_Chain(spins) # create spin chain object
  H = 0 # initialize Hamiltonian
  for i in range(len(spins)-1):
    h = 0
    expSx = -2 * sc2.Sx[i+1]*sc2.Sx[i+1] + mo.obj2MO(1)
    expSz = -2 * sc2.Sz[i] * sc2.Sz[i] + mo.obj2MO(1)
    h += sc2.Sx[i]*sc2.Sx[i+1]
    h += sc2.Sy[i]*expSz*expSx*sc2.Sy[i+1]
    h += sc2.Sz[i]*sc2.Sz[i+1]
    H += h
  sc2.set_hamiltonian(H)
  return sc2



def get_original_AKLT(L=__L__):
  #original AKLT
  spins = ["S=1" for i in range(L)]  # S=1 chain
  sc = spinchain.Spin_Chain(spins)  # create spin chain object
  H = 0  # initialize Hamiltonian
  for i in range(len(spins) - 1):
    h = 0
    h += + sc.Sx[i] * sc.Sx[i + 1]
    h += + sc.Sy[i] * sc.Sy[i + 1]
    h += + sc.Sz[i] * sc.Sz[i + 1]
    H += h + 1./3 * h*h
  sc.set_hamiltonian(H)
  return sc



def get_transformed_AKLT(L=__L__):
  #transformed AKLT
  spins = ["S=1" for i in range(L)] # S=1 chain
  sc2 = spinchain.Spin_Chain(spins) # create spin chain object
  H = 0 # initialize Hamiltonian
  for i in range(len(spins)-1):
    h = 0
    expSx = -2 * sc2.Sx[i+1]*sc2.Sx[i+1] + mo.obj2MO(1)
    expSz = -2 * sc2.Sz[i] * sc2.Sz[i] + mo.obj2MO(1)
    h += sc2.Sx[i]*sc2.Sx[i+1]
    h += sc2.Sy[i]*expSz*expSx*sc2.Sy[i+1]
    h += sc2.Sz[i]*sc2.Sz[i+1]
    H += h + 1./3 * h*h
  sc2.set_hamiltonian(H)
  return sc2


for l in [12, 14, 16, 30, 40, 50]:
  __L__ = l
  tH = (get_transformed_Haldanechain(L=__L__), 'transformed_Heisenberg')
  tA = (get_transformed_AKLT(L=__L__), 'transformed_AKLT')
  for sc, sc_name in [tH, tA]:
    #sc = get_transformed_Haldanechain(L=__L__)
    es = sc.get_excited(n=1,mode="DMRG")
    print(es)
    #gap = es[1]-es[0] # compute gap
    #print("Gap of the Haldane chain",gap)
    #print('GS-Energy', es[0])
    print('GS-Energy', es)

    path = 'run/DMRG'
    dataname = ''.join(('DMRG_Energy_', str(__L__), '_', sc_name, '.csv'))
    try:
      os.makedirs(path)
    except OSError:
      print("Creation of the directory %s failed" % path)
    else:
      print("Successfully created the directory %s" % path)
    dataname = '/'.join((path, dataname))

    # save to csv file
    np.savetxt(dataname, np.asarray([es]), delimiter=';')

    #Measure StringCorr from site 0 to site i
    cs = [sc.vev(sc.Sz[0]*sc.Sz[i]).real for i in range(__L__)]
    print(cs)
    dataname = ''.join(('DMRG_StringCorr_', str(__L__), '_', sc_name, '.csv'))

    #Measure Sz^2 on each site
    cs2 = [sc.vev(sc.Sz[i] * sc.Sz[i]).real for i in range(__L__)]
    print(cs2)
    dataname2 = ''.join(('DMRG_Sz2_', str(__L__), '_', sc_name, '.csv'))


    try:
      os.makedirs(path)
    except OSError:
      print("Creation of the directory %s failed" % path)
    else:
      print("Successfully created the directory %s" % path)
    dataname = '/'.join((path, dataname))
    dataname2 = '/'.join((path, dataname2))

    # save to csv file
    np.savetxt(dataname, np.asarray(cs), delimiter=';')
    np.savetxt(dataname2, np.asarray(cs2), delimiter=';')

