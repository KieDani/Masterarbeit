"""Definition of some hamiltonians and the correspondending graphs and hilbert spaces

Here, some hamiltonians can be loaded together with the graph and hilbert space.
So far, the Heisnbergchain and the AKLT model are implemented.
Use the method get_hamiltonian to create hamiltonian, graph and hilbert space.

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch, tqdm, matplotlib
This file contains the following functions:

    * build_Heisenbergchain_S1
    * build_Heisenbergchain_S1_transformed
    * build_AKLTchain
    * build_AKLTchain_transformed
    * get_hamiltonian
"""
import netket as nk
import numpy as np
import scipy
import networkx as nx
import sys


def build_Heisenbergchain_S1(L, periodic = False):
    """Loading the Heisenberg chain
        The original Heisenberg chain is created

             Args:
                L (int) : The number of sites of the lattice
                periodic (bool) : True, if we have a periodic lattice. False, if we have an open lattice.
                """
    print('Building the normal S=1 Heisenberg chain')
    J = [1]
    gnx = nx.Graph()
    if(periodic == False):
        tmp = -1
    else:
        tmp = 0
    for i in range(L + tmp):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, N=g.n_nodes)

    # Pauli Matrices for Spin 1
    sigmax = 1. / np.sqrt(2) * np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    sigmaz = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sigmay = 1. / np.sqrt(2) * np.asarray([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])

    # Bond Operator
    interaction = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)

    bond_operator = [
        (J[0] * interaction).tolist(),
    ]

    # Custom Graph Hamiltonian operator
    ha = nk.operator.GraphOperator(hi, g, bond_ops=bond_operator)

    return ha, hi, g


#TODO check, if the results are correct
def build_AKLTchain(L, periodic = False):
    """Loading the AKLT chain
        The original AKLT chain is loaded

            Args:
                L (int) : The number of sites of the lattice
                periodic (bool) : True, if we have a periodic lattice. False, if we have an open lattice.
                """
    print('Building the normal AKLT chain')
    J = [1]
    gnx = nx.Graph()
    if (periodic == False):
        tmp = -1
    else:
        tmp = 0
    for i in range(L + tmp):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, N=g.n_nodes)

    # Pauli Matrices for Spin 1
    sigmax = 1. / np.sqrt(2) * np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    sigmaz = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sigmay = 1. / np.sqrt(2) * np.asarray([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])

    linearterm = np.kron(sigmaz, sigmaz) + np.kron(sigmax, sigmax) + np.kron(sigmay, sigmay)

    # Bond Operator
    interaction = linearterm + 1./3 * np.dot(linearterm, linearterm)

    bond_operator = [
        (J[0] * interaction).tolist(),
    ]

    # Custom Graph Hamiltonian operator
    ha = nk.operator.GraphOperator(hi, g, bond_ops=bond_operator)

    return ha, hi, g


def build_Heisenbergchain_S1_transformed(L, periodic = False):
    """Loading the transformed Heisenberg chain. See https://doi.org/10.1007/BF02097239
        The transformed model is easier to solve with NetKet.

            Args:
                L (int) : The number of sites of the lattice
                periodic (bool) : True, if we have a periodic lattice. False, if we have an open lattice.
                """
    print('Building the transformed S=1 Heisenberg chain')
    J = [1]
    gnx = nx.Graph()
    if (periodic == False):
        tmp = -1
    else:
        tmp = 0
    for i in range(L + tmp):
        gnx.add_edge(i, (i + 1) % L)
        print(str(i) + ' ' + str(i + 1))
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, N=g.n_nodes)

    interaction = np.asarray([[-1, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, -1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, -1, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, -1]])

    bond_operator = [
        (J[0] * interaction).tolist(),
    ]

    # Custom Graph Hamiltonian operator
    ha = nk.operator.GraphOperator(hi, g, bond_ops=bond_operator)

    return ha, hi, g


#TODO check, if the results are correct
def build_AKLTchain_transformed(L, periodic = False):
    """Loading the transformed Heisenberg chain. See https://doi.org/10.1007/BF02097239
        The transformed model is easier to solve with NetKet.

            Args:
                L (int) : The number of sites of the lattice
                periodic (bool) : True, if we have a periodic lattice. False, if we have an open lattice.
                """
    print('Building the normal AKLT chain')
    J = [1]
    gnx = nx.Graph()
    if (periodic == False):
        tmp = -1
    else:
        tmp = 0
    for i in range(L + tmp):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, N=g.n_nodes)

    # Pauli Matrices for Spin_size_1_t 1
    sigmax = 1. / np.sqrt(2) * np.asarray([[0, 1, 0], [1, 0, 1], [0, 1, 0]])
    sigmaz = np.asarray([[1, 0, 0], [0, 0, 0], [0, 0, -1]])
    sigmay = 1. / np.sqrt(2) * np.asarray([[0, -1j, 0], [1j, 0, -1j], [0, 1j, 0]])
    expX = np.asarray([[0, 0, -1], [0, -1, 0], [-1, 0, 0]])
    expZ = np.asarray([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])

    h = -np.kron(sigmaz, sigmaz) - np.kron(sigmax, sigmax) + np.kron(np.dot(sigmay, expZ), np.dot(expX, sigmay))

    # Bond Operator
    interaction = h + 1./3 * np.dot(h, h)

    bond_operator = [
        (J[0] * interaction).tolist(),
    ]

    # Custom Graph Hamiltonian operator
    ha = nk.operator.GraphOperator(hi, g, bond_ops=bond_operator)

    return ha, hi, g


def get_hamiltonian(hamiltonian_name, L, periodic = False):
    """Method to choose the desired model.
        Mutiple models can be easily chosen.

            Args:
                L (int) : The number of sites of the lattice
                periodic (bool) : True, if we have a periodic lattice. False, if we have an open lattice.
                hamiltonian_name (str) : Possible Inputs are 'transformed_Heisenberg', 'original_heisenberg',
                        'transformed_AKLT', 'original_AKLT'
                """
    if(hamiltonian_name == 'transformed_Heisenberg'):
        return build_Heisenbergchain_S1_transformed(L, periodic)
    elif(hamiltonian_name == 'original_Heisenberg'):
        return build_Heisenbergchain_S1(L, periodic)
    elif(hamiltonian_name == 'transformed_AKLT'):
        return build_AKLTchain_transformed(L, periodic)
    elif(hamiltonian_name == 'original_AKLT'):
        return build_AKLTchain(L, periodic)
    else:
        print('The desired hamiltonian was spelled wrong!')
        sys.stdout.flush()
        return None
