import netket as nk
import numpy as np
import scipy
import networkx as nx
import sys


def build_Heisenbergchain_S1(L):
    print('Building the normal S=1 Heisenberg chain')
    J = [1]
    gnx = nx.Graph()
    for i in range(L - 1):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, graph=g)

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
    ha = nk.operator.GraphOperator(hi, bond_ops=bond_operator)

    return ha, hi, g


#TODO check, if the results are correct
def build_AKLTchain(L):
    print('Building the normal AKLT chain')
    J = [1]
    gnx = nx.Graph()
    for i in range(L - 1):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, graph=g)

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
    ha = nk.operator.GraphOperator(hi, bond_ops=bond_operator)

    return ha, hi, g


def build_Heisenbergchain_S1_transformed(L):
    print('Building the transformed S=1 Heisenberg chain')
    J = [1]
    gnx = nx.Graph()
    for i in range(L - 1):
        gnx.add_edge(i, (i + 1) % L)
        print(str(i) + ' ' + str(i + 1))
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, graph=g)

    interaction = np.asarray([[-1, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, -1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, -1, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, -1]])

    bond_operator = [
        (J[0] * interaction).tolist(),
    ]

    # Custom Graph Hamiltonian operator
    ha = nk.operator.GraphOperator(hi, bond_ops=bond_operator)

    return ha, hi, g


#TODO check, if the results are correct
def build_AKLTchain_transformed(L):
    print('Building the normal AKLT chain')
    J = [1]
    gnx = nx.Graph()
    for i in range(L - 1):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    #print('This graph has ' + str(g.n_nodes) + ' sites')
    #print('with the following set of edges: ' + str(g.n_edges))

    hi = nk.hilbert.Spin(s=1, graph=g)

    # Pauli Matrices for Spin 1
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
    ha = nk.operator.GraphOperator(hi, bond_ops=bond_operator)

    return ha, hi, g


def get_hamiltonian(hamiltonian_name, L):
    if(hamiltonian_name == 'transformed_Heisenberg'):
        return build_Heisenbergchain_S1_transformed(L)
    elif(hamiltonian_name == 'original_heisenberg'):
        return build_Heisenbergchain_S1(L)
    elif(hamiltonian_name == 'transformed_AKLT'):
        return build_AKLTchain_transformed(L)
    elif(hamiltonian_name == 'original_AKLT'):
        return build_AKLTchain(L)
    else:
        print('The desired hamiltonian was spelled wrong!')
        sys.stdout.flush()
        return None
