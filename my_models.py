import netket as nk
import numpy as np
import scipy
import networkx as nx


def build_Heisenbergchain_S1(L):
    print('Building the normal S=1 Heisenberg chain')
    J = [1]
    gnx = nx.Graph()
    for i in range(L - 1):
        gnx.add_edge(i, (i + 1) % L)
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    print('This graph has ' + str(g.n_sites) + ' sites')
    print('with the following set of edges: ' + str(g.edges))

    if (L%2 == 0):
        hi = nk.hilbert.Spin(s=1, total_sz=0.0, graph=g)
    else:
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
    op = nk.operator.GraphOperator(hi, bond_ops=bond_operator)

    return op, hi, g


def build_Heisenbergchain_S1_transformed(L):
    print('Building the transformed S=1 Heisenberg chain')
    J = [1]
    gnx = nx.Graph()
    for i in range(L - 1):
        gnx.add_edge(i, (i + 1) % L)
        print(str(i) + ' ' + str(i + 1))
    g = nk.graph.Graph(nodes=list(gnx.nodes), edges=list(gnx.edges))

    # Printing out the graph information
    print('This graph has ' + str(g.n_sites) + ' sites')
    print('with the following set of edges: ' + str(g.edges))

    if (L%2 == 0):
        hi = nk.hilbert.Spin(s=1, total_sz=0.0, graph=g)
    else:
        hi = nk.hilbert.Spin(s=1, graph=g)

    interaction = np.asarray([[-1, 0, 0, 0, -1, 0, 0, 0, 0], [0, 0, 0, -1, 0, 0, 0, 0, 0], [0, 0, 1, 0, 0, 0, 0, 0, 0],
                              [0, -1, 0, 0, 0, 0, 0, 0, 0], [-1, 0, 0, 0, 0, 0, 0, 0, -1], [0, 0, 0, 0, 0, 0, 0, -1, 0],
                              [0, 0, 0, 0, 0, 0, 1, 0, 0], [0, 0, 0, 0, 0, -1, 0, 0, 0], [0, 0, 0, 0, -1, 0, 0, 0, -1]])

    bond_operator = [
        (J[0] * interaction).tolist(),
    ]

    # Custom Graph Hamiltonian operator
    op = nk.operator.GraphOperator(hi, bond_ops=bond_operator)

    return op, hi, g
