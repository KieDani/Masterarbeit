import itertools
from operator import add
import numpy as np
from quimb import *
import time

def ham_heis_2D(n, m, J=-1.0, bz=0.0, cyclic=False,
                sparse=True, S=1.):

    dims = [2] * m # shape (n, m)

    # generate tuple of all site coordinates
    sites = tuple(itertools.product(range(n), range(m)))
    print('sites: ', sites)

    # generate neighbouring pairs of coordinates
    def gen_pairs():
        for i, j, in sites:
            above, right = (i + 1) % n, (j + 1) % m
            # ignore wraparound coordinates if not cyclic
            if cyclic or above != 0:
                yield ((i, j), (above, j))
            if cyclic or right != 0:
                yield ((i, j), (i, right))

    # generate all pairs of coordinates and directions
    pairs_ss = tuple(itertools.product(gen_pairs(), 'xyz'))
    print('pairs_ss: ', pairs_ss)

    # make XX, YY and ZZ interaction from pair_s
    #     e.g. arg ([(3, 4), (3, 5)], 'z')
    def interactions(pair_s):
        pair, s = pair_s
        print('s: ', s)
        print('pair: ', pair)
        Sxyz = spin_operator(s, S=S, sparse=True)
        print('Sxyz: ', Sxyz)
        result = ikron([J * Sxyz, Sxyz], dims, inds=pair)
        #print('result: ', result)
        return result

    # function to make Z field at ``site``
    def fields(site):
        Sz = spin_operator('z', S=S, sparse=True)
        return ikron(bz * Sz, dims, inds=[site])

    # combine all terms
    all_terms = itertools.chain(map(interactions, pairs_ss),
                                map(fields, sites) if bz != 0.0 else ())
    H = sum(all_terms)

    # can improve speed of e.g. eigensolving if known to be real
    if isreal(H):
        H = H.real

    if not sparse:
        H = qarray(H.A)

    return H

n, m = 1, 12
dims = [2] * m

print(dims)

H = ham_heis_2D(n, m, cyclic=False)

#H = H + 0.2 * ikron(spin_operator('Z', sparse=True), dims, [(1, 2)])


start = time.time()
ge, gs = eigh(H, k=1)
end = time.time()

print('time: ', end-start, 's')

print(ge[0])

# abc = spin_operator('z', S=1.)
#
# print(abc)


