import itertools
from operator import add
import numpy as np
import quimb
from quimb import *
import time
import functools

def ham_heis_2D(n, m, J=1.0, bz=0.0, cyclic=False,
                sparse=True, S=1.):

    dims = (3,) * m # shape (n, m)

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


def hamiltonian_builder(fn):
    """Wrap a function to perform some generic postprocessing and take the
    kwargs ``stype`` and ``sparse``. This assumes the core function always
    builds the hamiltonian in sparse form. The wrapper then:

    1. Checks if the operator is real and, if so, discards imaginary part if no
       explicity `dtype` was given
    2. Converts the operator to dense or the correct sparse form
    3. Makes the operator immutable so it can be safely cached
    """

    @functools.wraps(fn)
    def ham_fn(*args, stype='csr', sparse=False, **kwargs):
        H = fn(*args, **kwargs)

        if kwargs.get('dtype', None) is None and isreal(H):
            H = H.real

        if not sparse:
            H = qarray(H.A)
        elif H.format != stype:
            H = H.asformat(stype)

        quimb.core.make_immutable(H)

        return H

    return ham_fn


@functools.lru_cache(maxsize=8)
@hamiltonian_builder
def ham_heis(n, j=1.0, b=0.0, cyclic=False,
             parallel=False, nthreads=None, ownership=None):
    """Constructs the nearest neighbour 1d heisenberg spin-1/2 hamiltonian.
    Parameters
    ----------
    n : int
        Number of spins.
    j : float or tuple(float, float, float), optional
        Coupling constant(s), with convention that positive =
        antiferromagnetic. Can supply scalar for isotropic coupling or
        vector ``(jx, jy, jz)``.
    b : float or tuple(float, float, float), optional
        Magnetic field, defaults to z-direction only if tuple not given.
    cyclic : bool, optional
        Whether to couple the first and last spins.
    sparse : bool, optional
        Whether to return the hamiltonian in sparse form.
    stype : str, optional
        What format of sparse operator to return if ``sparse``.
    parallel : bool, optional
        Whether to build the operator in parallel. By default will do this
        for n > 16.
    nthreads : int optional
        How mny threads to use in parallel to build the operator.
    ownership : (int, int), optional
        If given, which range of rows to generate.
    kwargs
        Supplied to :func:`~quimb.core.quimbify`.
    Returns
    -------
    H : immutable operator
        The Hamiltonian.
    """
    dims = (3,) * n
    S = 1.
    try:
        jx, jy, jz = j
    except TypeError:
        jx = jy = jz = j

    try:
        bx, by, bz = b
    except TypeError:
        bz = b
        bx = by = 0.0

    parallel = (n > 16) if parallel is None else parallel

    op_kws = {'sparse': True, 'stype': 'coo'}
    ikron_kws = {'sparse': True, 'stype': 'coo',
                 'coo_build': True, 'ownership': ownership}

    # The basic operator (interaction and single b-field) that can be repeated.
    two_site_term = sum(
        j * kron(spin_operator(s, S=S, **op_kws), spin_operator(s, S=S, **op_kws))
        for j, s in zip((jx, jy, jz), 'xyz')
    ) - sum(
        b * kron(spin_operator(s, S=S, **op_kws), eye(2, **op_kws))
        for b, s in zip((bx, by, bz), 'xyz') if b != 0.0
    )

    single_site_b = sum(-b * spin_operator(s, S=S, **op_kws)
                        for b, s in zip((bx, by, bz), 'xyz') if b != 0.0)

    def gen_term(i):
        # special case: the last b term needs to be added manually
        if i == -1:
            return ikron(single_site_b, dims, n - 1, **ikron_kws)

        # special case: the interaction between first and last spins if cyclic
        if i == n - 1:
            return sum(
                j * ikron(spin_operator(s, S=S, **op_kws),
                          dims, [0, n - 1], **ikron_kws)
                for j, s in zip((jx, jy, jz), 'xyz') if j != 0.0)

        # General term, on-site b-field plus interaction with next site
        return ikron(two_site_term, dims, [i, i + 1], **ikron_kws)

    terms_needed = range(0 if not any((bx, by, bz)) else -1,
                         n if cyclic else n - 1)

    if parallel:
        pool = get_thread_pool(nthreads)
        ham = quimb.core.par_reduce(operator.add, pool.map(gen_term, terms_needed))
    else:
        ham = sum(map(gen_term, terms_needed))

    return ham


n, m = 1, 4
#dims = [2] * m

#print(dims)

#H = ham_heis_2D(n, m, cyclic=False)

H = ham_heis(9)

#H = H + 0.2 * ikron(spin_operator('Z', sparse=True), dims, [(1, 2)])


start = time.time()
ge, gs = eigh(H, k=1)
end = time.time()

print('time: ', end-start, 's')

print(ge[0])

# abc = spin_operator('z', S=1.)
#
# print(abc)


