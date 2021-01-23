"""Implementation of custom Monte-Carlo sampler

Implementation of Monte-Carlo Kernels that define the update process of the MCMC.
It is used with netket.sampler.jax.MetropolisHastings.

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch, tqdm, matplotlib

This file contains the following classes:

    * _JaxVBSKernel

This file contains the following functions:

    *  getVBSSampler
"""
import jax
import jax.numpy as jnp
import netket as nk


class _JaxVBSKernel:
    """A Monte Carlo Kernel to use with netket.sampler.jax.MetropolisHastings.

    Local spinflips are performed and with a propbability of 20% the state is repaired, so that it is a VBS state again.
    It only works with Spin-1 hilbert spaces.
    """
    def __init__(self, local_states, size):
        self.local_states = jax.numpy.sort(jax.numpy.array(local_states))
        self.size = size
        self.n_states = self.local_states.size

    def transition(self, key, state):
        """Here, the update of a state is performed"""
        def local_update(val):
            key = val[0]
            state = val[1]
            keys = jax.random.split(key, 2)
            si = jax.random.randint(keys[0], shape=(1,), minval=0, maxval=self.size)
            rs = jax.random.randint(keys[1], shape=(1,), minval=0, maxval=self.n_states - 1)
            return jax.ops.index_update(
                state, si, self.local_states[rs + (self.local_states[rs] >= state[si])]
            )
        def VBS_update(val):
            key = val[0]
            state = val[1]

            #first a local spinflip
            keys = jax.random.split(key, 2)
            state = local_update((keys[0], state))

            save_direction = 0.
            save_direction = jax.lax.cond(jnp.logical_or(state[0] > 1., state[0] < -1.), lambda xTrue: xTrue[0],
                                          lambda xFalse: xFalse[1], (state[0], save_direction))
            rand_flips = jax.random.randint(keys[1], shape=(self.size,), minval=-1, maxval=1)
            for i in range(1, self.size):
                q_num = state[i]
                condition = jnp.logical_or(jnp.logical_and(q_num > 1., save_direction > 1.), jnp.logical_and(q_num < -1., save_direction < -1.))
                #flips spin i or changes it to 0, if spin i breaks the spin order.
                new_q_num = jax.lax.cond(condition, lambda xTrue: xTrue[0]*xTrue[1], lambda xFalse: xFalse[1], (rand_flips[i], q_num))
                #if spin i was flipped, save_direction is updated
                save_direction = jax.lax.cond(q_num*new_q_num < -0.5, lambda xTrue: xTrue[0], lambda xFalse: xFalse[1], (new_q_num, save_direction))
                state = jax.ops.index_update(state, jax.ops.index[i], new_q_num)
            return state

        keys = jax.random.split(key, 2)
        rand_num = jax.random.randint(keys[0], shape=(1,), minval=1, maxval=11)[0]
        #print(rand_num)
        return jax.lax.cond(rand_num < 9, local_update, VBS_update , (keys[1], state))


    def random_state(self, key, state):
        """Here, a random VBS state is created."""
        def helper_while_body(val):
            new_key = val[0]
            new_key = jax.random.split(new_key, 1)[0]
            return (new_key, 2 * jax.random.randint(new_key, shape=(1,), minval=-1, maxval=2)[0], val[2])
        def helper_while_cond(val):
            return (jnp.logical_or(jnp.logical_and(val[1] > 1., val[2] > 1.), jnp.logical_and(val[1] < -1., val[2] < -1.)))

        keys = jax.random.split(key, self.size)
        state = jnp.empty(shape=(self.size,))
        rand_qnum = 2 * jax.random.randint(keys[0], shape=(1,), minval=-1, maxval=2)[0]
        #print('rand_qnum', rand_qnum)
        state = jax.ops.index_update(state, jax.ops.index[0], rand_qnum)
        #print('state', state)
        save_direction = 0
        save_direction = jax.lax.cond(jnp.logical_or(rand_qnum > 1., rand_qnum < -1.), lambda xTrue: xTrue[0], lambda xFalse: xFalse[1], (rand_qnum, save_direction))

        for i in range(1, self.size):
            rand_qnum = 2 * jax.random.randint(keys[i], shape=(1,), minval=-1, maxval=2)[0]
            #prikeys = jax.random.split(key, 2)nt('rand_qnum', rand_qnum)
            new_key = keys[i]
            # while(jnp.logical_or(jnp.logical_and(rand_qnum > 1., save_direction > 1.), jnp.logical_and(rand_qnum < -1., save_direction < -1.))):
            #     new_key = jax.random.split(new_key, 1)[0]
            #     #print('new key', new_key)
            #     rand_qnum = 2 * jax.random.randint(new_key, shape=(1,), minval=-1, maxval=2)[0]
            #     #print(rand_qnum)
            new_key, rand_qnum, save_direction = jax.lax.while_loop(helper_while_cond, helper_while_body, (new_key, rand_qnum, save_direction))
            save_direction = jax.lax.cond(jnp.logical_or(rand_qnum > 1., rand_qnum < -1.), lambda xTrue: xTrue[0],
                                          lambda xFalse: xFalse[1], (rand_qnum, save_direction))
            state = jax.ops.index_update(state, jax.ops.index[i], rand_qnum)
            # print('state', state)
        #print('state', state)
        return keys[0], state

def getVBSSampler(machine):
    """Method to easily create a Metropolis Hastings sampler with _JaxVBSKernel.
        The sampler does not solve the problems with the original AKLT and Heisenberg chain

            Args:
                hilbert (netket.machine) : machine

            Returns:
                sampler (netket.sampler) : sampler
                                                        """
    kernel = _JaxVBSKernel(local_states=machine.hilbert._local_states, size=machine.hilbert._size)
    sampler = nk.sampler.jax.MetropolisHastings(machine, kernel, n_chains=16, sweep_size=1)
    return sampler