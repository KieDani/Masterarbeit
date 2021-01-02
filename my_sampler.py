import jax
import jax.numpy as jnp
import netket as nk


class _JaxVBSKernel:
    def __init__(self, local_states, size):
        self.local_states = jax.numpy.sort(jax.numpy.array(local_states))
        self.size = size
        self.n_states = self.local_states.size

    def transition(self, key, state):

        keys = jax.random.split(key, 2)
        si = jax.random.randint(keys[0], shape=(1,), minval=0, maxval=self.size)
        rs = jax.random.randint(keys[1], shape=(1,), minval=0, maxval=self.n_states - 1)

        return jax.ops.index_update(
            state, si, self.local_states[rs + (self.local_states[rs] >= state[si])]
        )

    def random_state(self, key, state):
        #val = (new_key, rand_qnum, save_direction)
        def helper_while_body(val):
            new_key = val[0]
            new_key = jax.random.split(new_key, 1)[0]
            #val[1] = 2 * jax.random.randint(new_key, shape=(1,), minval=-1, maxval=2)[0]
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
        # if(jnp.logical_or(rand_qnum > 1., rand_qnum < -1.)):
        #     save_direction = rand_qnum
        save_direction = jax.lax.cond(jnp.logical_or(rand_qnum > 1., rand_qnum < -1.), lambda xTrue: xTrue[0], lambda xFalse: xFalse[1], (rand_qnum, save_direction))

        for i in range(1, self.size):
            rand_qnum = 2 * jax.random.randint(keys[i], shape=(1,), minval=-1, maxval=2)[0]
            #print('rand_qnum', rand_qnum)
            new_key = keys[i]
            # while(jnp.logical_or(jnp.logical_and(rand_qnum > 1., save_direction > 1.), jnp.logical_and(rand_qnum < -1., save_direction < -1.))):
            #     new_key = jax.random.split(new_key, 1)[0]
            #     #print('new key', new_key)
            #     rand_qnum = 2 * jax.random.randint(new_key, shape=(1,), minval=-1, maxval=2)[0]
            #     #print(rand_qnum)
            new_key, rand_qnum, save_direction = jax.lax.while_loop(helper_while_cond, helper_while_body, (new_key, rand_qnum, save_direction))
            # if (jnp.logical_or(rand_qnum > 1., rand_qnum < -1.)):
            #     save_direction = rand_qnum
            save_direction = jax.lax.cond(jnp.logical_or(rand_qnum > 1., rand_qnum < -1.), lambda xTrue: xTrue[0],
                                          lambda xFalse: xFalse[1], (rand_qnum, save_direction))
            state = jax.ops.index_update(state, jax.ops.index[i], rand_qnum)
            # print('state', state)
        #print('state', state)



        return keys[0], state

def getVBSSampler(machine):
    kernel = _JaxVBSKernel(local_states=machine.hilbert._local_states, size=machine.hilbert._size)
    sampler = nk.sampler.jax.MetropolisHastings(machine, kernel, n_chains=16, sweep_size=1)
    return sampler