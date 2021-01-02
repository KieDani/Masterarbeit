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
        keys = jax.random.split(key, self.size)

        state = jnp.empty(shape=(self.size,))
        # To expand it to non-Spin-1 problems, use self.local_states
        possibilities = [-2., 0., 2.]

        rand_qnum = possibilities[
            jax.random.randint(keys[0], shape=(1,), minval=0, maxval=len(possibilities))[0]]
        if (rand_qnum > 1):
            up = True
            down = False
        elif (rand_qnum < 1):
            up = False
            down = True
        else:
            up = False
            down = False
        print('rand_qnum', rand_qnum)
        #state[0] = rand_qnum
        state = jax.ops.index_update(state, jax.ops.index[0], rand_qnum)
        print('state', state)

        for i in range(1, self.size):
            rand_qnum = possibilities[
                jax.random.randint(keys[i], shape=(1,), minval=0, maxval=len(possibilities))[0]]
            print('rand_qnum', rand_qnum)
            new_key = keys[i]
            while((rand_qnum > 1 and up == True) or (rand_qnum < -1 and down == True)):
                new_key = jax.random.split(new_key, 1)[0]
                #print('new key', new_key)
                rand_qnum = possibilities[
                    jax.random.randint(new_key, shape=(1,), minval=0, maxval=len(possibilities))[0]]
                print(rand_qnum)
            #state[i] = rand_qnum
            state = jax.ops.index_update(state, jax.ops.index[i], rand_qnum)
            print('state', state)
            if (rand_qnum > 1):
                up = True
                down = False
            elif (rand_qnum < 1):
                up = False
                down = True
            else:
                up = False
                down = False



        return keys[0], state

    def getVBSSampler(self, machine):
        sa = nk.sampler.jax._JaxMetropolisHastings(machine=machine, kernel= )