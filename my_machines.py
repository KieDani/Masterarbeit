"""Implementation of different Neural Networks

Some (parts of) Neural Networks are implemented. The Networks (and the optimizer and sampler) can be easily loaded
using the function get_machine.
If a pretrained machine is loaded, you get an optimizer and sampler using the function load_machine

This project requires the following libraries:
netket, numpy, scipy, jax, jaxlib, networkx, torch

This file contains the following functions:

    * get_machine
    * load_machine
    * JaxRBM
    * JaxSymmRBM
    * JaxUnaryRBM
    * JaxFFNN
    * JaxResFFNN
    * JaxUnaryFFNN
    * JaxSymmFFNN
    * JaxConv3NN
    * JaxResConvNN
    * JaxDeepFFNN
    * TorchFFNN
    * TorchConvNN
    * logcosh
    * modrelu
    * complexrelu
    * SumLayer
    * FormatLayer
    * InputForConvLayer
    * FixSrLayer
    * InputForDenseLayer
    * PaddingLayer
    * ResFFLayer
    * ResConvLayer
    * UnaryLayer

This file contains the following classes:

    * Torch_FFNN_model
    * Torch_ConvNN_model
    * Torch_Conv1d_Layer
"""
import netket as nk
import numpy as np
import torch
import jax
from jax.experimental.stax import Dense, Relu, LogSoftmax, Dropout
from jax.experimental import stax
from netket.optimizer import Torch
from torch.optim import SGD, Adam, Adamax
from netket.optimizer.jax import Wrap
from jax.experimental.optimizers import sgd as SgdJax
from jax.experimental.optimizers import adam as AdamJax
from jax.experimental.optimizers import adamax as AdaMaxJax
import jax.numpy as jnp
from torch.nn import functional as F
from torch.nn.modules.conv import _ConvNd, Conv1d
from torch import Tensor
from torch.nn.modules.utils import _single
from torch.nn.common_types import _size_1_t
import functools
import sys
import my_sampler





@jax.jit
def logcosh(x):
    """logcosh activation function. To use this function as layer, use LogCoshLayer.
    """
    x = x * jax.numpy.sign(x.real)
    return x + jax.numpy.log(1.0 + jax.numpy.exp(-2.0 * x)) - jax.numpy.log(2.0)
LogCoshLayer = stax.elementwise(logcosh)

#https://arxiv.org/pdf/1705.09792.pdf
#complex activation function, see https://arxiv.org/pdf/1802.08026.pdf
@jax.jit
def modrelu(x):
    """modrelu activation function. To use this function as layer, use ModReLu.

        See https://arxiv.org/pdf/1705.09792.pdf
        """
    return jnp.maximum(1, jnp.abs(x)) * x/jnp.abs(x)
ModReLu = stax.elementwise(modrelu)

#https://arxiv.org/pdf/1705.09792.pdf
#complex activation function, see https://arxiv.org/pdf/1802.08026.pdf
@jax.jit
def complexrelu(x):
    """complexrelu activation function. To use this function as layer, use ComplexReLu.

            See https://arxiv.org/pdf/1705.09792.pdf
            """
    return jnp.maximum(0, x.real) + 1j* jnp.maximum(0, x.imag)
ComplexReLu = stax.elementwise(complexrelu)


def SumLayer():
    """Layer to sum the input. output_shape = (..., 1).
        I use an other implementation than NetKet. Maybe this will be faster for GPU training.
                """
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        #return inputs.sum(axis=-1)
        W = jnp.ones((inputs.shape[1], 1), dtype=jnp.int64)
        return jnp.dot(inputs, W).T

    return init_fun, apply_fun
SumLayer = SumLayer()


def FormatLayer():
    """Ensures the correct dimension of the output of a network.
        It was needed in an old version of NetKet. I do not know, if this is still needed
                    """
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        return inputs[:,0]

    return init_fun, apply_fun
FormatLayer = FormatLayer()


def InputForConvLayer():
    """Adds an additional dimension to the data. Use this Layer before the first convolutional layer.
                        """
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], input_shape[1], 1)
        return output_shape, ()
    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        outputs = jnp.empty((inputs.shape[0], inputs.shape[1], 1), dtype=jnp.complex128)
        outputs = jax.ops.index_update(outputs, jax.ops.index[:, :, 0], inputs[:, :])
        return outputs
    return init_fun, apply_fun
InputForConvLayer = InputForConvLayer()

def FixSrLayer():
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], input_shape[1])
        return output_shape, ()
    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        if(len(inputs.shape) ==1):
            second_shape = inputs.shape[0]
            first_shape = 1
            outputs = jnp.empty((first_shape, second_shape), dtype=jnp.complex128)
            outputs = jax.ops.index_update(outputs, jax.ops.index[0, :], inputs[:])
        else:
            outputs = inputs
        return outputs
    return init_fun, apply_fun
FixSrLayer = FixSrLayer()


def InputForDenseLayer():
    """Flattens the data. Use this Layer after the last convolutional layer.
        You can use stax. Flatten instead.
                            """
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], input_shape[1]*input_shape[2])
        return output_shape, ()
    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        num_channels = inputs.shape[2]
        input_size = inputs.shape[1]
        outputs = jnp.empty((inputs.shape[0], input_size*num_channels), dtype=jnp.complex128)
        for i in range(0, num_channels):
            outputs = jax.ops.index_update(outputs, jax.ops.index[:, i*input_size:(i+1)*input_size], inputs[:, :, i])
        return outputs
    return init_fun, apply_fun
InputForDenseLayer = InputForDenseLayer()


#periodic padding
def PaddingLayer():
    """Periodic padding. Input dimension L -> Output dimension 2*L-1.
        This is especially useful for lattices with periodic boundary conditions.
                                """
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], 2*input_shape[1]-1, input_shape[2])
        return output_shape, ()
    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        input_size = inputs.shape[1]
        outputs = jnp.empty((inputs.shape[0], 2 * input_size- 1, inputs.shape[2]), dtype=jnp.complex128)
        outputs = jax.ops.index_update(outputs, jax.ops.index[:, 0:input_size, :], inputs[:, :, :])
        outputs = jax.ops.index_update(outputs, jax.ops.index[:, input_size:2 * inputs.shape[1] - 1, :],
                                       inputs[:, 0:input_size - 1, :])
        return outputs
    return init_fun, apply_fun
PaddingLayer = PaddingLayer()


def ResFFLayer(W_init=jax.nn.initializers.glorot_normal(), b_init=jax.nn.initializers.normal()):
    """Fully connected layer with residual connection. Dense, complex ReLU, Dense, Add input data.
                                    """
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], input_shape[1])
        k1, k2, k3, k4 = jax.random.split(rng, num=4)
        W, W2, b, b2 = W_init(k1, (input_shape[-1], input_shape[1])), W_init(k2, (input_shape[-1], input_shape[1])), b_init(k3, (input_shape[1],)), b_init(k4, (input_shape[1],))
        return output_shape, (W, W2, b, b2)

    def apply_fun(params, inputs, **kwargs):
        W, W2, b, b2 = params
        # inputs_rightShape = jnp.empty((inputs.shape[0], alpha*inputs.shape[1]), dtype=jnp.complex128)
        # for i in range(alpha):
        #     inputs_rightShape = jax.ops.index_update(inputs_rightShape, jax.ops.index[:, alpha*inputs.shape[1]:(alpha+1)*inputs.shape[1]], inputs[:, :])
        outputs = jnp.dot(inputs, W) + b
        outputs = jax.vmap(complexrelu)(outputs)
        outputs = jnp.dot(outputs, W2) + b2 + inputs
        #outputs = jax.vmap(complexrelu)(outputs)
        return outputs

    return init_fun, apply_fun

def ResConvLayer(out_chan, filter_shape = (3,), strides=None, W_init=None, b_init=jax.nn.initializers.normal(1e-6)):
    """Convolutional layer with residual connection. Conv1D, complex ReLU, Conv1D, Add input data.

    Args:
        out_chan (int) : number of output channels
        filter_shape (int) : shape of the filter. Recommended: (3,)
                                        """
    #1 dimensional Convolution
    dimension_numbers = ('NHC', 'HIO', 'NHC')
    #I need padding to ensure, that I can add the input and output dimension
    padding = 'Same'
    lhs_spec, rhs_spec, out_spec = dimension_numbers
    one = (1,) * len(filter_shape)
    strides = strides or one
    W_init = W_init or jax.nn.initializers.glorot_normal(rhs_spec.index('I'), rhs_spec.index('O'))
    def init_fun(rng, input_shape):
        filter_shape_iter = iter(filter_shape)
        kernel_shape = [out_chan if c == 'O' else
                        input_shape[lhs_spec.index('C')] if c == 'I' else
                        next(filter_shape_iter) for c in rhs_spec]
        output_shape = jax.lax.conv_general_shape_tuple(
            input_shape, kernel_shape, strides, padding, dimension_numbers)
        bias_shape = [out_chan if c == 'C' else 1 for c in out_spec]
        k1, k2, k3, k4 = jax.random.split(rng, 4)
        W, b = W_init(k1, kernel_shape), b_init(k2, bias_shape)
        W2, b2 = W_init(k3, kernel_shape), b_init(k4, bias_shape)
        return output_shape, (W, W2, b, b2)
    def apply_fun(params, inputs, **kwargs):
        W, W2, b, b2 = params
        outputs = jax.lax.conv_general_dilated(inputs, W, strides, padding, one, one,
                                        dimension_numbers=dimension_numbers) + b
        outputs = jax.vmap(complexrelu)(outputs)
        outputs = jax.lax.conv_general_dilated(outputs, W2, strides, padding, one, one,
                                               dimension_numbers=dimension_numbers) + b2
        outputs += inputs
        return outputs
    return init_fun, apply_fun


def UnaryLayer():
    """Reformats the data.
    Input: A spin is represented by one neuron as -2., 0., or 2.
    Output: A spin is represented by three neurons. Two are zero and one neuron is 1.
                                            """
    def init_fun(rng, input_shape):
        output_shape = (input_shape[0], 3 * input_shape[1])
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        def input_to_unary(input):
            unary_output = jnp.empty((input.shape[0], 3), dtype=jnp.int64)
            input1 = input[:]
            input2 = input[:]
            input3 = input[:]
            jnp.where(input1 <= 1, 1, 0 )
            jnp.where(input3 >= 1, 1, 0)
            jnp.where(input2 == 0, 1, 0)
            unary_output = jax.ops.index_update(unary_output, jax.ops.index[:, 0], input1[:])
            unary_output = jax.ops.index_update(unary_output, jax.ops.index[:, 1], input2[:])
            unary_output = jax.ops.index_update(unary_output, jax.ops.index[:, 2], input3[:])
            return unary_output

        input_size = inputs.shape[1]
        outputs = jnp.empty((inputs.shape[0], 3 * input_size), dtype=jnp.complex128)
        for i in range(input_size):
            unary_output = input_to_unary(inputs[:, i])
            outputs = jax.ops.index_update(outputs, jax.ops.index[:, 3*i:3*(i+1)], unary_output[:, :])
        return outputs

    return init_fun, apply_fun
UnaryLayer = UnaryLayer()

"""One dimensional convolutional layer. Conv1d                                     """
Conv1d = functools.partial(stax.GeneralConv, ('NHC', 'HIO', 'NHC'))


def JaxRBM(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex Restricted Boltzmann Machine implemented in Jax.
        Dense, LogCosh, Sum

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                            """
    print('JaxRBM is used')
    input_size = hilbert.size
    ma = nk.machine.Jax(
        hilbert,
        stax.serial(FixSrLayer, stax.Dense(alpha * input_size), LogCoshLayer, SumLayer),
        dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if(optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif(optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if(sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxRBM'
    return ma, op, sa, machine_name


def JaxSymmRBM(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex symmetric Restricted Boltzmann Machine implemented in Jax.
        Conv1d, LogCosh, Sum

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                """
    print('JaxSymmRBM is used')
    input_size = hilbert.size
    ma = nk.machine.Jax(
        hilbert,
        stax.serial(FixSrLayer, InputForConvLayer, PaddingLayer, Conv1d(alpha, (input_size,)), LogCoshLayer, InputForDenseLayer, SumLayer),
        dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxSymmRBM'
    return ma, op, sa, machine_name


#https://journals.aps.org/prb/abstract/10.1103/PhysRevB.99.155136
def JaxUnaryRBM(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex unary Restricted Boltzmann Machine implemented in Jax.
        UnaryLayer, Dense, LogCosh, Sum

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator

                                                """
    print('JaxUnaryRBM is used')
    input_size = hilbert.size
    ma = nk.machine.Jax(
        hilbert,
        stax.serial(FixSrLayer, UnaryLayer, stax.Dense(alpha * input_size), LogCoshLayer, SumLayer),
        dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if(optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif(optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if(sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxUnaryRBM'
    return ma, op, sa, machine_name


def JaxFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex Feed Forward Neural Network (fully connected) Machine implemented in Jax. One hidden layer.
        Dense, ComplexReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                """
    print('JaxFFNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer,
        Dense(input_size * alpha), ComplexReLu,
        Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif(sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxFFNN'
    return ma, op, sa, machine_name


def JaxResFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex deep residual Feed Forward Neural Network Machine implemented in Jax.
        Dense, ReLU, ResFF, ReLU, ResFF, Relu, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                """
    print('JaxResFFNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer, Dense(input_size*alpha), ComplexReLu, ResFFLayer(), ComplexReLu, ResFFLayer(), ComplexReLu, Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxResFFNN'
    return ma, op, sa, machine_name


def JaxUnaryFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex dunary Feed Forward Neural Network Machine implemented in Jax.
            UnaryLayer, Dense, ReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                    """
    print('JaxUnaryFFNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer, UnaryLayer,
        Dense(input_size * alpha), ComplexReLu,
        Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxUnaryFFNN'
    return ma, op, sa, machine_name


def JaxSymmFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex symmetric Feed Forward Neural Network Machine implemented in Jax.
            PaddingLayer, Conv1d, ComplexReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                    """
    print('JaxSymmFFNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer, InputForConvLayer, PaddingLayer, Conv1d(alpha, (input_size,)), ComplexReLu, InputForDenseLayer, Dense(input_size * alpha),
                                      Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxSymmFFNN'
    return ma, op, sa, machine_name

def JaxConv3NN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex Neural Network Machine with one convolutional filter implemented in Jax.
            Conv1d, complex ReLU, Dense, complex ReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                    """
    print('JaxConv3NN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer, InputForConvLayer, Conv1d(alpha, (3,)), ComplexReLu, stax.Flatten, Dense(input_size * alpha), ComplexReLu,
                                      Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxConv3NN'
    return ma, op, sa, machine_name


def JaxDeepConvNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex deep convolutional Neural Network Machine implemented in Jax.
            Conv1d, complexReLU, Conv1d, complexReLU, Conv1d, complexReLU,
            Conv1d, complexReLU, Dense, complexReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                    """
    print('JaxDeepConvNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer, InputForConvLayer, Conv1d(alpha, (3,)), ComplexReLu,
                                      Conv1d(alpha, (3,)), ComplexReLu, Conv1d(alpha, (3,)), ComplexReLu,
                                      Conv1d(alpha, (3,)), ComplexReLu, stax.Flatten,
                                      Dense(input_size * alpha), ComplexReLu, Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxResConvNN'
    return ma, op, sa, machine_name


def JaxResConvNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex deep residual convolutional Neural Network Machine implemented in Jax.
            Conv1d, complexReLU, ResConv, complexReLU, ResConv, complexReLU, ResConv, complexReLU, ResConv, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                                                    """
    print('JaxResConvNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer, InputForConvLayer, Conv1d(alpha, (3,)), ComplexReLu, ResConvLayer(alpha),
                                      ComplexReLu, ResConvLayer(alpha), ComplexReLu, ResConvLayer(alpha), ComplexReLu,
                                      ResConvLayer(alpha), ComplexReLu, stax.Flatten, Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxResConvNN'
    return ma, op, sa, machine_name


def JaxDeepFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Complex deep Feed Forward Neural Network Machine implemented in Jax with two hidden layer.
                Dense, complexReLU, Dense, complex ReLU, Dense, complex ReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                """
    print('JaxDeepFFNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer,
        Dense(input_size * alpha), ComplexReLu, Dense(input_size * alpha), ComplexReLu, Dense(input_size * alpha),
        ComplexReLu, Dense(1), FormatLayer)
    ma = nk.machine.Jax(
        hilbert,
        (init_fun, apply_fun), dtype=complex
    )
    ma.init_random_parameters(seed=12, sigma=0.01)
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    elif (sampler == 'VBS'):
        sa = my_sampler.getVBSSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxDeepFFNN'
    return ma, op, sa, machine_name


class Torch_FFNN_model(torch.nn.Module):
    """Class for a real Feed Forward Neural Network implemented in PyTorch."""
    def __init__(self, hilbert, alpha):
        super(Torch_FFNN_model, self).__init__()
        input_size = hilbert.size
        self.fc1 = torch.nn.Linear(input_size, alpha*input_size)
        self.fc2 = torch.nn.Linear(alpha*input_size, 2)
    def forward(self, x):
        #x.to(torch.device("cuda:0"))
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


#alpha should be twice as high as used with Jax, because PyTorch deals with real numbers!
def TorchFFNN(hilbert, hamiltonian, alpha=2, optimizer='Sgd', lr=0.1, sampler = 'Local'):
    """Real Feed Forward Neural Network Machine implemented in Pytorch with one hidden layer.
                Dense, ReLU, Dense, ReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                """
    print('TorchFFNN is used')
    Torch_TFFNN = Torch_FFNN_model(hilbert, alpha)
    ma = nk.machine.Torch(Torch_TFFNN, hilbert=hilbert)
    # Optimizer
    # Note: there is a mistake in netket/optimizer/torch.py -> change optim to _torch.optim
    if (optimizer == 'Sgd'):
        op = Torch(ma, SGD, lr=lr)
    elif (optimizer == 'Adam'):
        op = Torch(ma, Adam, lr=lr)
    else:
        op = Torch(ma, Adamax, lr=lr)
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    ma.init_random_parameters(seed=12, sigma=0.01)
    machine_name = 'TorchFFNN'
    return ma, op, sa, machine_name


#Should be used with padding=='circular'
#Should not used for multiple-Layer networks, because it transformes the shape of the input every time anew
class Torch_Conv1d_Layer(_ConvNd):
    """Real 1d convolutional Layer implemented with PyTorch. Can be used with periodic padding."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros'  # TODO: refine this type
    ):
        kernel_size = _single(kernel_size)
        stride = _single(stride)
        padding = _single(padding)
        dilation = _single(dilation)
        super(Conv1d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _single(0), groups, bias, padding_mode)
    def forward(self, input: Tensor) -> Tensor:
        if(self.padding_mode == 'zeros'):
            input = input.unsqueeze(1)
            tmp = F.conv1d(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)
            # changes the output-format of a conv-nn to the format of a ff-nn
            tmp = tmp.view(tmp.size(0), -1)
            return tmp
        #I implemeted circular padding
        else:
            input = input.unsqueeze(1)
            length = input.shape[2]
            x = torch.empty(input.shape[0], input.shape[1], 2*length-1, dtype=input.dtype)
            x[:, :, 0:length] = input[:, :, :]
            x[:, :, length:2*length-1] = input[:, :, 0:length-1]
            tmp = F.conv1d(x, self.weight, self.bias, self.stride,
                            self.padding, self.dilation, self.groups)
            #changes the output-format of a conv-nn to the format of a ff-nn
            tmp = tmp.view(tmp.size(0), -1)
            return tmp


# uses circular padding
#alpha is the number filters used per Conv1d layer
class Torch_ConvNN_model(torch.nn.Module):
    """Class for a real convolutional Neural Network implemented in PyTorch."""
    def __init__(self, hilbert, alpha=1):
        super(Torch_ConvNN_model, self).__init__()
        input_size = hilbert.size
        self.layer1 = torch.nn.Conv1d(1, alpha, kernel_size=input_size)
        self.layer2 = torch.nn.Linear(alpha*input_size, 2)
        self.padding_mode = 'circular'
    def forward(self, x):
        # Does circular padding
        def _do_padding(input):
            length = input.shape[2]
            tmp = torch.empty(input.shape[0], input.shape[1], 2 * length - 1, dtype=input.dtype)
            tmp[:, :, 0:length] = input[:, :, :]
            tmp[:, :, length:2 * length - 1] = input[:, :, 0:length - 1]
            return tmp
        #Converts the Linear to the Conv1d format
        x = x.unsqueeze(1)
        x = F.relu(self.layer1(_do_padding(x)))
        #now, here, more convolutional layers could be added
        #Converts the Conv1d to the linear format
        x = x.view(x.size(0), -1)
        x = self.layer2(x)
        return x


def TorchConvNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Real symmetric Neural Network Machine implemented in Pytorch with one hidden layer.
                    Conv1d, ReLU, Dense, ReLU, Dense

    Args:
        hilbert (netket.hilbert) : hilbert space
        hamiltonian (netket.hamiltonian) : hamiltonian
        alpha (int) : hidden layer density
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        ma (netket.machine) : machine
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
        machine_name (str) : name of the machine, see get_operator
                    """
    print('TorchConvNN is used')
    Torch_ConvNN = Torch_ConvNN_model(hilbert, alpha)
    ma = nk.machine.Torch(Torch_ConvNN, hilbert=hilbert)
    # Optimizer
    # Note: there is a mistake in netket/optimizer/torch.py -> change optim to optimizer
    if (optimizer == 'Sgd'):
        op = Torch(ma, SGD, lr=lr)
    elif (optimizer == 'Adam'):
        op = Torch(ma, Adam, lr=lr)
    else:
        op = Torch(ma, Adamax, lr=lr)
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    ma.init_random_parameters(seed=12, sigma=0.01)
    machine_name = 'TorchConvNN'
    return ma, op, sa, machine_name


# Only Jax-optimizers are used at the moment -> watch out, if PyTorch is used!
# Input: machine with already loaded parameters. Here, only optimizer and sampler are updated
def load_machine(machine, hamiltonian, optimizer='Sgd', lr=0.1, sampler='Local'):
    """Method to get an operator and sampler for a loaded machine. The machine is not loaded in this method!

    Args:
        machine (netket.machine) : loaded machine
        hamiltonian (netket.hamiltonian) : hamiltonian
        optimizer (str) : possible choices are 'Sgd', 'Adam', or 'AdaMax'
        lr (float) : learning rate
        sampler (str) : possible choices are 'Local', 'Exact', 'VBS'

    Returns:
        op (netket.optimizer) : optimizer
        sa (netket.sampler) : sampler
    """
    ma = machine
    # Optimizer
    if (optimizer == 'Sgd'):
        op = Wrap(ma, SgdJax(lr))
    elif (optimizer == 'Adam'):
        op = Wrap(ma, AdamJax(lr))
    else:
        op = Wrap(ma, AdaMaxJax(lr))
    # Sampler
    if (sampler == 'Local'):
        sa = nk.sampler.MetropolisLocal(machine=ma)
    elif (sampler == 'Exact'):
        sa = nk.sampler.ExactSampler(machine=ma)
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    return op, sa



#method to simply get the desired machine
def get_machine(machine_name):
    """Method to easily get the desired machine

    Args:
        machine_name (str) : possible choices are 'JaxRBM', 'JaxSymmRBM', 'JaxFFNN', 'JaxDeepFFNN', 'TorchFFNN',
           'TorchConvNN', 'JaxSymmFFNN', 'JaxUnaryRBM', 'JaxUnaryFFNN', 'JaxResFFNN', 'JaxConv3NN',
           'JaxResConvNN', or 'JaxDeepConvNN'
    """
    if(machine_name == 'JaxRBM'):
        return JaxRBM
    elif(machine_name == 'JaxSymRBM' or machine_name == 'JaxSymmRBM'):
        return JaxSymmRBM
    elif(machine_name == 'JaxFFNN'):
        return JaxFFNN
    elif(machine_name == 'JaxDeepFFNN'):
        return JaxDeepFFNN
    elif(machine_name == 'TorchFFNN'):
        return TorchFFNN
    elif(machine_name == 'TorchConvNN'):
        return TorchConvNN
    elif (machine_name == 'JaxSymFFNN' or machine_name == 'JaxSymmFFNN'):
        return JaxSymmFFNN
    elif(machine_name == 'JaxUnaryRBM'):
        return JaxUnaryRBM
    elif (machine_name == 'JaxUnaryFFNN'):
        return JaxUnaryFFNN
    elif (machine_name == 'JaxResNet' or machine_name == 'JaxResFFNN'):
        return JaxResFFNN
    elif (machine_name == 'JaxConv3NN'):
        return JaxConv3NN
    elif (machine_name == 'JaxResConvNN'):
        return JaxResConvNN
    elif (machine_name == 'JaxDeepConvNN'):
        return  JaxDeepConvNN
    else:
        print('The desired machine was spelled wrong!')
        sys.stdout.flush()
        return None










