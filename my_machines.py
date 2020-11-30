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





@jax.jit
def logcosh(x):
    x = x * jax.numpy.sign(x.real)
    return x + jax.numpy.log(1.0 + jax.numpy.exp(-2.0 * x)) - jax.numpy.log(2.0)
LogCoshLayer = stax.elementwise(logcosh)

#https://arxiv.org/pdf/1705.09792.pdf
#complex activation function, see https://arxiv.org/pdf/1802.08026.pdf
@jax.jit
def modrelu(x):
    return jnp.maximum(1, jnp.abs(x)) * x/jnp.abs(x)
ModReLu = stax.elementwise(modrelu)

#https://arxiv.org/pdf/1705.09792.pdf
#complex activation function, see https://arxiv.org/pdf/1802.08026.pdf
@jax.jit
def complexrelu(x):
    return jnp.maximum(0, x.real) + 1j* jnp.maximum(0, x.imag)
ComplexReLu = stax.elementwise(complexrelu)


def SumLayer():
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        return inputs.sum(axis=-1)

    return init_fun, apply_fun
SumLayer = SumLayer()


def FormatLayer():
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        return inputs[:,0]

    return init_fun, apply_fun
FormatLayer = FormatLayer()


def InputForConvLayer():
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
            second_shape = inputs.shape[1]
            outputs = jnp.empty((inputs.shape[0], second_shape), dtype=jnp.complex128)
            outputs = jax.ops.index_update(outputs, jax.ops.index[:, :], inputs[:, :])
        return outputs
    return init_fun, apply_fun
FixSrLayer = FixSrLayer()



def InputForDenseLayer():
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


Conv1d = functools.partial(stax.GeneralConv, ('NHC', 'HIO', 'NHC'))


def JaxRBM(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
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
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxRBM'
    return ma, op, sa, machine_name


def JaxSymmRBM(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
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
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxSymmRBM'
    return ma, op, sa, machine_name


def JaxDeepRBM(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler = 'Local'):
    print('JaxDeepRBM is used')
    input_size = hilbert.size
    ma = nk.machine.Jax(
        hilbert,
        stax.serial(stax.Dense(alpha * input_size), LogCoshLayer, stax.Dense(alpha * input_size), LogCoshLayer, SumLayer),
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
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxDeepRBM'
    return ma, op, sa, machine_name


def JaxFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
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
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxFFNN'
    return ma, op, sa, machine_name


def JaxDeepFFNN(hilbert, hamiltonian, alpha=1, optimizer='Sgd', lr=0.1, sampler='Local'):
    print('JaxDeepFFNN is used')
    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(FixSrLayer,
        Dense(input_size * alpha), ComplexReLu, Dense(input_size * alpha), ComplexReLu,
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
    else:
        sa = nk.sampler.MetropolisHamiltonian(machine=ma, hamiltonian=hamiltonian, n_chains=16)
    machine_name = 'JaxDeepFFNN'
    return ma, op, sa, machine_name


class Torch_FFNN_model(torch.nn.Module):
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













