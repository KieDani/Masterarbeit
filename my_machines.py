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





@jax.jit
def logcosh(x):
    x = x * jax.numpy.sign(x.real)
    return x + jax.numpy.log(1.0 + jax.numpy.exp(-2.0 * x)) - jax.numpy.log(2.0)
LogCoshLayer = stax.elementwise(logcosh)


def SumLayer():
    def init_fun(rng, input_shape):
        output_shape = (-1, 1)
        return output_shape, ()

    @jax.jit
    def apply_fun(params, inputs, **kwargs):
        return inputs.sum(axis=-1)

    return init_fun, apply_fun
SumLayer = SumLayer()


def JaxRBM(hilbert, alpha=1, optimizer='Sgd', lr=0.1):
    print('JaxRBM is used')

    input_size = hilbert.size
    ma = nk.machine.Jax(
        hilbert,
        stax.serial(stax.Dense(alpha * input_size), LogCoshLayer, SumLayer),
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
    sa = nk.sampler.MetropolisLocal(machine=ma)

    return ma, op, sa


def JaxDeepRBM(hilbert, alpha=1, optimizer='Sgd', lr=0.1):
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
    sa = nk.sampler.MetropolisLocal(machine=ma)

    return ma, op, sa


def JaxFFNN(hilbert, alpha=1, optimizer='Sgd', lr=0.1):
    print('JaxFFNN is used')

    input_size = hilbert.size
    init_fun, apply_fun = stax.serial(
        Dense(input_size * alpha), LogCoshLayer,
        Dense(input_size * alpha), LogCoshLayer,
        Dense(1))
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
    sa = nk.sampler.MetropolisLocal(machine=ma)

    return ma, op, sa



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
#Torch_TFFNN_model = Torch_TFFNN_model()


#alpha should be twice as high as used with Jax, because PyTorch deals with real numbers!
def TorchFFNN(hilbert, alpha=2, optimizer='Sgd', lr=0.1):
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
    sa = nk.sampler.MetropolisLocal(machine=ma)
    ma.init_random_parameters(seed=12, sigma=0.01)

    return ma, op, sa


#Can be used with padding=='circular'
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
        #usual settings
        if(self.padding_mode == 'zeros'):
            input = input.unsqueeze(1)
            tmp = F.conv1d(input, self.weight, self.bias, self.stride,
                           self.padding, self.dilation, self.groups)

            # changes the output-format of a conv-nn to the format of a ff-nn
            tmp = tmp.view(tmp.size(0), -1)
            return tmp

        #I implemeted circular padding
        #TODO Durch das Padding ist der Input verschoben -> das kann ich auch ohne Verschiebung machen
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
        #now mor convolutional layers could be added

        #Converts the Conv1d to the linear format
        x = x.view(x.size(0), -1)
        x = self.layer2(x)

        return x


def TorchConvNN(hilbert, alpha=1, optimizer='Sgd', lr=0.1):
    print('TorchConvNN is used')

    Torch_ConvNN = Torch_ConvNN_model(hilbert, alpha)

    ma = nk.machine.Torch(Torch_ConvNN, hilbert=hilbert)

    # Optimizer
    # Note: there is a mistake in netket/optimizer/torch.py -> change optim to optimizer
    if (optimizer == 'Sgd'):
        op = op = Torch(ma, SGD, lr=lr)
    elif (optimizer == 'Adam'):
        op = Torch(ma, Adam, lr=lr)
    else:
        op = Torch(ma, Adamax, lr=lr)

    # Sampler
    sa = nk.sampler.MetropolisLocal(machine=ma)
    ma.init_random_parameters(seed=12, sigma=0.01)

    return ma, op, sa

















