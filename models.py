import torch
import torch.nn as nn
from torchdiffeq import odeint_adjoint as odeint
from einops import rearrange

import torch
import torch.nn.functional as F

################################################################
# BFNO layer
################################################################
class WeightsBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(WeightsBlock, self).__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.scale = (1 / (in_channels))

        self.weights1 = nn.Parameter(self.scale * torch.rand(2, in_channels, in_channels))
        self.bias1  = nn.Parameter(self.scale * torch.rand(2, in_channels))

    # Complex multiplication
    def compl_mul2d(self, input, weights):
        return torch.einsum("bxyi,io->bxyo", input, weights)


    def forward(self, out_ft_real, out_ft_imag):

        
        out_ft_real_1 = F.relu(self.compl_mul2d(out_ft_real, self.weights1[0]) - \
                               self.compl_mul2d(out_ft_imag, self.weights1[1]) + \
                               self.bias1[0])      

        out_ft_imag_1 = F.relu(self.compl_mul2d(out_ft_imag, self.weights1[0]) + \
                               self.compl_mul2d(out_ft_real, self.weights1[1]) + \
                               self.bias1[1])

        return out_ft_real_1, out_ft_imag_1

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, layer_num = 2):
        super(SpectralConv2d, self).__init__()

        """
        2D BFNO layer. It does FFT, linear transform, and Inverse FFT.    
        """

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.scale = (1 / (in_channels))

        self.weight_block_1 = WeightsBlock(self.in_channels, self.out_channels)
        self.weight_block_2 = WeightsBlock(self.in_channels, self.out_channels)

        self.concat_real_layer_1 = nn.Linear(int(layer_num*self.out_channels), int(self.out_channels))
        self.concat_imag_layer_1 = nn.Linear(int(layer_num*self.out_channels), int(self.out_channels))

    def forward(self, x): 
        bias = x

        # Compute Fourier coeffcients up to factor of e^(- something constant)
        out_ft = torch.fft.rfft2(x, norm='ortho') 
        out_ft_real = out_ft.real
        out_ft_imag = out_ft.imag

        out_ft_real = out_ft_real.permute(0, 2, 3, 1)
        out_ft_imag = out_ft_imag.permute(0, 2, 3, 1)
        
        out_ft_real_1, out_ft_imag_1 = self.weight_block_1(out_ft_real, out_ft_imag) 
        out_ft_real_2, out_ft_imag_2 = self.weight_block_2(out_ft_real, out_ft_imag) 

        total_real = torch.cat([out_ft_real_1, out_ft_real_2], dim=3)
        total_imag = torch.cat([out_ft_imag_1, out_ft_imag_2], dim=3)

        total_real = self.concat_real_layer_1(total_real) 
        total_imag = self.concat_imag_layer_1(total_imag)
 
        total_real = total_real.permute(0, 3, 1, 2) 
        total_imag = total_imag.permute(0, 3, 1, 2) 

        out_ft = torch.stack([total_real, total_imag], dim=-1)
        out_ft = torch.view_as_complex(out_ft)

        # Return to physical space
        x = torch.fft.irfft2(out_ft, s=(x.size(-2), x.size(-1)), norm='ortho')
        
        return x + bias
            
class DF_NO(nn.Module):

    def __init__(self, in_channels, nhidden, out_channels=None, args=None):
        super(DF_NO, self).__init__()
        self.args = args
        in_dim = in_channels

        self.activation = nn.ReLU(inplace=True) #nn.LeakyReLU(0.3)
        
        self.fc1 = SpectralConv2d(nhidden, nhidden)
        self.fc2 = SpectralConv2d(nhidden, nhidden)
        self.fc3 = SpectralConv2d(nhidden, nhidden)
        
        self.initial = nn.Linear(in_dim + 1, nhidden)
        self.final = nn.Linear(nhidden, in_dim)

    def forward(self, t, x0):
        out = rearrange(x0, 'b d c x y -> b (d c) x y')

        t_img = torch.ones_like(out[:, :1, :, :]).to(device=self.args.gpu) * t
        out = torch.cat([out, t_img], dim=1)

        out = out.permute(0,2,3,1)
        out = self.initial(out)
        out = out.permute(0,3,1,2)

        out = self.fc1(out)
        out = self.activation(out)

        out = self.fc2(out)
        out = self.activation(out)

        out = self.fc3(out)

        out = out.permute(0,2,3,1)
        out = self.final(out)
        out = out.permute(0,3,1,2)
        
        out = rearrange(out, 'b c x y -> b 1 c x y')
        return out

class NODElayer(nn.Module):
    def __init__(self, df, args, evaluation_times=(0.0, 1.0)):
        super(NODElayer, self).__init__()
        self.df = df
        self.evaluation_times = torch.as_tensor(evaluation_times)
        self.args = args

    def forward(self, x0):
        out = odeint(self.df, x0, self.evaluation_times, rtol=self.args.tol, atol=self.args.tol)
        return out[1]

    def to(self, device, *args, **kwargs):
        super().to(device, *args, **kwargs)
        self.evaluation_times.to(device)

class NODE(nn.Module):
    def __init__(self, df=None, **kwargs):
        super(NODE, self).__init__()
        self.__dict__.update(kwargs)
        self.df = df
        self.nfe = 0

    def forward(self, t, x):
        self.nfe += 1
        return self.df(t, x)

class anode_initial_velocity(nn.Module):

    def __init__(self, in_channels, aug, args):
        super(anode_initial_velocity, self).__init__()
        self.args = args
        self.aug = aug
        self.in_channels = in_channels

    def forward(self, x0):
        x0 = rearrange(x0.float(), 'b c x y -> b 1 c x y')
        outshape = list(x0.shape)
        outshape[2] = self.aug
        out = torch.zeros(outshape).to(self.args.gpu)
        out[:, :, :3] += x0
        return out

class predictionlayer(nn.Module):
    def __init__(self, in_channels, width=32, height=32):
        super(predictionlayer, self).__init__()
        self.dense = nn.Linear(in_channels * width  * height, 10)

    def forward(self, x):
        x = rearrange(x[:,0], 'b c x y -> b (c x y)')
        x = self.dense(x)
        return x