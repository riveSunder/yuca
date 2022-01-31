import numpy as np

import torch
import torch.nn as nn

from yuca.utils import query_kwargs


class Identity(nn.Module):

    def __init__(self, **kwargs):
        
        super(Identity, self).__init__()

    def forward(self, x):

        return x

class Polynomial(nn.Module):

    def __init__(self, **kwargs):

        super(Polynomial, self).__init__()

        coefficients = query_kwargs("coefficients", \
                torch.tensor([0.1, 0.1, 0.1]), **kwargs)

        if type(coefficients) is not torch.Tensor:
            coefficients = torch.tensor(coefficients)

        self.coefficients = nn.Parameter(coefficients)

        self.register_parameter("coef", self.coefficients)

        self.mode = query_kwargs("mode", 0, **kwargs)

    def forward(self, x):
        eps = 1e-9
        
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        result = torch.zeros_like(x)

        for ii, value in enumerate(self.coefficients):
            
            result += value * x**ii
            
        if self.mode:
            result = result * 2 - 1
        else:
            pass

        return result

class CosOverX2(nn.Module):

    def __init__(self, **kwargs):
        super(CosOverX2, self).__init__()

        mu = query_kwargs("mu", 0.0, **kwargs)
        omega = query_kwargs("omega", 25.12, **kwargs)

        if type(mu) is not torch.Tensor:
            mu = torch.tensor(mu)
        if type(omega) is not torch.Tensor:
            omega = torch.tensor(omega)

        self.mu = nn.Parameter(mu)
        self.omega = nn.Parameter(omega)

        self.register_parameter("mu", self.mu)
        self.register_parameter("omega", self.omega)

        self.mode = query_kwargs("mode", 0, **kwargs)

    def forward(self, x):

        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        inputs = torch.abs(self.mu - x)
        if self.mode:
            result = torch.cos(self.omega * inputs) / (1+(10*inputs)**2)
        else:
            result = torch.cos(self.omega * inputs) / (1+(10*inputs)**2) + 1.0
            

        return result
        
class Gaussian(nn.Module):

    def __init__(self, **kwargs):

        super(Gaussian, self).__init__()

        mu = query_kwargs("mu", 0.0, **kwargs)
        sigma = query_kwargs("sigma", 0.1, **kwargs)

        if type(mu) is not torch.Tensor:
            mu = torch.tensor(mu)
        if type(sigma) is not torch.Tensor:
            sigma = torch.tensor(sigma)

        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)

        self.register_parameter("mu", self.mu)
        self.register_parameter("sigma", self.sigma)

        self.mode = query_kwargs("mode", 0, **kwargs)

    def forward(self, x):
        eps = 1e-9
        
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        if self.mode:
            result = torch.exp(- ((x - self.mu) / (eps + self.sigma))**2 / 2 ) * 2 -1
        else:
            result = torch.exp(- ((x - self.mu) / (eps + self.sigma))**2 / 2 )

        return result

class GaussianMixture(nn.Module):

    def __init__(self, **kwargs):
        super(GaussianMixture, self).__init__()

        self.mode = query_kwargs("mode", 0, **kwargs)
        parameters = query_kwargs("parameters", [0.5, 0.15], **kwargs)

        self.amplitudes = torch.tensor([])

        self.gaussians = []

        if len(parameters) % 3 == 0:

            for param_index in range(0, len(parameters), 3):

                self.amplitudes = torch.cat([self.amplitudes, \
                        parameters[param_index:param_index + 1]])

                self.gaussians.append(\
                        Gaussian(mu = parameters[param_index + 1], \
                        sigma = parameters[param_index + 2], \
                        mode = 0))

                for name, param in self.gaussians[-1].named_parameters():
                    self.register_parameter(f"{name}_{param_index}", param)
        else:
            for param_index in range(0,len(parameters), 2):

                self.amplitudes = torch.cat([self.amplitudes, \
                        torch.tensor([1.0])])

                self.gaussians.append(\
                        Gaussian(mu = parameters[param_index], \
                        sigma = parameters[param_index + 1], \
                        mode = 0))

                for name, param in self.gaussians[-1].named_parameters():
                    self.register_parameter(f"{name}_{param_index}", param)

    def forward(self, x):

        result = self.amplitudes[0] * self.gaussians[0](x)

        for amplitude, gaussian in zip(self.amplitudes[1:], self.gaussians[1:]):
            result += amplitude * gaussian(x)

        if self.mode:
            result = torch.clamp(result * 2 - 1, -1., 1.)
        else:
            result = torch.clamp(result, 0.0, 1.0)

        return result

class DoubleGaussian(nn.Module):

    def __init__(self, **kwargs):
        super(DoubleGaussian, self).__init__()

        mu = query_kwargs("mu", 0.0, **kwargs)
        sigma = query_kwargs("sigma", 0.1, **kwargs)
        self.mode = query_kwargs("mode", 0, **kwargs)

        if type(mu) is not torch.Tensor:
            mu = torch.tensor(mu)
        if type(sigma) is not torch.Tensor:
            sigma = torch.tensor(sigma)

        if mu.shape[0] == 1:
            mu = torch.cat([mu, mu])
        if sigma.shape[0] == 1:
            sigma = torch.cat([sigma, sigma])

        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)

        self.register_parameter("mu", self.mu)
        self.register_parameter("sigma", self.sigma)


    def forward(self, x):
        eps = 1e-9
        
        if type(x) is not torch.Tensor:
            x = torch.tensor(x)

        if self.mode:
            result = torch.exp(- \
                    ((x - self.mu[0]) / (eps + self.sigma[0]))**2 / 2 ) \
                    + torch.exp(-\
                    ((x - self.mu[1]) / (eps + self.sigma[1]))**2 / 2 )
            result = torch.clamp(result*2-1, -1.0, 1.0)
        else:
            result = torch.exp(- \
                    ((x - self.mu[0]) / (eps + self.sigma[0]))**2 / 2 ) \
                    + torch.exp(- \
                    ((x - self.mu[1]) / (eps + self.sigma[1]))**2 / 2 ) 
            result = torch.clamp(result, 0.0, 1.0)

        return result


class DoGaussian(nn.Module):

    def __init__(self, **kwargs):
        super(DoGaussian, self).__init__()

        mu = query_kwargs("mu", 0.0, **kwargs)
        sigma = query_kwargs("sigma", 0.1, **kwargs)
        self.dx = query_kwargs("dx", 1e-2, **kwargs)

        if type(mu) is not torch.Tensor:
            mu = torch.tensor(mu)
        if type(sigma) is not torch.Tensor:
            sigma = torch.tensor(sigma)

        self.mu = nn.Parameter(mu)
        self.sigma = nn.Parameter(sigma)

        self.register_parameter("mu", self.mu)
        self.register_parameter("sigma", self.sigma)

        self.mode = query_kwargs("mode", 0, **kwargs)

    def forward(self, x):
        eps = 1e-9

        if type(x) is not torch.Tensor:
            x = torch.tensor(x)
    
        gaussian_1 = torch.exp(\
                - (((x+self.dx) - self.mu) / (eps + self.sigma))**2 / 2 )
        gaussian_2 = torch.exp(\
                - (((x-self.dx) - self.mu) / (eps + self.sigma))**2 / 2 )

        if self.mode:
            result = gaussian_1 - gaussian_2 * 2 -1
        else:
            result = gaussian_1 - gaussian_2

        return result
