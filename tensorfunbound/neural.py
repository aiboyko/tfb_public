import numpy as np
import matplotlib.pyplot as plt
import time
from scipy.io import loadmat
import os


import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from ipypb import ipb as tqdm_notebook


class Dasha_SIREN(torch.nn.Module):
    '''Dasha-made code'''
    def __init__(self, n_layers, hid_dim, inp_dim, out_dim=1, omega_0=30, device='cpu'):
        # md = Dasha_SIREN(n_layers=5, hid_dim=512, inp_dim=2, out_dim=1)
        # md.apply(lambda x: weights_init_siren(x, c=6))
        # md = md.to(device)
        # for p in md.parameters():
        #     p.data *= md.omega_0
        #     break


        super().__init__()
        self.omega_0 = omega_0
        self.device = device
        if n_layers > 1:
            layers = [torch.nn.Linear(inp_dim, hid_dim)]
            layers.extend([torch.nn.Linear(hid_dim, hid_dim) for _ in range(n_layers-2)])
            layers.append(torch.nn.Linear(hid_dim, out_dim))
        else:
            layers = [torch.nn.Linear(inp_dim, out_dim)]

        self.layers = torch.nn.ModuleList(layers)
    
    def _weights_init_siren(m, c=6):
        if isinstance(m, torch.nn.Linear):
            inp_dim = m.weight.shape[0]
            torch.nn.init.uniform_(m.weight, -np.sqrt(c/inp_dim), np.sqrt(c/inp_dim))
            torch.nn.init.zeros_(m.bias)

    def do_the_magic_init(self):
        self.apply(lambda x: self._weights_init_siren(x, c=6))
        self = self.to(self.device)
        for p in self.parameters():
            p.data *= self.omega_0
            break
    
    def forward(self, x):
        for l in self.layers[:-1]:
            x = l(x)
            x = torch.sin(x)
        return self.layers[-1](x)


class SineLayer(nn.Module):
    '''the original code'''
    # See paper sec. 3.2, final paragraph, and supplement Sec. 1.5 for discussion of omega_0.
    
    # If is_first=True, omega_0 is a frequency factor which simply multiplies the activations before the 
    # nonlinearity. Different signals may require different omega_0 in the first layer - this is a 
    # hyperparameter.
    
    # If is_first=False, then the weights will be divided by omega_0 so as to keep the magnitude of 
    # activations constant, but boost gradients to the weight matrix (see supplement Sec. 1.5)
    
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        
        self.init_weights()
    
    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features, 
                                             1 / self.in_features)      
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0, 
                                             np.sqrt(6 / self.in_features) / self.omega_0)
        
    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))
    
    def forward_with_intermediate(self, input): 
        # For visualization of activation distributions
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate
    
    
class SirenOriginal(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=True, 
                 first_omega_0=30, hidden_omega_0=30.):
        super().__init__()
        
        self.net = []
        self.net.append(SineLayer(in_features, hidden_features, 
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features, 
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0, 
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
                
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features, 
                                      is_first=False, omega_0=hidden_omega_0))
        
        self.net = nn.Sequential(*self.net)
    
    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        output = self.net(coords)
        return output, coords        

    def forward_with_activations(self, coords, retain_grad=False):
        '''Returns not only model output, but also intermediate activations.
        Only used for visualizing activations later!'''
        activations = OrderedDict()

        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x
        for i, layer in enumerate(self.net):
            if isinstance(layer, SineLayer):
                x, intermed = layer.forward_with_intermediate(x)
                
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                    
                activations['_'.join((str(layer.__class__), "%d" % activation_count))] = intermed
                activation_count += 1
            else: 
                x = layer(x)
                
                if retain_grad:
                    x.retain_grad()
                    
            activations['_'.join((str(layer.__class__), "%d" % activation_count))] = x
            activation_count += 1

        return activations


def train_siren_pixels(model, dataloader, data, L, n_epochs=100, lrs=[1e-3], randomize=True):
    device = list(model.parameters())[0].device
    losses = []
    for lr in lrs:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)   
        for epoch in tqdm_notebook(range(n_epochs)):
            for x in dataloader:
                optimizer.zero_grad()
                indices = torch.stack(x, dim=1).squeeze().T
                y_goal = data[list(indices.T)]
                x = indices.float().to(device)
                if randomize:
                    r = torch.rand_like(x) - .5
                    x = x + r
                y_predicted = model(x)
                loss = L(y_predicted.squeeze(-1), y_goal.float()) 
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append(loss.detach())
    return losses


def train2(model, dataloader, data, L, n_epochs=100, lrs=[1e-4]):
    device = list(model.parameters())[0].device
    losses = []
    for lr in lrs:
        optimizer = torch.optim.Adam(params=model.parameters(), lr=lr)   
        for epoch in tqdm_notebook(range(n_epochs)):
            for x in dataloader:
                optimizer.zero_grad()
                indices = torch.stack(x, dim=1).squeeze().T
                y_goal = data[list(indices.T)]
                


                x = indices.float().to(device)
                r = torch.randn_like(x) 
                y_predicted, _ = model(x+r)

                # print(y_predicted)
                loss = L(y_predicted.squeeze(-1), y_goal.float()) 
                loss.backward(retain_graph=True)
                optimizer.step()
                losses.append(loss.detach())
    return losses