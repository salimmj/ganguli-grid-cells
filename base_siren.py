import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from hessianfree import HessianFree

from visualize import compute_ratemaps, save_seq_err

from pytorch_lightning.core.lightning import LightningModule
from place_cells import PlaceCells
from data_generator import DataGenerator

import os

from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt

import time


##### TMP
def get_mgrid(sidelen, dim=2):
    '''Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int'''
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid




#####

class SineLayer(nn.Module):
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
    
    
class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features, outermost_linear=False, 
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
    
    def forward(self, inputs, place_outputs, pos):
        coords = pos.clone().detach().requires_grad_(True) # allows to take derivative w.r.t. input
        layer_outputs = []
        x = coords
        with torch.no_grad():
            for layer in self.net:
                x = layer(x)
                layer_outputs.append(x.clone().detach().requires_grad_(False))

        l_outputs = torch.Tensor(len(layer_outputs)-1, *layer_outputs[0].shape).cuda()
        torch.cat(layer_outputs[:-1], out=l_outputs)
        return layer_outputs[-1], coords, l_outputs

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

def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad

logsoftmax = nn.LogSoftmax(dim=-1)
softmax = nn.Softmax(dim=-1)

def cross_entropy(preds, labels):
    return (-logsoftmax(preds) * softmax(labels)).sum(dim=-1).mean()


class BaseSiren(LightningModule):
    """
    This class defines the data-loading, score calculating, visualization and
    logging structure of the model. It is to be used as a superclass for
    the different RNN models to be used (VanillaRNN, LSTM, GRU)
    """
    def __init__(self, options):
        super().__init__()
        self.options = options

        # TODO: set-up checkpointing so that saved model along with saved place cell centers can be loaded
        self.pc = PlaceCells(options, gpu=torch.cuda.is_available()) # this gpu might fail if is available but not used
        self.criterion = cross_entropy if self.options.loss == 'CE' else torch.nn.MSELoss()

    def training_step(self, batch, batch_idx):
        inputs, place_outputs, pos = batch
        output, coord, layer_outputs = self(inputs, place_outputs, pos)
        loss = self.criterion(output, place_outputs)
        print('LOSS', coord.shape)
        print('LOSS', loss)
        # Weight regularization 
        # loss += self._l2_loss() * self.options.weight_decay
        # loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1]**2)

        pred_pos = self.pc.get_nearest_cell_pos(output)
        err = torch.mean(torch.sqrt(torch.sum((pos - pred_pos)**2, dim=-1)))

        tensorboard_logs = {'train_loss': loss, 'train_err': err}
        return {'loss': loss, 'err': err, 'output': output, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.options.optim == 'SGD':
            return optim.SGD(self.siren.parameters(), lr=self.options.learning_rate, momentum=0.9)
        elif self.options.optim == 'Adam':
            return optim.Adam(self.siren.parameters(), lr=self.options.learning_rate)
        elif self.options.optim == 'RMSProp':
            return optim.RMSprop(self.siren.parameters(), lr=self.options.learning_rate)
        elif self.options.optim == 'LBFGS':
            return optim.LBFGS(self.siren.parameters(), lr=self.options.learning_rate)
        elif self.options.optim == 'HessianFree':
            return HessianFree(self.siren.parameters(), lr=self.options.learning_rate) #  use_gnm=True, verbose=True

    def train_dataloader(self):
        dg = DataGenerator(self.options, self.pc, gpu=self.on_gpu, train=True)
        return dg

    # def optimizer_step(self, current_epoch, batch_nb, optimizer, optimizer_i, second_order_closure, on_tpu, using_native_amp, using_lbfgs):
    #     optimizer.step()

    # TODO: Average of error per step to a csv 
    def validation_step(self, batch, batch_idx):
        inputs, place_outputs, pos = batch
        output, coord, layer_outputs  = self(inputs, place_outputs, pos)
        loss = self.criterion(output, place_outputs)

        pred_pos = self.pc.get_nearest_cell_pos(output)
        # err = torch.mean(torch.sqrt(torch.sum((pos - pred_pos)**2, dim=-1)))


        err = torch.mean(torch.sqrt((pos - pred_pos)**2), dim=0)

        return {'val_loss': loss, 'layer_outputs': layer_outputs, 'coord':coord, 'val_err': err, 'output': output}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_err = torch.stack([x['val_err'].mean() for x in outputs]).mean()

        # avg position error per step in sequence
        avg_seq_err = (torch.stack([x['val_err'] for x in outputs]).mean(dim=0)).mean(dim=-1)
        save_seq_err(avg_seq_err, self.options)

        # maybe only need to do outputs['val_loss']
        tensorboard_logs = {'val_loss': avg_loss, 'val_err': avg_err}
        # these are the full size of the epoch
        pos = torch.cat([x['coord'] for x in outputs], dim=0)
        # all_but_last_two_dims = pos.size()[:-2]
        # pos = pos.view(*all_but_last_two_dims, -1)
        pos = pos.to('cpu').detach().numpy()
        act = torch.cat([x['layer_outputs'] for x in outputs], dim=0)
        # all_but_last_two_dims = act.size()[:-2]
        # act = act.view(*all_but_last_two_dims, -1)
        act = act.to('cpu').detach().numpy()
        # TODO: fix these vars

        # Save a picture of rate maps
        # save_ratemaps(self.model, self.trajectory_generator, self.options, step=tot_step)
        for i in range(pos.shape[0], act.shape[0]+pos.shape[0], pos.shape[0]):
            tmp_options = self.options
            j = i/pos.shape[0]
            tmp_options.run_ID + '_L{j}' 
            ppos = pos.reshape(-1, pos.shape[-1])
            aact = act[i-pos.shape[0]:i].reshape(-1, act[i-pos.shape[0]:i].shape[-1])
            compute_ratemaps(ppos, aact, tmp_options, epoch=self.current_epoch)
         # TODO: specify range
        del pos, act
        return {'val_loss': avg_loss, 'val_err': avg_err, 'log': tensorboard_logs}

    def val_dataloader(self):
        # TODO: do a real train/val split
        dg = DataGenerator(self.options, self.pc, gpu=self.on_gpu, train=False)
        return dg

    def _l2_loss(self):
        return self.rnn.weight_hh_l0.norm(2)