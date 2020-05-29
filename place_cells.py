import numpy as np
import random
import torch 
import torch.nn.functional as F

class PlaceCells:
  def __init__(self, n_P=512, L=2.2, sigma_1=20, sigma_2=40, func_type='DiffGaussTuningCurve', gpu=True, noise=True):
    """
    n_P: Number of place cells
    L: Length of the square environment's sides (in meters)
    c_i = (n_P, 2)-shaped matrix of place cell receptive field centers 
    """
    self.n_P = n_P
    self.L = L 
    if gpu:
        self.c_i  = ((torch.rand(self.n_P,2)*self.L)-(self.L/2.0)).cuda()
    else:
        self.c_i  = ((torch.rand(self.n_P,2)*self.L)-(self.L/2.0))
    self.sigma_1 = sigma_1
    self.sigma_2 = sigma_2
    self.types = {'GaussTuningCurve': self._gaussian_tuning_curve,
      'DiffGaussTuningCurve': self._difference_gaussian_tuning_curve}
    self.func_type = func_type
    self.gpu = gpu
    self.noise = noise

  def get_closest_center(self, locations):
    return torch.norm(locations.unsqueeze(1) - self.c_i, dim=2).argmin(dim=-1)

  def decode_pos(self, act):
    return act.topk(3, dim=-1, largest=False)[1]

  def encode_pos(self, locations):
    """
    locations: (n_X, 2)-shaped matrix where each row describes (x,y) coordinates
                and n_X is the number of steps taken in the trajectory
    """
    # apply gaussian on each row of location matrix
    # encode_pos in torch.Size([200, 50, 2])
    # encode_pos out torch.Size([200, 50, 512])
    batch_size = locations.size(0)
    traj_len = locations.size(1)
    locations = locations.view(traj_len*batch_size, 2)
    res = self.encode_init_pos(locations).view((batch_size, traj_len, self.n_P))
    del locations, traj_len, batch_size
    return res

  def encode_init_pos(self, locations):
    # encode_init_pos in torch.Size([200, 2])
    res= self.types[self.func_type](locations).float()
    # encode_init_pos out torch.Size([200, 512])
    return res

  def _gaussian_tuning_curve(self, x):
    shift = torch.empty(self.n_P,  2)
    if self.gpu:
        shift = shift.cuda()
        x = x.cuda()
    # shifts the center by some noise
    if self.noise:
        shift = shift.normal_(mean=0,std=0.4*self.sigma_1) #stdv is 1/10 of sigma
    s_c_i = self.c_i + shift
    res = torch.exp(-torch.pow(torch.norm(x.unsqueeze(1) - s_c_i, dim=2),2))/(2*(self.sigma_1**2))
    del shift, c, s_c_i
    return res

  def _difference_gaussian_tuning_curve(self, x):
    shift = torch.empty(self.n_P,  2)
    if self.gpu:
        shift = shift.cuda()
        x = x.cuda()
    # shifts the center by some noise
    if self.noise:
        shift = shift.normal_(mean=0,std=0.4*self.sigma_1) #stdv is 1/10 of sigma
    s_c_i = self.c_i + shift
    den = torch.norm(x.unsqueeze(1)-s_c_i, dim=2)
    den = -torch.pow(den,2)
    res = torch.exp(den/(2*(self.sigma_1**2)))-0.5*torch.exp(den/(2*(self.sigma_2**2)))
    del shift, x, s_c_i, den
    return res

