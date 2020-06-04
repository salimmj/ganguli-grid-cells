import numpy as np
import torch
import scipy.interpolate

class PlaceCells(object):
    def __init__(self, options, gpu=False):
        self.Np = options.nP
        self.sigma = options.sigma
        self.surround_scale = options.surround_scale
        self.box_width = options.box_width
        self.box_height = options.box_height
        self.is_periodic = options.periodic
        self.DoG = options.DoG
        self.device = 'cuda' if gpu else 'cpu'
        torch.manual_seed(0) # for reproducibility -- TODO start saving place cell centers instead
        # Randomly tile place cell centers across environment
        usx = torch.rand((self.Np,), dtype=torch.float32).to(self.device)*self.box_width - self.box_width/2
        usy = torch.rand((self.Np,), dtype=torch.float32).to(self.device)*self.box_height - self.box_height/2
        self.us = torch.stack([usx, usy], axis=-1)

        
    def get_activation(self, pos):
        '''
        Get place cell activations for a given position.
        Args:
            pos: 2d position of shape [batch_size, sequence_length, 2].
        Returns:
            outputs: Place cell activations with shape [batch_size, sequence_length, Np].
        '''
        d = torch.abs(pos.unsqueeze(2) - self.us.unsqueeze(0).unsqueeze(0))
        
        if self.is_periodic:
            dx = d.select(-1, 0) 
            dy = d.select(-1, 1) 
            dx = torch.min(dx, self.box_width - dx)
            dy = torch.min(dy, self.box_height - dy)
            d = torch.stack([dx, dy], axis=-1)

        norm2 = torch.sum(d**2, dim=-1)

        # Normalize place cell outputs with prefactor alpha=1/2/np.pi/self.sigma**2,
        # or, simply normalize with softmax, which yields same normalization on 
        # average and seems to speed up training.
        outputs = torch.nn.functional.softmax(-norm2/(2*self.sigma**2), dim=2)

        if self.DoG:
            # Again, normalize with prefactor 
            # beta=1/2/np.pi/self.sigma**2/self.surround_scale, or use softmax.
            outputs -= torch.nn.functional.softmax(-norm2/(2*self.surround_scale*self.sigma**2), dim=2)

            # Shift and scale outputs so that they lie in [0,1].
            outputs += torch.abs(torch.min(outputs, dim=-1, keepdim=True).values)
            outputs /= torch.sum(outputs, dim=-1, keepdim=True)

        return outputs

    
    def get_nearest_cell_pos(self, activation, k=3):
        '''
        Decode position using centers of k maximally active place cells.
        
        Args: 
            activation: Place cell activations of shape [batch_size, sequence_length, Np].
            k: Number of maximally active place cells with which to decode position.
        Returns:
            pred_pos: Predicted 2d position with shape [batch_size, sequence_length, 2].
        '''
        idxs = torch.topk(activation, k=k).indices
        pred_pos = torch.mean(self.us[idxs], dim=2)

        return pred_pos
        

    def grid_pc(self, pc_outputs, res=32):
        ''' Interpolate place cell outputs onto a grid'''
        coordsx = np.linspace(-self.box_width/2, self.box_width/2, res)
        coordsy = np.linspace(-self.box_height/2, self.box_height/2, res)
        grid_x, grid_y = np.meshgrid(coordsx, coordsy)
        grid = np.stack([grid_x.ravel(), grid_y.ravel()]).T

        # Convert to numpy
        us_np = self.us.numpy()
        pc_outputs = pc_outputs.numpy().reshape(-1, self.Np)
        
        T = pc_outputs.shape[0]
        pc = np.zeros([T, res, res])
        for i in range(len(pc_outputs)):
            gridval = scipy.interpolate.griddata(us_np, pc_outputs[i], grid)
            pc[i] = gridval.reshape([res, res])
        
        return pc

    def compute_covariance(self, res=30):
        '''Compute spatial covariance matrix of place cell outputs'''
        pos = np.array(np.meshgrid(np.linspace(-self.box_width/2, self.box_width/2, res),
                         np.linspace(-self.box_height/2, self.box_height/2, res))).T

        pos = torch.Tensor(pos.astype(np.float32))

        pc_outputs = self.get_activation(pos)
        pc_outputs = torch.reshape(pc_outputs, (-1, self.Np))

        C = pc_outputs@pc_outputs.T
        Csquare = torch.reshape(C, (res,res,res,res))

        Cmean = np.zeros([res,res])
        for i in range(res):
            for j in range(res):
                Cmean += np.roll(np.roll(Csquare[i,j], -i, axis=0), -j, axis=1)
                
        Cmean = np.roll(np.roll(Cmean, res//2, axis=0), res//2, axis=1)

        return Cmean