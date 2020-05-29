import os
from torch import Tensor, cos, sin, stack, squeeze, from_numpy
import torch
import numpy as np

class DataGenerator(torch.utils.data.IterableDataset):
    def __init__(self, options, place_cells, gpu=False, train=True):
        super(DataGenerator).__init__()
        self.options = options
        self.place_cells = place_cells
        self.gpu = gpu
        self.train = train
        
    def __iter__(self):
        self.n = 0
        return self
    
    def __next__(self):
        if self.n <= len(self):
            traj = self.generate_trajectory(self.options.box_width, self.options.box_height, self.options.batch_size)

            v = stack((Tensor(traj['ego_v'])*cos(Tensor(traj['target_hd'])), 
                Tensor(traj['ego_v'])*sin(Tensor(traj['target_hd']))), axis=-1)

            pos = stack((Tensor(traj['target_x']), Tensor(traj['target_y'])), axis=-1)

            place_outputs = self.place_cells.encode_pos(pos)

            init_pos = squeeze(stack((Tensor(traj['init_x']), Tensor(traj['init_y'])), axis=-1))
            init_actv = self.place_cells.encode_init_pos(init_pos)

            inputs = (v, init_actv)
            del traj, v
            self.n += 1
            return (inputs, place_outputs, pos)
        else:
            raise StopIteration
        
    def __len__(self):
        if self.train:
            return self.options.train_epoch_size
        else:
            return self.options.val_epoch_size
    
    def avoid_wall(self, position, hd, box_width, box_height):
        '''
        Compute distance and angle to nearest wall
        '''
        x = position[:,0]
        y = position[:,1]
        dists = [box_width/2-x, box_height/2-y, box_width/2+x, box_height/2+y]
        d_wall = np.min(dists, axis=0)
        angles = np.arange(4)*np.pi/2
        theta = angles[np.argmin(dists, axis=0)]
        hd = np.mod(hd, 2*np.pi)
        a_wall = hd - theta
        a_wall = np.mod(a_wall + np.pi, 2*np.pi) - np.pi

        is_near_wall = (d_wall < self.border_region)*(np.abs(a_wall) < np.pi/2)
        turn_angle = np.zeros_like(hd)
        turn_angle[is_near_wall] = np.sign(a_wall[is_near_wall])*(np.pi/2 - np.abs(a_wall[is_near_wall]))

        return is_near_wall, turn_angle

    def generate_trajectory(self, box_width, box_height, batch_size):
        '''Generate a random walk in a rectangular box'''
        # TODO make this more PyTorch
        samples = self.options.sequence_length
        dt = 0.02  # time step increment (seconds)
        sigma = 5.76 * 2  # stdev rotation velocity (rads/sec)
        b = 13.0 * 2 * np.pi # forward velocity rayleigh dist scale (m/sec)
        mu = 0  # turn angle bias 
        self.border_region = 0.03  # meters

        # Initialize variables
        position = np.zeros([batch_size, samples+2, 2])
        head_dir = np.zeros([batch_size, samples+2])
        position[:,0,0] = np.random.uniform(-box_width/2, box_width/2, batch_size)
        position[:,0,1] = np.random.uniform(-box_height/2, box_height/2, batch_size)
        head_dir[:,0] = np.random.uniform(0, 2*np.pi, batch_size)
        velocity = np.zeros([batch_size, samples+2])

        # Generate sequence of random boosts and turns
        random_turn = np.random.normal(mu, sigma, [batch_size, samples+1])
        random_vel = np.random.rayleigh(b, [batch_size, samples+1])
        v = np.abs(np.random.normal(0, b*np.pi/2, batch_size))

        for t in range(samples+1):
            # Update velocity
            v = random_vel[:,t]
            turn_angle = np.zeros(batch_size)

            if not self.options.periodic:
                # If in border region, turn and slow down
                is_near_wall, turn_angle = self.avoid_wall(position[:,t], head_dir[:,t], box_width, box_height)
                v[is_near_wall] *= 0.25

            # Update turn angle
            turn_angle += dt*random_turn[:,t]

            # Take a step
            velocity[:,t] = v*dt
            update = velocity[:,t,None]*np.stack([np.cos(head_dir[:,t]), np.sin(head_dir[:,t])], axis=-1)
            position[:,t+1] = position[:,t] + update

            # Rotate head direction
            head_dir[:,t+1] = head_dir[:,t] + turn_angle

        # Periodic boundaries
        if self.options.periodic:
            position[:,:,0] = np.mod(position[:,:,0] + box_width/2, box_width) - box_width/2
            position[:,:,1] = np.mod(position[:,:,1] + box_height/2, box_height) - box_height/2

        head_dir = np.mod(head_dir + np.pi, 2*np.pi) - np.pi # Periodic variable

        traj = {}
        # Input variables
        traj['init_hd'] = head_dir[:,0,None]
        traj['init_x'] = position[:,1,0,None]
        traj['init_y'] = position[:,1,1,None]

        traj['ego_v'] = velocity[:,1:-1]

        ang_v = np.diff(head_dir, axis=-1)
        traj['phi_x'], traj['phi_y'] = np.cos(ang_v)[:,:-1], np.sin(ang_v)[:,:-1]

        # Target variables
        traj['target_hd'] = head_dir[:,1:-1]
        traj['target_x'] = position[:,2:,0]
        traj['target_y'] = position[:,2:,1]

        return traj

    
    def get_generator(self, batch_size=None, box_width=None, box_height=None):
        '''
        Returns a generator that yields batches of trajectories
        '''
        if not batch_size:
             batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height
            
        while True:
            traj = self.generate_trajectory(box_width, box_height, batch_size)

            v = stack((Tensor(traj['ego_v'])*cos(Tensor(traj['target_hd'])), 
                Tensor(traj['ego_v'])*sin(Tensor(traj['target_hd']))), axis=-1)

            pos = stack((Tensor(traj['target_x']), Tensor(traj['target_y'])), axis=-1)

            place_outputs = self.place_cells.encode_pos(pos)

            init_pos = squeeze(stack((Tensor(traj['init_x']), Tensor(traj['init_y'])), axis=-1))
            init_actv = self.place_cells.encode_init_pos(init_pos)

            inputs = (v, init_actv)

            yield (inputs, place_outputs, pos)



    def get_test_batch(self, batch_size=None, box_width=None, box_height=None):
        ''' For testing performance, returns a batch of smample trajectories'''
        if not batch_size:
             batch_size = self.options.batch_size
        if not box_width:
            box_width = self.options.box_width
        if not box_height:
            box_height = self.options.box_height
            
        traj = self.generate_trajectory(box_width, box_height, batch_size)

        v = stack((Tensor(traj['ego_v'])*cos(Tensor(traj['target_hd'])), 
            Tensor(traj['ego_v'])*sin(Tensor(traj['target_hd']))), axis=-1)

        pos = stack((Tensor(traj['target_x']), Tensor(traj['target_y'])), axis=-1)
        
        place_outputs = self.place_cells.encode_pos(pos)

        init_pos = squeeze(stack((Tensor(traj['init_x']), Tensor(traj['init_y'])), axis=-1))
        init_actv = self.place_cells.encode_init_pos(init_pos)

        inputs = (v, init_actv)
    
        return (inputs, place_outputs, pos)