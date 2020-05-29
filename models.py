import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

import numpy as np

import utils
import scores

from pytorch_lightning.core.lightning import LightningModule
from place_cells import PlaceCells
from data_generator import DataGenerator

# from hessianfree import HessianFree

#torch.backends.cudnn.benchmark = False

logsoftmax = nn.LogSoftmax(dim=-1)
softmax = nn.Softmax(dim=-1)

def cross_entropy(preds, labels):
    return (-logsoftmax(preds) * softmax(labels)).sum(dim=-1).mean()

criterion = cross_entropy
# criterion = torch.nn.MSELoss()

starts = [0.2] * 10
ends = np.linspace(0.4, 1.0, num=10)
masks_parameters = zip(starts, ends.tolist())
scorer = scores.GridScorer(20, ((-110.0, 110.0), (-110.0, 110.0)), masks_parameters)

# TODO: an instance of a model should contain info about pc_centers, traj length, and this should all be saved in metadata
# TODO: maybe use comet
# TODO: try GRU and LSTM to solve vanishing gradient
class VanillaRNN(LightningModule):
    def __init__(self, options, input_size=2, hidden_size=512, output_size=512, num_layers=1, nonlinearity='relu', bias=False, batch_first=True, dropout=0):
        super().__init__()
        self.rnn = nn.RNN(input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            nonlinearity=nonlinearity,
            bias=bias,
            batch_first=batch_first,
            dropout=dropout)
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.hidden_transform = nn.Linear(in_features=output_size, out_features=hidden_size*num_layers, bias=False)
        self.readout = nn.Linear(in_features=hidden_size, out_features=output_size, bias=False)
        self.logsoftmax = nn.LogSoftmax(dim=-1)
        self.softmax = nn.Softmax(dim=-1)
        nn.init.orthogonal_(self.rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('relu'))
        nn.init.orthogonal_(self.rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('relu'))
        # self.rnn.weight_ih_l0.data = nn.init.xavier_uniform_(self.rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('relu'))
        # self.rnn.weight_hh_l0.data = nn.init.xavier_uniform_(self.rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('relu'))
        # He init below works better :)
        # self.rnn.weight_ih_l0.data = nn.init.kaiming_uniform_(self.rnn.weight_ih_l0.data, mode='fan_in', nonlinearity='relu')
        # self.rnn.weight_hh_l0.data = nn.init.kaiming_uniform_(self.rnn.weight_hh_l0.data, mode='fan_in', nonlinearity='relu')
        self.options = options
        # TODO: set-up checkpointing so that saved model along with saved place cell centers can be loaded
        self.pc = PlaceCells(n_P=512, L=self.options.box_width, sigma_1=20, sigma_2=40, func_type='DiffGaussTuningCurve', gpu=torch.cuda.is_available()) # this gpu might fail if is available but not used

    def forward(self, x, init_pos):
        h_0 = self.hidden_transform(init_pos)
        h_0 = h_0.view(self.num_layers, h_0.shape[0], self.hidden_size)
        h_states, h_n = self.rnn(x, h_0) # should provide h_0 too as it defaults to 0 now
        end = self.readout(h_states)
        return end, h_n, h_states

    def training_step(self, batch, batch_idx):
        inputs, place_outputs, _ = batch
        output, _, _ = self(inputs[0], inputs[1])
        loss = criterion(output, place_outputs)
        tensorboard_logs = {'train_loss': loss}
        return {'loss': loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=0.0001, weight_decay=1e-5)

    def train_dataloader(self):
        dg = DataGenerator(self.options, self.pc, gpu=self.on_gpu, train=True)
        return dg

    def validation_step(self, batch, batch_idx):
        inputs, place_outputs, pos = batch
        output, _, act = self(inputs[0], inputs[1])
        loss = criterion(output, place_outputs)
        pos = pos.reshape(-1, pos.shape[-1])
        act = act.reshape(-1, act.shape[-1])
        return {'val_loss': loss, 'pos': pos, 'act': act}

    #TODO update the plot generation with bsorch's (seems faster)
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        # maybe only need to do outputs['val_loss']
        tensorboard_logs = {'val_loss': avg_loss}
        # these are the full size of the epoch
        pos = torch.cat([x['pos'] for x in outputs], dim=0).to('cpu').detach().numpy()
        act = torch.cat([x['act'] for x in outputs], dim=0).to('cpu').detach().numpy()
        # TODO: fix these vars
        plots_dir = 'D:\\research\\grid-cells-research\\with_mse'
        epoch = 0
        _ = utils.get_scores_and_plot(scorer, pos, act, plots_dir, 'activations_epoch_{}.pdf'.format(epoch))
        del pos, act
        return {'val_loss': avg_loss, 'log': tensorboard_logs}

    def val_dataloader(self):
        # TODO: do a real train/val split
        dg = DataGenerator(self.options, self.pc, gpu=self.on_gpu, train=False)
        return dg