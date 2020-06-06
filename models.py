import torch.nn as nn

from base_rnn import BaseRNN

#from nan_police import hasnan
# from hessianfree import HessianFree

#torch.backends.cudnn.benchmark = False


# TODO: an instance of a model should contain info about pc_centers, traj length, and this should all be saved in metadata
# TODO: maybe use comet
# TODO: try GRU and LSTM to solve vanishing gradient
class VanillaRNN(BaseRNN):
    def __init__(self, options):
        super().__init__(options)
        self.rnn = nn.RNN(input_size=2,
            hidden_size=self.options.nG,
            num_layers=1,
            nonlinearity=self.options.activation,
            bias=False,
            batch_first=True,
            dropout=0)
        self.hidden_transform = nn.Linear(in_features=self.options.nP, out_features=self.options.nG, bias=False)
        self.readout = nn.Linear(in_features=self.options.nG, out_features=self.options.nP, bias=False)
        # nn.init.orthogonal_(self.rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('relu'))
        self.rnn.weight_ih_l0.data = nn.init.xavier_uniform_(self.rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('relu'))
        self.rnn.weight_hh_l0.data = nn.init.xavier_uniform_(self.rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('relu'))
        # He init below works better :)
        # self.rnn.weight_ih_l0.data = nn.init.kaiming_uniform_(self.rnn.weight_ih_l0.data, mode='fan_in', nonlinearity='relu')
        # self.rnn.weight_hh_l0.data = nn.init.kaiming_uniform_(self.rnn.weight_hh_l0.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x, init_pos):
        h_0 = self.hidden_transform(init_pos)
        h_0 = h_0.view(1, h_0.shape[0], self.options.nG)
        output, h_n = self.rnn(x, h_0) # should provide h_0 too as it defaults to 0 now
        end = self.readout(output)
        return end, h_n, output


# TODO use this instead with https://github.com/huggingface/torchMoji/blob/master/torchmoji/lstm.py
# changing hardsigmoid/tanh to sigmoid/relu
class LSTM(BaseRNN):
    def __init__(self, options):
        super().__init__(options)
        self.rnn = nn.LSTM(input_size=2,
            hidden_size=self.options.nG,
            num_layers=1,
            # nonlinearity=self.options.activation, # TODO check works
            bias=False,
            batch_first=True,
            dropout=0)
        self.hidden_transform = nn.Linear(in_features=self.options.nP, out_features=self.options.nG, bias=False)
        self.cell_transform = nn.Linear(in_features=self.options.nP, out_features=self.options.nG, bias=False)
        self.readout = nn.Linear(in_features=self.options.nG, out_features=self.options.nP, bias=False)
        # nn.init.orthogonal_(self.rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('relu'))
        # nn.init.orthogonal_(self.rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('relu'))
        self.rnn.weight_ih_l0.data = nn.init.xavier_uniform_(self.rnn.weight_ih_l0.data, gain=nn.init.calculate_gain('relu'))
        self.rnn.weight_hh_l0.data = nn.init.xavier_uniform_(self.rnn.weight_hh_l0.data, gain=nn.init.calculate_gain('relu'))
        # He init below works better :)
        # self.rnn.weight_ih_l0.data = nn.init.kaiming_uniform_(self.rnn.weight_ih_l0.data, mode='fan_in', nonlinearity='relu')
        # self.rnn.weight_hh_l0.data = nn.init.kaiming_uniform_(self.rnn.weight_hh_l0.data, mode='fan_in', nonlinearity='relu')

    def forward(self, x, init_pos):
        h_0 = self.hidden_transform(init_pos)
        h_0 = h_0.view(1, h_0.shape[0], self.options.nG)
        c_0 = self.cell_transform(init_pos)
        c_0 = c_0.view(1, c_0.shape[0], self.options.nG)
        output, (h_n, c_n) = self.rnn(x, (h_0, c_0)) # should provide h_0 too as it defaults to 0 now
        end = self.readout(output)
        return end, h_n, output
