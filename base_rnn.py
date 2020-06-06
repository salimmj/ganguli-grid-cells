import torch
import torch.nn as nn
import torch.optim as optim

from hessianfree import HessianFree

from visualize import compute_ratemaps

from pytorch_lightning.core.lightning import LightningModule
from place_cells import PlaceCells
from data_generator import DataGenerator

logsoftmax = nn.LogSoftmax(dim=-1)
softmax = nn.Softmax(dim=-1)

def cross_entropy(preds, labels):
    return (-logsoftmax(preds) * softmax(labels)).sum(dim=-1).mean()


class BaseRNN(LightningModule):
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
        output, _, act = self(inputs[0], inputs[1])
        loss = self.criterion(output, place_outputs)
        # Weight regularization 
        loss += self._l2_loss() * self.options.weight_decay
        # loss += self.weight_decay * tf.reduce_sum(self.RNN.weights[1]**2)

        pred_pos = self.pc.get_nearest_cell_pos(act)
        err = torch.mean(torch.sqrt(torch.sum((pos - pred_pos)**2, dim=-1)))

        tensorboard_logs = {'train_loss': loss, 'train_err': err}
        return {'loss': loss, 'err': err, 'output': output, 'log': tensorboard_logs}

    def configure_optimizers(self):
        if self.options.optim == 'SGD':
            return optim.SGD(self.parameters(), lr=self.options.learning_rate, momentum=0.9)
        elif self.options.optim == 'Adam':
            return optim.Adam(self.parameters(), lr=self.options.learning_rate)
        elif self.options.optim == 'RMSProp':
            return optim.RMSprop(self.parameters(), lr=self.options.learning_rate)
        elif self.options.optim == 'HessianFree':
            return HessianFree(self.parameters(), lr=self.options.learning_rate) #  use_gnm=True, verbose=True

    def train_dataloader(self):
        dg = DataGenerator(self.options, self.pc, gpu=self.on_gpu, train=True)
        return dg

    def validation_step(self, batch, batch_idx):
        inputs, place_outputs, pos = batch
        output, _, act = self(inputs[0], inputs[1])
        loss = self.criterion(output, place_outputs)

        pred_pos = self.pc.get_nearest_cell_pos(act)
        err = torch.mean(torch.sqrt(torch.sum((pos - pred_pos)**2, dim=-1)))

        pos = pos.reshape(-1, pos.shape[-1])
        act = act.reshape(-1, act.shape[-1])
        return {'val_loss': loss, 'pos': pos, 'act': act, 'val_err': err, 'output': output}

    #TODO update the plot generation with bsorch's (seems faster)
    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        avg_err = torch.stack([x['val_err'] for x in outputs]).mean()
        # maybe only need to do outputs['val_loss']
        tensorboard_logs = {'val_loss': avg_loss, 'val_err': avg_err}
        # these are the full size of the epoch
        pos = torch.cat([x['pos'] for x in outputs], dim=0).to('cpu').detach().numpy()
        act = torch.cat([x['act'] for x in outputs], dim=0).to('cpu').detach().numpy()
        # TODO: fix these vars
        plots_dir = '.'
        epoch = 0
        # Save a picture of rate maps
        # save_ratemaps(self.model, self.trajectory_generator, self.options, step=tot_step)
        compute_ratemaps(pos, act, self.options, epoch=self.current_epoch) # TODO: specify range
        del pos, act
        return {'val_loss': avg_loss, 'val_err': avg_err, 'log': tensorboard_logs}

    def val_dataloader(self):
        # TODO: do a real train/val split
        dg = DataGenerator(self.options, self.pc, gpu=self.on_gpu, train=False)
        return dg

    def _l2_loss(self):
        return self.rnn.weight_hh_l0.norm(2)