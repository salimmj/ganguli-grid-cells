from pytorch_lightning import Trainer
from models import VanillaRNN, LSTM

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

options = AttrDict()

options.RNN_type = 'RNN'
options.sequence_length = 20 # length of rat trajectory
options.batch_size = 200
options.activation = 'relu' # activation of the Recurrent Neural Network cell
options.nG = 512 # size of the hidden layer (grid cells)
options.nP = 512*2 # size of the output layers (number of place cells to predict)
options.sigma = 14 # place cell sigma
options.DoG = True # Difference of Gaussians for the place cell activations
options.surround_scale = 2 # surround scale between the two gaussians (sigma_2 = sqrt(sigma**2 * surround_scale))
options.learning_rate = 1e-4
options.box_width = 220 # width of the box in which the rat walks
options.box_height = 220 # height of the box in which the rat walks
options.weight_decay = 0 # weight decay (if 0, we do not use L2 regularization)
options.train_epoch_size = 10000 # number of mini batches in one epoch of training
options.val_epoch_size = 50 # number of mini batches to use for validation and generating plots
options.periodic = False
options.loss = 'CE' # cross entropy (CE) or MSE

def generate_run_ID(options):
    ''' 
    Create a unique run ID from the most relevant
    parameters. Remaining parameters can be found in 
    params.npy file. 
    '''
    params = [
        'steps', str(options.sequence_length),
        'batch', str(options.batch_size),
        options.RNN_type,
        str(options.nG),
        options.activation,
        'nP', str(options.nP),
        'sigma', str(options.sigma),
        'DoG', str(options.DoG),
        'lr', str(options.learning_rate),
        'weight_decay', str(options.weight_decay),
        'bw', str(options.box_width),
        'bh', str(options.box_height),
        'loss', options.loss,
        ]
    separator = '_'
    run_ID = separator.join(params)
    run_ID = run_ID.replace('.', '')
    return run_ID

run_ID = generate_run_ID(options)
options.run_ID = run_ID

models = {'RNN': VanillaRNN, 'LSTM': LSTM}

model = models[options.RNN_type](options)

from pytorch_lightning.callbacks import ModelCheckpoint

checkpoint_callback = ModelCheckpoint(
    filepath='./experiments/'+run_ID+'/{epoch}-{val_loss:.2f}',
    verbose=True,
    monitor='val_loss',
    mode='min'
)

# most basic trainer, uses good defaults
trainer = Trainer(gpus=1, num_nodes=1, nb_sanity_val_steps=0, checkpoint_callback=checkpoint_callback, track_grad_norm=2, log_gpu_memory=True)
trainer.fit(model)