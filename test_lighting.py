from pytorch_lightning import Trainer
from models import VanillaRNN

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

options = AttrDict()


options.sequence_length = 50
options.periodic = False
options.batch_size = 200
options.box_width = 220
options.box_height = 220
options.train_epoch_size = 10000
options.val_epoch_size = 2

model = VanillaRNN(options)

# most basic trainer, uses good defaults
trainer = Trainer(gpus=1, num_nodes=1, nb_sanity_val_steps=0)
trainer.fit(model)