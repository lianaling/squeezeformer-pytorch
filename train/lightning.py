from mimetypes import init
import pytorch_lightning as pl
from squeezeformer.model import Squeezeformer

class LitAutoEncoder(pl.LightningModule):
    def __init__(self):
        super().__init__()
        self.model = Squeezeformer(2)

    def forward(self, x):
        # Inference/prediction
        pass

    def training_step(self, batch, batch_idx):
        # Training loop