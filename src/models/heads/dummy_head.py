import pytorch_lightning as pl
import torch.nn as nn

class Dummy(pl.LightningModule):
    def __init__(self,):
        super().__init__()
        self.net = nn.Linear(1, 1, bias=True, device=self.device)
        pass
    def forward(x)
        return self.net(x)
