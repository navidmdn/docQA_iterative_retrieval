import torch
from torch import optim, nn

import pytorch_lightning as pl
from retriever.roberta_retriever import RobertaRetriever
from transformers import AutoConfig, AutoTokenizer
from retriever.criterions import mhop_loss
from typing import Dict, Any

Batch = Dict[str, Any]


class RetrieverModule(pl.LightningModule):
    def __init__(self, retriever_model_name, lr=1e-3, **kwargs):
        super().__init__()
        self.retriever_model_name = retriever_model_name
        self.lr = lr
        model_config = AutoConfig.from_pretrained(self.retriever_model_name)
        self.retriever = RobertaRetriever(model_config, model_name=self.retriever_model_name)
        self.retriever.to(self.device)

    def training_step(self, batch: Batch, batch_idx: int):
        self.retriever.train()
        loss = mhop_loss(self.retriever, batch)
        self.log('train_loss', loss)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        self.retriever.eval()
        with torch.no_grad():
            loss = mhop_loss(self.retriever, batch)
            self.log('val_loss', loss)

    def on_train_start(self) -> None:
        pass

    def on_train_epoch_end(self) -> None:
        pass

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer
