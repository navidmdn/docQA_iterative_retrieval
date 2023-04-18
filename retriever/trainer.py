from torch import optim, nn

import lightning.pytorch as pl
from retriever.roberta_retriever import RobertaRetriever
from transformers import AutoConfig, AutoTokenizer


class RetrieverTrainer(pl.LightningModule):
    def __init__(self, retriever_model_name, lr=1e-3, **kwargs):
        super().__init__()
        self.retriever_model_name = retriever_model_name
        self.lr = lr

        model_config = AutoConfig.from_pretrained(self.retriever_model_name)
        self.retriever = RobertaRetriever(model_config, model_name=self.retriever_model_name)

    def training_step(self, batch, batch_idx):


    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.lr)
        return optimizer


# init the autoencoder