import torch
from torch import optim, nn

import lightning as pl
from retriever.roberta_retriever import RobertaRetriever
from transformers import AutoConfig, AutoTokenizer
from retriever.criterions import mhop_loss, mhop_eval
from typing import Dict, Any, Iterable
from transformers import get_cosine_schedule_with_warmup

Batch = Dict[str, Any]


class RetrieverModule(pl.LightningModule):
    def __init__(self, retriever_model_name, lr=1e-3, warmup_steps=100, huggingface_cache_dir=None, **kwargs):
        super().__init__()
        self.retriever_model_name = retriever_model_name
        self.lr = lr
        self.warmup_steps = warmup_steps
        self.huggingface_cache_dir = huggingface_cache_dir
        model_config = AutoConfig.from_pretrained(self.retriever_model_name, cache_dir=self.huggingface_cache_dir)
        self.retriever = RobertaRetriever(model_config, model_name=self.retriever_model_name,
                                          cache_dir=self.huggingface_cache_dir)
        self.tokenizer = AutoTokenizer.from_pretrained(self.retriever_model_name, cache_dir=self.huggingface_cache_dir)
        self.retriever.to(self.device)

    def training_step(self, batch: Batch, batch_idx: int):
        self.retriever.train()
        loss = mhop_loss(self.retriever, batch)
        self.log('train_loss', loss, prog_bar=True, logger=True, on_step=True)
        self.log('lr_optim', self.trainer.optimizers[0].param_groups[0]['lr'], prog_bar=True, logger=True, on_step=True)

        return loss

    def validation_step(self, batch: Batch, batch_idx: int):
        self.retriever.eval()
        with torch.no_grad():
            # todo: duplicate code; change loss fc
            loss = mhop_loss(self.retriever, batch)
            metrics = mhop_eval(self.retriever(batch))

        self.log('val_loss', loss, logger=True)
        self.log_dict(metrics, logger=True)

    def on_train_start(self) -> None:
        if self.logger is not None:
            self.logger.log_hyperparams(self.hparams)  # type: ignore
            assert self.trainer is not None
            print(f"Logging to {self.trainer.log_dir}")

    def get_optimizers(
            self,
            parameters: Iterable[torch.nn.parameter.Parameter],
            lr: float,
            num_warmup_steps: int,
            num_training_steps: int,
    ) -> Dict[str, Any]:
        """
        Get an AdamW optimizer with linear learning rate warmup and cosine decay.
        """
        optimizer = torch.optim.AdamW(parameters, lr=lr)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step",
            },
        }

    def configure_optimizers(self) -> Dict[str, Any]:
        assert self.trainer is not None
        if self.trainer.max_steps != -1:
            max_steps = self.trainer.max_steps
        else:
            max_steps = (
                self.trainer.max_epochs
                * len(self.trainer.datamodule.train_dataloader())  # type: ignore
                // self.trainer.accumulate_grad_batches
            )
        return self.get_optimizers(
            self.parameters(),
            self.lr,
            self.warmup_steps,
            max_steps,
        )

    def lr_scheduler_step(self, scheduler, metric):
        scheduler.step()