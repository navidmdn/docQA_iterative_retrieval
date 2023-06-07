import torch
from transformers import Trainer as HFTrainer
from retriever.criterions import mhop_loss, mhop_eval
from typing import Dict, Any, Iterable
from transformers import get_cosine_schedule_with_warmup
from typing import Optional
from tqdm import tqdm
from dataclasses import dataclass, field
from retriever.roberta_retriever import RobertaRetriever
import transformers
from retriever.hf_dataloader import DataArguments, DataModule
from transformers import DefaultDataCollator
from torch.utils.data import DataLoader, Dataset, RandomSampler, SequentialSampler
from typing import List
from transformers.trainer_utils import EvalLoopOutput, EvalPrediction
from transformers.trainer_pt_utils import IterableDatasetShard, nested_concat
from transformers.trainer import has_length, find_batch_size, denumpify_detensorize
import numpy as np


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="roberta-base")


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={"help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."},
    )
    output_dir: str = field(default="results")
    epochs: int = field(default=1)
    report_to: str = field(default="tensorboard")
    remove_unused_columns: bool = field(default=False)
    evaluation_strategy: str = field(default="steps")
    eval_steps: int = field(default=3)
    eval_accumulation_steps: int = field(default=1)
    use_mps_device: bool = field(default=False)
    per_device_train_batch_size: int = field(default=8)
    per_device_eval_batch_size: int = field(default=8)
    save_strategy: str = field(default="steps")
    save_steps: int = field(default=2)


class Trainer(HFTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
    ):
        super().__init__(model, args, data_collator, train_dataset, eval_dataset, tokenizer,
                         model_init, compute_metrics, callbacks, optimizers, preprocess_logits_for_metrics)

    def compute_loss(self, model, inputs, return_outputs=False):
        return mhop_loss(model, inputs, return_outputs=return_outputs)

    @staticmethod
    def get_average_metrics(metrics_list: List[Dict]):
        metrics = {}
        for key in metrics_list[0].keys():
            metrics[key] = np.mean([x[key] for x in metrics_list])
        return metrics

    def evaluation_loop(
        self,
        dataloader: DataLoader,
        description: str,
        prediction_loss_only: Optional[bool] = None,
        ignore_keys: Optional[List[str]] = None,
        metric_key_prefix: str = "eval",
    ) -> EvalLoopOutput:

        model = self._wrap_model(self.model, training=False, dataloader=dataloader)
        model.eval()
        batch_size = self.args.eval_batch_size
        eval_dataset = getattr(dataloader, "dataset", None)

        num_samples = len(eval_dataset)
        observed_num_examples = 0
        num_batches = num_samples//batch_size

        all_metrics = []
        losses_all = []

        for step, inputs in tqdm(enumerate(dataloader), total=num_batches):

            observed_batch_size = find_batch_size(inputs)
            observed_num_examples += observed_batch_size

            inputs = self._prepare_inputs(inputs)
            with torch.no_grad():
                with self.compute_loss_context_manager():
                    loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
                cur_eval_metrics = mhop_eval(outputs)
                losses_all.append(loss.mean().detach().cpu().numpy())
                all_metrics.append(cur_eval_metrics)

        metrics = self.get_average_metrics(all_metrics)
        metrics[f"{metric_key_prefix}_loss"] = np.mean(losses_all)
        metrics = denumpify_detensorize(metrics)

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_"):
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        if num_samples == 0 and observed_num_examples > 0:
            num_samples = observed_num_examples

        return EvalLoopOutput(
            predictions=None,
            label_ids=None,
            metrics=metrics,
            num_samples=num_samples)


def train():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    config = transformers.AutoConfig.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
    )

    model = RobertaRetriever(config, model_args.model_name_or_path, training_args.cache_dir)

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=training_args.cache_dir,
        model_max_length=training_args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    def compute_metrics(eval_preds):
        result = {}
        return {}

    data_module = DataModule(tokenizer=tokenizer, train_path=data_args.train_path, test_path=data_args.test_path,
                             dev_path=data_args.dev_path, batch_size=data_args.batch_size,
                             num_workers=data_args.num_workers, max_c_len=data_args.max_c_len,
                             max_q_len=data_args.max_q_len, max_q_sp_len=data_args.max_q_sp_len,
                             cache_dir=training_args.cache_dir)

    ds_dict = data_module.load_dataset()
    # print available cuda devices
    print(f"train size: {len(ds_dict['train_dataset'])} validation size: {len(ds_dict['eval_dataset'])}")
    print(torch.cuda.device_count(), torch.cuda.is_available())
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=None, **ds_dict)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
