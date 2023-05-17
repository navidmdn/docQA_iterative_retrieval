import torch
from transformers import Trainer as HFTrainer
from retriever.criterions import mhop_loss, mhop_eval
from typing import Dict, Any, Iterable
from transformers import get_cosine_schedule_with_warmup
from typing import Optional
from dataclasses import dataclass, field
from retriever.roberta_retriever import RobertaRetriever
import transformers
from retriever.hf_dataloader import DataArguments, DataModule
from transformers import DefaultDataCollator


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

    # def compute_metrics(eval_preds):
    #     result = {}
    #     preds, labels = eval_preds
    #
    #     # remov§ed the token distribution in custom eval loop so don't need the following line
    #     # preds = np.argmax(preds, axis=-1)
    #
    #     # todo: stop after eos token: isn't there a better way to do it?
    #     preds_list = []
    #     for p, l in zip(preds, labels):
    #         # last ignore token is the first prediction
    #         label_start = max(0, np.where(l != -100)[0][0]-1)
    #         p = p[label_start:]
    #         if tokenizer.eos_token_id in p:
    #             first_pad_idx = np.where(p == tokenizer.eos_token_id)
    #             if len(first_pad_idx) > 0 and first_pad_idx[0][0] < len(p)-1:
    #                 p[first_pad_idx[0][0]+1:] = tokenizer.pad_token_id
    #         preds_list.append(p)
    #
    #     decoded_preds = tokenizer.batch_decode(preds_list, skip_special_tokens=True)
    #     labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    #     decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    #
    #     rouge_result = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    #     exact_match_result = exact_match_metric.compute(predictions=decoded_preds, references=decoded_labels)
    #     if random() < 0.01:
    #         print(list(zip(decoded_labels, decoded_preds))[:5])
    #     generated_pred = np.where(labels == tokenizer.pad_token_id, 0, preds)
    #     prediction_lens = [np.count_nonzero(pred) for pred in generated_pred]
    #     rouge_result["gen_len"] = np.mean(prediction_lens)
    #
    #     result.update(rouge_result)
    #     result.update(exact_match_result)
    #
    #     return result
    data_module = DataModule(tokenizer=tokenizer, train_path=data_args.train_path, test_path=data_args.test_path,
                             dev_path=data_args.dev_path, batch_size=data_args.batch_size,
                             num_workers=data_args.num_workers, max_c_len=data_args.max_c_len,
                             max_q_len=data_args.max_q_len, max_q_sp_len=data_args.max_q_sp_len)

    ds_dict = data_module.load_dataset()
    trainer = Trainer(model=model, tokenizer=tokenizer, args=training_args, compute_metrics=None, **ds_dict)
    trainer.train()
    trainer.save_state()
    trainer.save_model(output_dir=training_args.output_dir)


if __name__ == "__main__":
    train()
