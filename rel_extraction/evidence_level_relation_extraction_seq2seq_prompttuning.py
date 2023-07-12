import os

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer, default_data_collator, get_linear_schedule_with_warmup
from peft import get_peft_model, PromptTuningConfig, TaskType, PromptTuningInit
from torch.utils.data import DataLoader
from tqdm import tqdm
from datasets import load_dataset
import transformers

os.environ["TOKENIZERS_PARALLELISM"] = "false"

device = "mps"
model_name_or_path = "t5-small"
tokenizer_name_or_path = "t5-small"

checkpoint_name = "evidence_rels.pt"
text_column = "input"
label_column = "output"

max_length = 256
lr = 1e-4
num_epochs = 10
batch_size = 8

peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_2_SEQ_LM,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=20,
    prompt_tuning_init_text="extract the relations in this text?\n",
    inference_mode=False,
    tokenizer_name_or_path=model_name_or_path,
)

model = AutoModelForSeq2SeqLM.from_pretrained(model_name_or_path)
model = get_peft_model(model, peft_config)
model.print_trainable_parameters()


dataset = load_dataset("json", data_files="data/Re-DocRED-main/evidence_rels.json")
dataset = dataset['train'].train_test_split(test_size=0.02, seed=42, shuffle=True)
dataset['valid'] = dataset['test']
del(dataset['test'])

tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

print(dataset)
print(dataset['train'][0])


def preprocess_function(examples):
    inputs = examples[text_column]
    targets = examples[label_column]
    model_inputs = tokenizer(inputs, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt")
    labels = tokenizer(
        targets, max_length=max_length, padding="max_length", truncation=True, return_tensors="pt"
    )
    labels = labels["input_ids"]
    labels[labels == tokenizer.pad_token_id] = -100
    model_inputs["labels"] = labels
    return model_inputs

train_dataset = dataset['train'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Running tokenizer on dataset",
)

eval_dataset = dataset['valid'].map(
    preprocess_function,
    batched=True,
    remove_columns=dataset["train"].column_names,
    desc="Running tokenizer on dataset",
)


def compute_metrics(preds):
    preds, labels = preds
    return {}


trainer = transformers.Trainer(
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        args=transformers.TrainingArguments(
            per_device_train_batch_size=8,
            gradient_accumulation_steps=8,
            warmup_steps=10,
            num_train_epochs=num_epochs,
            learning_rate=lr,
            fp16=False,
            logging_steps=10,
            optim="adamw_torch",
            evaluation_strategy="steps",
            save_strategy="steps",
            eval_steps=1,
            use_mps_device=True,
            save_steps=50,
            output_dir='results/t5',
            save_total_limit=1,
            load_best_model_at_end=True,
            report_to="tensorboard",#"wandb" if use_wandb else None,
            # run_name=wandb_run_name if use_wandb else None,
        ),
        data_collator=default_data_collator
    )

trainer.train()
