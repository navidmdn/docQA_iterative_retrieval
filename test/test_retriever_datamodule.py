import pytest
from datamodules.retriever import RetrieverDataModule
from transformers import AutoTokenizer
import pathlib
import os


@pytest.mark.parametrize("dataset_path", ["data/hotpot/hotpot_dev_with_neg_v0.json",])
@pytest.mark.parametrize("max_len", [128,])
@pytest.mark.parametrize("batch_size", [10,])
def test_preprocessed_data_format(dataset_path, max_len, batch_size):
    current_dir = pathlib.Path(__file__).parent.resolve()
    preprocess_path = os.path.join(current_dir, "preprocessed_data")

    tokenizer = AutoTokenizer.from_pretrained("roberta-base")
    dm = RetrieverDataModule(
        tokenizer=tokenizer,
        train_path=dataset_path,
        dev_path=dataset_path,
        test_path=dataset_path,
        max_q_len=max_len,
        max_q_sp_len=max_len,
        max_c_len=max_len,
        preprocessed_data_dir=preprocess_path,
        batch_size=batch_size,
        device='cpu',
    )
    dm.prepare_data()
    dm.setup(stage='fit')
    dl = dm.train_dataloader()
    batch = next(iter(dl))

    assert batch['q_input_ids'].shape[0] == batch_size
    assert batch['q_input_ids'].shape[1] <= max_len





