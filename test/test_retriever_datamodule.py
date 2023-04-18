import pytest
from datamodules.retriever import RetrieverDataModule
from transformers import AutoTokenizer
import pathlib
import os


@pytest.mark.parametrize("dataset_path", ["data/hotpot/hotpot_dev_with_neg_v0.json",])
@pytest.mark.parametrize("max_len", [128,])
def test_preprocessed_data_format(dataset_path, max_len):
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
        batch_size=10,
        device='cpu',
    )
    dm.prepare_data()
    dm.setup(stage='fit')
    dl = dm.train_dataloader()
    batch = next(iter(dl))
    assert batch['q_codes']['input_ids'].shape == (10, max_len)
    assert batch['q_codes']['attention_mask'].shape == (10, max_len)
    assert batch['q_sp_codes']['input_ids'].shape == (10, max_len)
    assert batch['q_sp_codes']['attention_mask'].shape == (10, max_len)
    assert batch['start_para_codes']['input_ids'].shape == (10, max_len)
    assert batch['start_para_codes']['attention_mask'].shape == (10, max_len)
    assert batch['bridge_para_codes']['input_ids'].shape == (10, max_len)
    assert batch['bridge_para_codes']['attention_mask'].shape == (10, max_len)
    assert batch['neg_codes_1']['input_ids'].shape == (10, max_len)
    assert batch['neg_codes_1']['attention_mask'].shape == (10, max_len)
    assert batch['neg_codes_2']['input_ids'].shape == (10, max_len)
    assert batch['neg_codes_2']['attention_mask'].shape == (10, max_len)
    assert batch['answers'].shape == (10,)






