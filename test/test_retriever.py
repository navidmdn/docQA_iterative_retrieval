import pytest
import torch

from retriever.data_module import RetrieverDataModule
from retriever.criterions import mhop_loss
from transformers import AutoTokenizer
import pathlib
import os



@pytest.mark.parametrize("dataset_path", ["data/hotpot/hotpot_dev_with_neg_v0.json",])
@pytest.mark.parametrize("max_len", [128,])
@pytest.mark.parametrize("batch_size", [10,])
def test_preprocessed_data_format(dataset_path, max_len, batch_size):
    current_dir = pathlib.Path(__file__).parent.resolve()
    preprocess_path = os.path.join(current_dir, "preprocessed_data")

    dm = RetrieverDataModule(
        tokenizer_cp='roberta-base',
        train_path=dataset_path,
        dev_path=dataset_path,
        test_path=dataset_path,
        max_q_len=max_len,
        max_q_sp_len=max_len,
        max_c_len=max_len,
        preprocessed_data_dir=preprocess_path,
        batch_size=batch_size,
    )
    dm.prepare_data()
    dm.setup(stage='fit')
    dl = dm.train_dataloader()
    batch = next(iter(dl))

    assert batch['q_input_ids'].shape[0] == batch_size
    assert batch['q_input_ids'].shape[1] <= max_len


def test_retriever_loss_function_no_momentum():
    B = 2
    h = 5

    q = torch.rand(B, h)
    c1 = torch.rand(B, h)
    c2 = torch.rand(B, h)
    neg_1 = torch.rand(B, h)
    neg_2 = torch.rand(B, h)
    q_sp1 = torch.rand(B, h)

    batch_output = {'q': q, 'c1': c1, "c2": c2, "neg_1": neg_1, "neg_2": neg_2, "q_sp1": q_sp1}
    model = lambda x: x
    loss = mhop_loss(model, batch_output)


def test_if_padding_affects_cls():
    from retriever.roberta_retriever import RobertaRetriever
    from transformers import AutoConfig

    config = AutoConfig.from_pretrained('roberta-base')
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
    model = RobertaRetriever(config, 'roberta-base')
    model.eval()

    text = "This is a test"
    encs1 = tokenizer(text, return_tensors='pt', padding='max_length', max_length=128, truncation=True)
    encs2 = tokenizer(text, return_tensors='pt', padding='max_length', max_length=256, truncation=True)

    with torch.no_grad():
        cls1 = model.encode_seq(encs1['input_ids'], encs1['attention_mask'])
        cls2 = model.encode_seq(encs2['input_ids'], encs2['attention_mask'])

    assert torch.allclose(cls1, cls2)






