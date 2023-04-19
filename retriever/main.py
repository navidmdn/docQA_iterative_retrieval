from lightning.pytorch.cli import LightningCLI
from retriever.data_module import RetrieverDataModule
from retriever.trainer_module import RetrieverModule


# class CLI(LightningCLI):
    # pass
    #
    # def add_arguments_to_parser(self, parser: Any) -> None:
    #     parser.link_arguments("model.model_name", "data.model_name")
    #     parser.link_arguments("model.stepwise", "data.stepwise")
    #     parser.link_arguments("data.dataset", "model.dataset")
    #     parser.link_arguments("data.max_input_len", "model.max_input_len")


def main() -> None:
    cli = LightningCLI(RetrieverModule, RetrieverDataModule, save_config_kwargs={"overwrite": True})
    print("Configuration: \n", cli.config)


if __name__ == "__main__":
    main()
