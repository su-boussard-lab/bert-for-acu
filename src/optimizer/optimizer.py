"""
Defines the optimizers based on the config file
"""
from torch import optim
from src.model.model import (
    ACUClassifier,
    ACUMultiModalClassifier,
    ACUMultiModalAttentionClassifier,
    ACUMMSequenceAttentionClassifier,
)


class OptimizerWrapper:
    def __init__(self, model, config) -> None:
        parameter_configs = [
            {
                "params": model.fc.parameters(),
                "lr": config.optimizer.lr,
            },
            {
                "params": model.bert.parameters(),
                "lr": config.optimizer.lr_main,
            },
        ]
        # Make the right parameter configs for the different models
        if hasattr(model, "MLP"):
            parameter_configs += [
                {
                    "params": model.MLP.parameters(),
                    "lr": config.optimizer.lr,
                },
            ]
        if hasattr(model, "trans_tab_with_nlp"):
            parameter_configs += [
                {
                    "params": model.trans_tab_with_nlp.parameters(),
                    "lr": config.optimizer.lr,
                },
            ]
        if hasattr(model, "trans_nlp_with_tab"):
            parameter_configs += [
                {
                    "params": model.trans_nlp_with_tab.parameters(),
                    "lr": config.optimizer.lr,
                },
            ]
        if config.ordinal_regression:
            parameter_configs += [
                {
                    "params": model.ordinallink.parameters(),
                    "lr": config.optimizer.lr,
                },
            ]
        if config.optimizer.name.lower() == "adam":
            self.optimizer = optim.Adam(
                parameter_configs,
                betas=config.optimizer.betas,
                weight_decay=config.optimizer.weight_decay,
            )
        elif config.optimizer.name.lower() == "sgd":
            self.optimizer = optim.SGD(
                parameter_configs,
                momentum=config.optimizer.momentum,
                weight_decay=config.optimizer.weight_decay,
            )
        else:
            raise NotImplementedError

    def step(self) -> None:
        """Take an optimization step
        Args:

        Returns:
            None
        """
        self.optimizer.step()

    def zero_grad(self) -> None:
        """Resets the gradient of the optimizer"""
        self.optimizer.zero_grad()
