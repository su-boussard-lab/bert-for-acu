"""
Traininig script for ACU classification with deep NLP models
Taken from this tutorial: https://pytorch.org/tutorials/beginner/text_sentiment_ngrams_tutorial.html

Author:
Claudio Fanconi
fanconic@stanford.edu / fanconic@ethz.ch / fanconic@gmail.com
"""

import os
from typing import List
import torch
from torch import nn
import random
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from src.layers.consisten_dropout import patch_module
from src.data.dataloader import ACUDataLoader
from src.model.model import (
    ACUClassifier,
    ACUMultiModalClassifier,
    ACUMultiModalAttentionClassifier,
    ACUMMSequenceAttentionClassifier,
)
from src.optimizer.optimizer import OptimizerWrapper
from torch.utils.data import DataLoader
from src.utils.config import config
from src.utils.logger import log
from src.data.dataset import custom_collate
import IPython
from sklearn.calibration import calibration_curve
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    log_loss,
    average_precision_score,
    plot_precision_recall_curve,
)
import calibration as cal
import matplotlib.pyplot as plt
from spacecutter.losses import CumulativeLinkLoss


def set_seed(seed: int) -> None:
    """Set all random seeds
    Args:
        seed (int): integer for reproducible experiments
    returns:
        None
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True


def bootstrap(df: pd.DataFrame, func=roc_auc_score):
    """Bootstrap for calculating the confidence interval of a metric function
    Args:
        df (pd.DataFrame): dataframe containing 'predictions' and ' outcomes'
        func (Callable): metric function that takes (y_true, y_pred) as parameters
    Returns:
        lower, upper 95% confidence interval
    """
    aucs = []
    for i in range(1000):  # change back to 10000
        sample = df.sample(
            n=df.shape[0] - int(df.shape[0] / 5), random_state=i
        )  # take 80% for the bootstrap
        aucs.append(func(sample["outcomes"], sample["predictions"]))

    return np.percentile(np.array(aucs), 2.5), np.percentile(np.array(aucs), 97.5)


def ece(y_true: np.array, y_proba: np.array):
    """computes the calibration error
    Args:
        y_true(np.array): true labels
        y_pred(np.array): predicted probabilities
    Returns:
        the calibration error of the model
    """
    return cal.get_calibration_error(y_proba, y_true.astype(int))


def prediction_entropy(y_true: np.ndarray, y_proba: np.ndarray) -> float:
    """computes the entropy of the predictions.
    Args:
        y_true (np.ndarray): ground thruths
        y_proba (np.ndarray): predicted probabilities
    """
    return -np.mean(y_proba * np.log(y_proba) + (1 - y_proba) * np.log(1 - y_proba))


def set_dropout_p(m: nn.Module, p: float) -> None:
    """Set the dropout rate of all the model
    Args:
        m (nn.module): model
        p (float): dropout probability
    """
    if isinstance(m, nn.Dropout):
        m.p = p


def enable_mc_dropout(m: nn.Module) -> None:
    """Function to enable the dropout layers during test-time
    Args:
        m (nn.Module): network module
    Returns:
        None
    """
    if isinstance(m, nn.Dropout):
        m.train()


def main(random_state: int = 42) -> None:
    """Main function which trains the deep learning model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """
    set_seed(random_state)
    use_cuda = torch.cuda.is_available()
    print(f"Cuda is available: { use_cuda}")
    device = torch.device("cuda" if use_cuda else "cpu")

    multimodal = config.use_tabular

    # import BERT backbone architecture
    tokenizer = AutoTokenizer.from_pretrained(config.model.bert_model, pad_token_id=0)
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
    bert = AutoModel.from_pretrained(config.model.bert_model)
    bert.resize_token_embeddings(len(tokenizer))

    # build the tokenized vocabulary:
    data_loader = ACUDataLoader(
        data_path=config.data.data_path,
        label_path=config.data.label_path,
        label_type=config.data.label_type,
        train_ids=config.data.train_ids,
        test_ids=config.data.test_ids,
        max_length=config.data.max_length,
        max_words=config.data.max_words,
        tokenizer=tokenizer,
        test_only=True,
        use_tabular=multimodal,
        tab_path=config.data.tabular_data_path,
        ordinal_regression=config.ordinal_regression,
    )
    _, test, num_class = data_loader.get_train_test_data()

    print(f"##################### Test Dataset size : [{len(test)}]")
    print(f"##################### class size : [{num_class}]")

    if multimodal:
        if config.model.fusion_model.lower() == "logistic":
            ACUModel = ACUMultiModalClassifier
        elif config.model.fusion_model.lower() == "attention":
            ACUModel = ACUMultiModalAttentionClassifier
        elif config.model.fusion_model.lower() == "sequence_attention":
            ACUModel = ACUMMSequenceAttentionClassifier
        else:
            raise ValueError(f"'{self_type}' is an unknown network type")
        tabular_size = data_loader.get_tabular_size()
    else:
        ACUModel = ACUClassifier
        tabular_size = None

    model = ACUModel(
        bert=bert,
        num_class=num_class,
        bert_finetuning=config.model.bert_finetuning,
        dropout_p=config.model.dropout_p,
        cls_pooling=config.model.cls_pooling,
        tabular_size=tabular_size,
        intermediate_mlp_size=config.model.intermediate_mlp_size,
        ordinal_regression=config.ordinal_regression,
    )

    # move model to device
    model.to(device)

    test_loader = DataLoader(
        dataset=test,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )

    # loss function
    if config.ordinal_regression:
        criterion = CumulativeLinkLoss()
    else:
        criterion = nn.BCELoss()

    def validation(loader: DataLoader, mc_dropout: bool = False) -> tuple:
        """Validation loop
        Args:
            loader (Dataloader): the test data loader
            mc_dropout (bool, False): flag if to perform montecarlo dropout inference
        Returns:
            y_preds: predictions
            y_true: ground truth labels
        """
        model.eval()

        # Set dropout layers for MC-Dropout.
        if config.model.mc_dropout:
            print("##################### Patched Monte Carlo Dropout modules")
            model.apply(lambda m: enable_mc_dropout(m))

        with torch.no_grad():
            y_preds = []
            y_true = []
            for batch in loader:
                batch = tuple(
                    t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
                )
                input_dict, y, chunks = batch
                if mc_dropout:
                    y_pred_mc = []
                    for _ in range(config.model.mc_samples):
                        y_pred = model(
                            **input_dict,
                            chunks=chunks,
                            boost=config.train.boost,
                            indipendent_chunks=config.indipendent_chunks,
                        )
                        y_pred_mc.append(y_pred)
                    # take the MLE of the output predictive distribution
                    y_pred = torch.stack(y_pred_mc).mean(0)
                else:
                    y_pred = model(
                        **input_dict,
                        chunks=chunks,
                        boost=config.train.boost,
                        indipendent_chunks=config.indipendent_chunks,
                    )
                y_preds.append(y_pred)
                y_true.append(y)

        return torch.cat(y_preds), torch.cat(y_true)

    def log_test_results(test_loader, model) -> None:
        """Logs the testing results at the end of training.
            The decorator at the top of this function is crucial
        Args:
            engine:
        returns:
            None
        """
        paths = os.listdir(config.model.save_path + config.name)
        paths = [path for path in paths if ".pt" in path]
        paths = sorted(paths, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        model.load_state_dict(
            torch.load(os.path.join(config.model.save_path, config.name, paths[-1]))
        )

        model.apply(lambda m: set_dropout_p(m, p=config.model.dropout_p))
        print(f"Load best model '{paths[0]}'")
        y_preds, y_true = validation(test_loader, config.model.mc_dropout)
        y_preds, y_true = y_preds.cpu().numpy(), y_true.cpu().numpy()

        if config.ordinal_regression:
            results_30 = pd.DataFrame(
                {"predictions": y_preds[:, 3], "outcomes": (y_true > 2).squeeze()}
            )
            results_180 = pd.DataFrame(
                {
                    "predictions": y_preds[:, 2:4].sum(axis=1),
                    "outcomes": (y_true > 1).squeeze(),
                }
            )
            results_365 = pd.DataFrame(
                {
                    "predictions": y_preds[:, 1:4].sum(axis=1),
                    "outcomes": (y_true > 0).squeeze(),
                }
            )
            calculate_results(results_30, "ANY_30")
            calculate_results(results_180, "ANY_180")
            calculate_results(results_365, "ANY_365")

        else:
            results = pd.DataFrame({"predictions": y_preds, "outcomes": y_true})
            calculate_results(results, config.data.label_type)

    def calculate_results(results: pd.DataFrame, name: str) -> None:
        """Prints all the results with 95% confidence interval, bootstrapped from the test predictions
        Furthermore, also creates a calibration plot
        Args:
            results (pd.DataFrame): dataframe with groundtruths and predictions.
            name (str): name of the experiment
        Returns:
            None, only prints and creates a pdf üplot of the calibration curve
        """
        # compute output
        print(f"Experiment: {name}")
        y_true = results["outcomes"]
        y_preds = results["predictions"]
        low_95, high_95 = bootstrap(results, func=roc_auc_score)
        print(
            f"AUROC: {roc_auc_score(y_true, y_preds):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
        )
        low_95, high_95 = bootstrap(results, func=average_precision_score)
        print(
            f"AUPRC: {average_precision_score(y_true, y_preds):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
        )
        low_95, high_95 = bootstrap(results, func=log_loss)
        print(
            f"LL: {log_loss(y_true, y_preds):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
        )
        low_95, high_95 = bootstrap(results, func=ece)
        print(f"ECE: {ece(y_true, y_preds):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})")
        low_95, high_95 = bootstrap(results, func=prediction_entropy)
        print(
            f"Entropy: {prediction_entropy(y_true, y_preds):.3f} (95%-CI:{low_95:.3f},{high_95:.3f})"
        )

        # Creating Calibration Curve
        x, y = calibration_curve(y_true, y_preds, n_bins=20)
        plt.plot([0, 1], [0, 1], linestyle="--", label="Ideally Calibrated")
        plt.plot(y, x, marker=".", label=config.name)
        plt.savefig(
            os.path.join(
                "./experiments",
                config.name,
                config.name + f"_calibration_plot_{name}.pdf",
            )
        )
        print("\n")

    # At last, we can run training
    log_test_results(test_loader, model)


if __name__ == "__main__":
    print(f"Running testing for experiment: {config.name}")
    main(random_state=config.seed)
