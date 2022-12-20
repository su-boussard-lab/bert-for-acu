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
from src.data.dataset import custom_collate
from torch_scatter import scatter_mean
import IPython

from ignite.engine import Engine, Events
from ignite.metrics import Loss
from ignite.contrib.metrics import ROC_AUC, AveragePrecision
from ignite.handlers import ModelCheckpoint
from ignite.contrib.handlers import TensorboardLogger, global_step_from_engine
from ignite.handlers import ModelCheckpoint, EarlyStopping
from ignite.contrib.handlers.param_scheduler import LRScheduler
from torch.optim.lr_scheduler import StepLR

from src.layers.losses import CumulativeLinkLoss, WeightedBCE
from src.layers.callbacks import AscensionCallback
from accelerate import Accelerator, DistributedDataParallelKwargs


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


def set_dropout_p(m: nn.Module, p: float) -> None:
    """Set the dropout rate of all the model
    Args:
        m (nn.module): model
        p (float): dropout probability
    """
    if isinstance(m, nn.Dropout):
        m.p = p


def main(random_state: int = 42) -> None:
    """Main function which trains the deep learning model
    Args:
        random_state (int, 42): random state for reproducibility
    Returns:
        None
    """
    distributed_args = DistributedDataParallelKwargs(find_unused_parameters=True)
    accelerator = Accelerator(kwargs_handlers=[distributed_args])
    accelerator.print(f"Running training for experiment: {config.name}")
    set_seed(random_state)
    use_cuda = torch.cuda.is_available()
    accelerator.print(f"Cuda is available: {use_cuda}")
    if use_cuda:
        accelerator.print(f"Nr of GPUs: {torch.cuda.device_count()}")
    device = torch.device("cuda" if use_cuda else "cpu")

    multimodal = config.use_tabular
    pretrained = config.pretrained

    # import BERT backbone architecture
    tokenizer = AutoTokenizer.from_pretrained(
        config.model.bert_model, pad_token_id=0, return_token_type_ids=True
    )
    bert = AutoModel.from_pretrained(config.model.bert_model)

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
        use_tabular=multimodal,
        tab_path=config.data.tabular_data_path,
        ordinal_regression=config.ordinal_regression,
    )
    train, _, num_class = data_loader.get_train_test_data()
    if config.train.weighted_loss:
        if config.ordinal_regression:
            weight_vector = torch.tensor(
                [
                    1 - ((train.labels == i).sum() / len(train.labels))
                    if i == 0
                    else 1
                    - (
                        ((train.labels <= i) & (train.labels > 0)).sum()
                        / len(train.labels)
                    )
                    for i in range(num_class)
                ]
            ).to(device)
            accelerator.print(weight_vector)
        else:
            event_rate = train.labels.mean()
            weight_vector = torch.tensor([event_rate, 1 - event_rate]).to(device)
            accelerator.print(weight_vector)

    # obtain training indices that will be used for validation
    train, valid = data_loader.get_train_val_data(train, config.data.val_size)

    accelerator.print(f"##################### Train Dataset size : [{len(train)}]")
    accelerator.print(f"##################### Valid Dataset size : [{len(valid)}]")
    accelerator.print(f"##################### class size : [{num_class}]")

    if multimodal:
        if config.model.fusion_model.lower() == "logistic":
            ACUModel = ACUMultiModalClassifier
        elif config.model.fusion_model.lower() == "attention":
            ACUModel = ACUMultiModalAttentionClassifier
        elif config.model.fusion_model.lower() == "sequence_attention":
            ACUModel = ACUMMSequenceAttentionClassifier
        else:
            raise ValueError(
                f"'{config.model.fusion_model}' is an unknown network type"
            )
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

    model.apply(lambda m: set_dropout_p(m, p=config.model.dropout_p))

    if pretrained:

        paths = os.listdir(config.model.save_path + config.model.pretrained_path)
        paths = [path for path in paths if ".pt" in path]

        bert_dict = torch.load(
            os.path.join(
                config.model.save_path, config.model.pretrained_path, paths[-1]
            )
        )
        bert_dict = {
            k.replace("bert.", ""): v for k, v in bert_dict.items() if "bert" in k
        }

        # 3. load the new state dict
        model.bert.load_state_dict(bert_dict)

    train_loader = DataLoader(
        dataset=train,
        batch_size=config.train.batch_size,
        shuffle=True,
        collate_fn=custom_collate,
    )
    val_loader = DataLoader(
        dataset=valid,
        batch_size=config.train.batch_size,
        shuffle=False,
        collate_fn=custom_collate,
    )

    # prepare teh optimizer
    optimizer = OptimizerWrapper(model, config)

    # Accelerate
    train_loader, val_loader, model, optimizer = accelerator.prepare(
        train_loader,
        val_loader,
        model,
        optimizer,
    )

    step_scheduler = StepLR(
        optimizer.optimizer,
        step_size=config.scheduler.step_size,
        gamma=config.scheduler.lr_reduce_factor,
    )
    lr_scheduler = LRScheduler(step_scheduler)

    # loss function
    if config.ordinal_regression:
        if config.train.weighted_loss:
            criterion = CumulativeLinkLoss(class_weights=weight_vector)
        else:
            criterion = CumulativeLinkLoss()
    else:
        if config.train.weighted_loss:
            criterion = WeightedBCE(class_weights=weight_vector)
        else:
            criterion = nn.BCELoss()

    if config.ordinal_regression:
        val_metrics = {
            "roc_auc_30": ROC_AUC(
                output_transform=lambda x: (x[0][:, 3], x[1] > 2), device=device
            ),
            "roc_auc_180": ROC_AUC(
                output_transform=lambda x: (x[0][:, 2:4].sum(axis=1), x[1] > 1),
                device=device,
            ),
            "roc_auc_365": ROC_AUC(
                output_transform=lambda x: (x[0][:, 1:4].sum(axis=1), x[1] > 0),
                device=device,
            ),
            "prc_auc_30": AveragePrecision(
                output_transform=lambda x: (x[0][:, 3], x[1] > 2), device=device
            ),
            "prc_auc_180": AveragePrecision(
                output_transform=lambda x: (x[0][:, 2:4].sum(axis=1), x[1] > 1),
                device=device,
            ),
            "prc_auc_365": AveragePrecision(
                output_transform=lambda x: (x[0][:, 1:4].sum(axis=1), x[1] > 0),
                device=device,
            ),
            "loss": Loss(criterion),
        }
    else:
        val_metrics = {
            "roc_auc": ROC_AUC(device=device),
            "prc_auc": AveragePrecision(device=device),
            "loss": Loss(criterion),
        }

    def train_step(engine, batch: List) -> torch.FloatTensor:
        """Training step that is passed to the pytorch ignite engine
        Args:
            engine:
            batch (List): current batch of inputs, mask, labels
        Return
            loss (torch.FloatTensor): loss of the current batch
        """
        model.train()
        optimizer.zero_grad()
        batch = tuple(t.to(device) if isinstance(t, torch.Tensor) else t for t in batch)
        input_dict, y, chunks = batch
        y_pred = model(
            **input_dict,
            chunks=chunks,
            boost=config.train.boost,
            indipendent_chunks=config.indipendent_chunks,
        )
        if config.indipendent_chunks:
            new_y = []
            for i, y_ in enumerate(y):
                new_y += [y_] * int(chunks[i])
            y = torch.Tensor(new_y).to(device, dtype=torch.int64).unsqueeze(1)

        if hasattr(model, "module"):
            L1 = model.module.get_L1_loss()
        else:
            L1 = model.get_L1_loss()
        loss = criterion(y_pred, y) + config.model.L1_weight * L1

        accelerator.backward(loss)
        optimizer.step()

        return loss.item()

    trainer = Engine(train_step)

    def validation_step(engine, batch: List) -> tuple:
        """Validation step for the pytorhc ignite engine
        Args:
            engine: pytorch ignite
            batch (List): current batch
        Returns:
            y_pred: predictions
            y: ground truth labels
        """
        model.eval()
        with torch.no_grad():
            batch = tuple(
                t.to(device) if isinstance(t, torch.Tensor) else t for t in batch
            )
            input_dict, y, chunks = batch
            y_pred = model(
                **input_dict,
                chunks=chunks,
                boost=config.train.boost,
                indipendent_chunks=config.indipendent_chunks,
            )
            if config.indipendent_chunks:
                step_chunks = []
                for i, chunk in enumerate(chunks):
                    step_chunks += [i] * int(chunk)
                step_chunks = torch.Tensor(step_chunks).to(device, dtype=torch.int64)
                y_pred = scatter_mean(y_pred, step_chunks, dim=0)
                y_pred = torch.nn.functional.normalize(y_pred, dim=1, p=1)

            return y_pred, y

    train_evaluator = Engine(validation_step)
    val_evaluator = Engine(validation_step)

    # Attach metrics to the evaluators
    for name, metric in val_metrics.items():
        metric.attach(train_evaluator, name)
        metric.attach(val_evaluator, name)

    if config.ordinal_regression:
        asc_callback = AscensionCallback()

    @trainer.on(Events.ITERATION_COMPLETED(every=1))
    def asciension_callback(engine) -> None:
        """Makes sure that when ordinal regression is used, the cutpoints are in ordered fashion
        Args:
            engine:
        returns:
            None
        """
        if config.ordinal_regression:
            asc_callback.on_batch_end(model)

    @trainer.on(Events.ITERATION_COMPLETED(every=config.train.log_interval))
    def log_training_loss(engine) -> None:
        """Logs the training loss, during the engine training loop, and logs every specified interval.
            The decorator at the top of this function is crucial
        Args:
            engine:
        returns:
            None
        """
        accelerator.print(
            f"Epoch[{engine.state.epoch}], Iter[{engine.state.iteration}] Loss: {engine.state.output:.3f}"
        )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_training_results(trainer) -> None:
        """Logs the training results at the end of every epoch.
            The decorator at the top of this function is crucial
        Args:
            engine:
        returns:
            None
        """
        train_evaluator.run(train_loader)
        metrics = train_evaluator.state.metrics
        if config.ordinal_regression:
            accelerator.print(
                f"Training Results - Epoch[{trainer.state.epoch}] Loss: {metrics['loss']:.4f}"
            )
            accelerator.print(
                f"\tAUROC_30: {metrics['roc_auc_30']:.3f} AUROC_180: {metrics['roc_auc_180']:.3f} AUROC_365: {metrics['roc_auc_365']:.3f}"
            )
            accelerator.print(
                f"\tAUPRC_30: {metrics['prc_auc_30']:.3f} AUPRC_180: {metrics['prc_auc_180']:.3f} AUPRC_365: {metrics['prc_auc_365']:.3f}"
            )
        else:
            accelerator.print(
                f"Training Results - Epoch[{trainer.state.epoch}] Loss: {metrics['loss']:.4f} AUROC: {metrics['roc_auc']:.3f} AUPRC: {metrics['prc_auc']:.3f}"
            )

    @trainer.on(Events.EPOCH_COMPLETED)
    def log_validation_results(trainer) -> None:
        """Logs the validation results at the end of every epoch.
            The decorator at the top of this function is crucial
        Args:
            engine:
        returns:
            None
        """
        val_evaluator.run(val_loader)
        metrics = val_evaluator.state.metrics
        if config.ordinal_regression:
            accelerator.print(
                f"Validation Results - Epoch[{trainer.state.epoch}] Loss: {metrics['loss']:.4f}"
            )
            accelerator.print(
                f"\tAUROC_30: {metrics['roc_auc_30']:.3f} AUROC_180: {metrics['roc_auc_180']:.3f} AUROC_365: {metrics['roc_auc_365']:.3f}"
            )
            accelerator.print(
                f"\tAUPRC_30: {metrics['prc_auc_30']:.3f} AUPRC_180: {metrics['prc_auc_180']:.3f} AUPRC_365: {metrics['prc_auc_365']:.3f}"
            )
        else:
            accelerator.print(
                f"Validation Results - Epoch[{trainer.state.epoch}] Loss: {metrics['loss']:.4f} AUROC: {metrics['roc_auc']:.3f} AUPRC: {metrics['prc_auc']:.3f}\n"
            )

    def score_function(engine) -> torch.Tensor:
        """Scoring function that determines when early stopping is applied
            In our case it's the AUROC
        Args:
            engine
        Returns:
            roc_auc (torch.Tensor): area under the receiving operating curve
        """
        if config.ordinal_regression:
            return -engine.state.metrics["loss"]
        else:
            return engine.state.metrics["roc_auc"]

    # Early stopping handler
    handler = EarlyStopping(
        patience=config.train.early_stop_patience,
        score_function=score_function,
        trainer=trainer,
    )
    val_evaluator.add_event_handler(Events.COMPLETED, handler)

    # Checkpoint handler
    checkpointer = ModelCheckpoint(
        score_function=score_function,
        dirname=config.model.save_path + config.name,
        filename_prefix=config.name,
        n_saved=1,
        create_dir=True,
    )

    if hasattr(model, "module"):
        save_model_dict = {"model": model.module}
    else:
        save_model_dict = {"model": model}

    val_evaluator.add_event_handler(
        Events.EPOCH_COMPLETED, checkpointer, save_model_dict
    )
    trainer.add_event_handler(Events.EPOCH_STARTED, lr_scheduler)

    # Tensorboard Logger
    tb_logger = TensorboardLogger(log_dir=config.tb_log_dir + config.name)
    tb_logger.attach_output_handler(
        trainer,
        event_name=Events.ITERATION_COMPLETED(every=config.train.log_interval),
        tag="training",
        output_transform=lambda loss: {"batch_loss": loss},
    )

    # Attach handler for plotting both evaluators' metrics after every epoch completes
    for tag, evaluator in [
        ("training", train_evaluator),
        ("validation", val_evaluator),
    ]:
        tb_logger.attach_output_handler(
            evaluator,
            event_name=Events.EPOCH_COMPLETED,
            tag=tag,
            metric_names="all",
            global_step_transform=global_step_from_engine(trainer),
        )

    # At last, we can run training
    trainer.run(train_loader, max_epochs=config.train.epochs)


if __name__ == "__main__":
    main(random_state=config.seed)
