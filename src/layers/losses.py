import numpy as np
import torch
from torch import nn
from typing import Optional


def _reduction(loss: torch.Tensor, reduction: str) -> torch.Tensor:
    """
    Reduce loss

    Parameters
    ----------
    loss : torch.Tensor, [batch_size, num_classes]
        Batch losses.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.

    Returns
    -------
    loss : torch.Tensor
        Reduced loss.

    """
    if reduction == "elementwise_mean":
        return loss.mean()
    elif reduction == "none":
        return loss
    elif reduction == "sum":
        return loss.sum()
    else:
        raise ValueError(f"{reduction} is not a valid reduction")


def cumulative_link_loss(
    y_pred: torch.Tensor,
    y_true: torch.Tensor,
    reduction: str = "elementwise_mean",
    class_weights: Optional[np.ndarray] = None,
) -> torch.Tensor:
    """
    Calculates the negative log likelihood using the logistic cumulative link
    function.

    See "On the consistency of ordinal regression methods", Pedregosa et. al.
    for more details. While this paper is not the first to introduce this, it
    is the only one that I could find that was easily readable outside of
    paywalls.

    Parameters
    ----------
    y_pred : torch.Tensor, [batch_size, num_classes]
        Predicted target class probabilities. float dtype.
    y_true : torch.Tensor, [batch_size, 1]
        True target classes. long dtype.
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    Returns
    -------
    loss: torch.Tensor

    """

    eps = 1e-15
    cum_likelihoods = []
    for i in range(y_true.shape[0]):
        if y_true[i] == 0:
            cum_likelihoods.append(y_pred[i, 0])
        else:
            cum_likelihoods.append(y_pred[i, y_true[i] : y_pred.shape[1]].sum())
    cum_likelihoods = torch.stack(cum_likelihoods)
    likelihoods = torch.clamp(cum_likelihoods, eps, 1 - eps)
    neg_log_likelihood = -torch.log(likelihoods)

    if class_weights is not None:
        # Make sure it's on the same device as neg_log_likelihood
        class_weights = torch.as_tensor(
            class_weights,
            dtype=neg_log_likelihood.dtype,
            device=neg_log_likelihood.device,
        )
        neg_log_likelihood *= class_weights[y_true][:, 0]

    loss = _reduction(neg_log_likelihood, reduction)
    return loss


class CumulativeLinkLoss(nn.Module):
    """
    Module form of cumulative_link_loss() loss function

    Parameters
    ----------
    reduction : str
        Method for reducing the loss. Options include 'elementwise_mean',
        'none', and 'sum'.
    class_weights : np.ndarray, [num_classes] optional (default=None)
        An array of weights for each class. If included, then for each sample,
        look up the true class and multiply that sample's loss by the weight in
        this array.

    """

    def __init__(
        self,
        reduction: str = "elementwise_mean",
        class_weights: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.class_weights = class_weights
        self.reduction = reduction

    def forward(self, y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
        return cumulative_link_loss(
            y_pred, y_true, reduction=self.reduction, class_weights=self.class_weights
        )


class WeightedBCE(nn.Module):
    def __init__(
        self,
        class_weights: torch.Tensor,
    ) -> None:
        super().__init__()
        if class_weights is not None:
            assert len(class_weights) == 2
        self.class_weights = class_weights

    def forward(self, output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """weighted binary cross entropy loss
        Args:
            output (torch.Tensor): model outputs
            target (torch.Tensor): targets
        Returns:
            weighted binary cross entropy torch tensor
        """
        loss = self.class_weights[1] * (
            target * torch.log(output)
        ) + self.class_weights[0] * ((1 - target) * torch.log(1 - output))
        return torch.neg(torch.mean(loss))
