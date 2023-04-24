from typing import Union
import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable


class FocalLoss(nn.Module):
    # https://leimao.github.io/blog/Focal-Loss-Explained/

    def __init__(
        self,
        gamma: int = 0,
        alpha: Union[int, float, list, None] = None,
        size_average=True,
        reduction: str = "none",
    ):
        """Initialize the focal loss

        Args:
            gamma (int, optional): the exponential component. Defaults to 0.
            alpha (Union[int, float, list, None], optional): Weight component.
                If float/int is given, num_class = 2 is implied,
                If list is given, num_class = len(alpha).
                If None is given, no weight will be applied.
                Defaults to None.
            size_average (bool, optional): If True, apply mean, if false, apply sum.
                Defaults to True.
        """
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha

        if isinstance(alpha, (float, int)):
            self.alpha = Tensor([alpha, 1 - alpha])

        if isinstance(alpha, list):
            self.alpha = Tensor(alpha)

        # self.size_average = size_average
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor):
        """Forward call

        Args:
            input (Tensor): 4D Tensor of
                (batch_size, num_class, H, W). Tensor
                will be apply softmax
            target (Tensor): 3D Tensor of
                (batch_size, H, W), dtype=torch.in64, value
                must be in range of [0, num_class)
                => num_class > 1

        Returns:
            Tensor: the loss, mean or sum by batch
        """
        batch_size = input.shape[0]
        if input.dim() > 2:
            input = input.view(input.size(0), input.size(1), -1)  # N,C,H,W => N,C,H*W
            input = input.transpose(1, 2)  # N,C,H*W => N,H*W,C
            input = input.contiguous().view(-1, input.size(2))  # N,H*W,C => N*H*W,C

        target = target.view(-1, 1)

        # The Cross Entropy component
        log_pt = F.log_softmax(input, dim=-1)
        log_pt = log_pt.gather(1, target)
        log_pt = log_pt.view(-1)
        pt = Variable(log_pt.data.exp())

        # The weight component
        if self.alpha is not None:
            if self.alpha.type() != input.data.type():
                self.alpha = self.alpha.type_as(input.data)

            at = self.alpha.gather(0, target.data.view(-1))
            log_pt = log_pt * Variable(at)

        # The focal component
        loss = -1 * (1 - pt) ** self.gamma * log_pt

        if self.reduction == "none":
            return loss.contiguous().view(batch_size, -1).mean(dim=1)

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()


class BinaryFocalLoss(FocalLoss):
    def __init__(
        self,
        gamma: int = 0,
        alpha: Union[int, float, None] = None,
        reduction: str = "none",
    ):
        if alpha:
            assert isinstance(
                alpha, (float, int)
            ), "Binary Focal Loss only need atomic weight"
        super().__init__(gamma, alpha, reduction)

    def forward(self, input: Tensor, target: Tensor):
        # input.shape == B, H, W
        # target.shape == B, H, W, binary value
        negative = 1.0 - input
        combine = torch.stack((negative, input), dim=1)
        return super().forward(combine, target)


class DiceLoss(nn.Module):
    def __init__(self, activation: nn.Sigmoid = None, reduction: str = "none"):
        super().__init__()
        self.activation = activation
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, smooth=1) -> Tensor:
        """
        Args:
            input (Tensor): 3D Tensor of (B, H, W)
            target (Tensor): 3D Tensor of (B, H, W)
            smooth (int, optional): smooth param. Defaults to 1.

        Returns:
            Tensor: loss, reduced or not
        """

        if self.activation:
            input = self.activation(input)

        # flatten label and prediction tensors
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)

        intersection = (input * target).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            input.sum(dim=1) + target.sum(dim=1) + smooth
        )
        dice = 1 - dice

        if self.reduction == "none":
            return dice

        if self.reduction == "sum":
            return dice.sum()

        if self.reduction == "mean":
            return dice.mean()


class IoULoss(nn.Module):
    def __init__(self, activation: nn.Sigmoid = None, reduction: str = "none"):
        super().__init__()
        self.activation = activation
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor, smooth=1) -> Tensor:
        """
        Args:
            input (Tensor): 3D Tensor of (B, H, W)
            target (Tensor): 3D Tensor of (B, H, W)
            smooth (int, optional): smooth param. Defaults to 1.

        Returns:
            Tensor: loss, reduced or not
        """

        if self.activation:
            input = self.activation(input)

        # flatten label and prediction tensors
        input = input.view(input.shape[0], -1)
        target = target.view(target.shape[0], -1)

        intersection = (input * target).sum(dim=1)
        dice = (2.0 * intersection + smooth) / (
            input.sum(dim=1) + target.sum(dim=1) - intersection + smooth
        )
        dice = 1 - dice

        if self.reduction == "none":
            return dice

        if self.reduction == "sum":
            return dice.sum()

        if self.reduction == "mean":
            return dice.mean()


class MeanSquareError(nn.Module):
    def __init__(self, reduction: str = "none") -> None:
        super().__init__()
        self.reduction = reduction

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        input = input.view(input.shape[0], -1)
        target = target.view(input.shape[0], -1)
        loss = torch.square(input - target).mean(dim=1)

        if self.reduction == "none":
            return loss

        if self.reduction == "sum":
            return loss.sum()

        if self.reduction == "mean":
            return loss.mean()


class SamLoss(nn.Module):
    def __init__(
        self,
        focal_gamma: int = 2.0,
        focal_alpha: Union[int, float, None] = None,
        dice_activation: nn.Module = nn.Sigmoid(),
    ) -> None:
        """Initialize the per mask loss for SAM

        Args:
            focal_gamma (int, optional): Check Focal Loss.
                Defaults to 2.0.
            focal_alpha (Union[int, float, None], optional):
                Check Focal Loss. Defaults to None.
            dice_activation (nn.Module, optional): Check Dice.
                Defaults to nn.Sigmoid().
        """
        super().__init__()
        self.focal = BinaryFocalLoss(
            gamma=focal_gamma, alpha=focal_alpha, reduction="none"
        )
        self.dice = DiceLoss(activation=dice_activation, reduction="none")
        self.iou = IoULoss(activation=dice_activation, reduction="none")
        self.mse = MeanSquareError(reduction="none")

    def forward(self, mask_pred: Tensor, mask_target: Tensor, iou_pred: Tensor):
        focal_loss = self.focal(mask_pred, mask_target)
        dice_loss = self.dice(mask_pred, mask_target)
        iou_target = self.iou(mask_pred, mask_target)
        iou_mse_loss = self.mse(iou_pred, iou_target)

        final_loss = focal_loss * 20 + dice_loss + iou_mse_loss
        return final_loss


class MultimaskSamLoss(nn.Module):
    def __init__(
        self,
        n_masks=3,
        reduction: str = "none",
        focal_gamma: int = 2.0,
        focal_alpha: Union[int, float, None] = None,
        dice_activation: nn.Module = nn.Sigmoid(),
    ) -> None:
        super().__init__()
        self.n_masks = n_masks
        self.reduction = reduction
        self.sam_loss = SamLoss(
            focal_alpha=focal_alpha,
            focal_gamma=focal_gamma,
            dice_activation=dice_activation,
        )
        pass

    # NOTE: Need unit test due to the a very hard problem.
    # As mention from the paper, for each batch, we compute the loss
    # On both 3 (or N) masks, and select the best mask (lowest grad)
    # to run backward from. Below is a optimized solution for GPU, by
    # batching all and run forward 1 time, and then reshape and select
    # min grad via min(dim=-1) and run backward from.
    def forward(
        self, multi_mask_pred: Tensor, multi_mask_target: Tensor, multi_iou_pred: Tensor
    ) -> torch.Tensor:
        batch_size, _, h, w = multi_mask_pred.shape

        multi_mask_pred = multi_mask_pred.contiguous().view(
            batch_size * self.n_masks, h, w
        )
        multi_mask_target = multi_mask_target.contiguous().view(
            batch_size * self.n_masks, h, w
        )

        multi_iou_pred = multi_iou_pred.contiguous().view(batch_size * self.n_masks, -1)

        losses: Tensor = self.sam_loss(
            multi_mask_pred, multi_mask_target, multi_iou_pred
        )

        # Note that torch.max have gradient flow through the minimum elements
        losses = losses.contiguous().view(batch_size, self.n_masks)

        # FIXME: check this carefully
        losses = losses.min(dim=-1).values
        if self.reduction == "none":
            return losses

        if self.reduction == "sum":
            return losses.sum()

        if self.reduction == "mean":
            return losses.mean()


if __name__ == "__main__":
    a = Tensor([[1, 2], [3, 4], [5, 6]])

    print(a)
    print(a.contiguous().view(-1))
    print(a.contiguous().view(3, 2))
    pass
