"""
Multi-Task Loss Functions for EdgeVLM.

Combines Autoregressive Cross-Entropy, Smooth L1 Coordinate Regression,
and Generalized IoU (GIoU) for spatial grounding and visual question answering.
"""

from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F


def generalized_box_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """
    Computes Generalized IoU (GIoU) between two sets of boxes.
    Boxes format: [ymin, xmin, ymax, xmax] normalized to [0, 1] or [0, 1000].
    Shape: [N, 4] and [N, 4]
    Returns: [N] tensor of GIoU values in [-1, 1].
    """
    # Clamp coordinates to ensure valid box dimensions
    y1_1, x1_1, y2_1, x2_1 = boxes1.unbind(-1)
    y1_2, x1_2, y2_2, x2_2 = boxes2.unbind(-1)

    ymin_1, ymax_1 = torch.min(y1_1, y2_1), torch.max(y1_1, y2_1)
    xmin_1, xmax_1 = torch.min(x1_1, x2_1), torch.max(x1_1, x2_1)

    ymin_2, ymax_2 = torch.min(y1_2, y2_2), torch.max(y1_2, y2_2)
    xmin_2, xmax_2 = torch.min(x1_2, x2_2), torch.max(x1_2, x2_2)

    # Intersection area
    inter_ymin = torch.max(ymin_1, ymin_2)
    inter_xmin = torch.max(xmin_1, xmin_2)
    inter_ymax = torch.min(ymax_1, ymax_2)
    inter_xmax = torch.min(xmax_1, xmax_2)

    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_area = inter_h * inter_w

    # Individual box areas
    area1 = (ymax_1 - ymin_1) * (xmax_1 - xmin_1)
    area2 = (ymax_2 - ymin_2) * (xmax_2 - xmin_2)
    union_area = area1 + area2 - inter_area + 1e-7

    iou = inter_area / union_area

    # Smallest enclosing convex box
    enclose_ymin = torch.min(ymin_1, ymin_2)
    enclose_xmin = torch.min(xmin_1, xmin_2)
    enclose_ymax = torch.max(ymax_1, ymax_2)
    enclose_xmax = torch.max(xmax_1, xmax_2)

    enclose_h = (enclose_ymax - enclose_ymin).clamp(min=0)
    enclose_w = (enclose_xmax - enclose_xmin).clamp(min=0)
    enclose_area = enclose_h * enclose_w + 1e-7

    giou = iou - ((enclose_area - union_area) / enclose_area)
    return giou


class MultiTaskVLMLoss(nn.Module):
    """
    Multi-task loss supporting:
    1. Autoregressive language modeling (Cross-Entropy)
    2. Coordinate regression (Smooth L1)
    3. Spatial overlap optimization (Generalized IoU)
    """

    def __init__(
        self,
        lambda_ce: float = 1.0,
        lambda_l1: float = 1.0,
        lambda_giou: float = 2.0,
        ignore_index: int = -100,
    ):
        super().__init__()
        self.lambda_ce = lambda_ce
        self.lambda_l1 = lambda_l1
        self.lambda_giou = lambda_giou
        self.ignore_index = ignore_index

    def forward(
        self,
        lm_logits: torch.Tensor,
        labels: torch.Tensor,
        pred_boxes: Optional[torch.Tensor] = None,
        target_boxes: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            lm_logits: [Batch, Seq_Len, Vocab_Size]
            labels: [Batch, Seq_Len]
            pred_boxes: Optional [N, 4] normalized coordinates
            target_boxes: Optional [N, 4] normalized coordinates

        Returns:
            total_loss: torch.Tensor
            loss_dict: Dict with component breakdown
        """
        # 1. Autoregressive Language Modeling Loss
        shift_logits = lm_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        ce_loss = F.cross_entropy(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1),
            ignore_index=self.ignore_index,
        )

        total_loss = self.lambda_ce * ce_loss
        loss_dict = {"loss_ce": float(ce_loss.detach().item())}

        # 2. Spatial Grounding Loss (if bounding boxes provided)
        if pred_boxes is not None and target_boxes is not None and pred_boxes.numel() > 0:
            pred_boxes_flat = pred_boxes.view(-1, 4)
            target_boxes_flat = target_boxes.view(-1, 4)

            # Smooth L1 Loss
            l1_loss = F.smooth_l1_loss(pred_boxes_flat, target_boxes_flat, beta=1.0)

            # GIoU Loss: 1 - GIoU
            giou = generalized_box_iou(pred_boxes_flat, target_boxes_flat)
            giou_loss = (1.0 - giou).mean()

            total_loss = total_loss + self.lambda_l1 * l1_loss + self.lambda_giou * giou_loss

            loss_dict["loss_l1"] = float(l1_loss.detach().item())
            loss_dict["loss_giou"] = float(giou_loss.detach().item())
            loss_dict["mean_giou"] = float(giou.mean().detach().item())
        else:
            loss_dict["loss_l1"] = 0.0
            loss_dict["loss_giou"] = 0.0

        loss_dict["loss_total"] = float(total_loss.detach().item())
        return total_loss, loss_dict
