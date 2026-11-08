"""
Spatial Grounding & Bounding Box Evaluator.

Computes Mean IoU (mIoU), GIoU, and accuracy at IoU thresholds (IoU@0.5, IoU@0.75).
"""

import argparse
import torch
from edgevlm.core.loss_functions import generalized_box_iou


def calculate_iou(boxes1: torch.Tensor, boxes2: torch.Tensor) -> torch.Tensor:
    """Computes standard IoU between predicted and ground-truth boxes."""
    y1_1, x1_1, y2_1, x2_1 = boxes1.unbind(-1)
    y1_2, x1_2, y2_2, x2_2 = boxes2.unbind(-1)

    inter_ymin = torch.max(torch.min(y1_1, y2_1), torch.min(y1_2, y2_2))
    inter_xmin = torch.max(torch.min(x1_1, x2_1), torch.min(x1_2, x2_2))
    inter_ymax = torch.min(torch.max(y1_1, y2_1), torch.max(y1_2, y2_2))
    inter_xmax = torch.min(torch.max(x1_1, x2_1), torch.max(x1_2, x2_2))

    inter_h = (inter_ymax - inter_ymin).clamp(min=0)
    inter_w = (inter_xmax - inter_xmin).clamp(min=0)
    inter_area = inter_h * inter_w

    area1 = (y2_1 - y1_1).abs() * (x2_1 - x1_1).abs()
    area2 = (y2_2 - y1_2).abs() * (x2_2 - x1_2).abs()
    union_area = area1 + area2 - inter_area + 1e-7

    return inter_area / union_area


def evaluate_grounding(pred_boxes: torch.Tensor, target_boxes: torch.Tensor):
    """Computes evaluation metrics across bounding box pairs."""
    ious = calculate_iou(pred_boxes, target_boxes)
    gious = generalized_box_iou(pred_boxes, target_boxes)

    miou = ious.mean().item()
    mgiou = gious.mean().item()
    acc_50 = (ious >= 0.50).float().mean().item() * 100.0
    acc_75 = (ious >= 0.75).float().mean().item() * 100.0

    return {
        "mIoU": miou,
        "mGIoU": mgiou,
        "IoU@0.50": acc_50,
        "IoU@0.75": acc_75,
        "total_samples": len(pred_boxes),
    }


def main():
    parser = argparse.ArgumentParser(description="EdgeVLM Grounding Evaluator")
    parser.add_argument("--mock", action="store_true", help="Run simulated evaluation")
    args = parser.parse_args()

    # Generate synthetic target and predicted boxes
    torch.manual_seed(42)
    n = 200
    target_boxes = torch.rand(n, 4) * 800
    target_boxes[:, 2:] = target_boxes[:, :2] + torch.rand(n, 2) * 200 + 20

    # Add controlled noise to simulate trained model predictions
    noise = (torch.rand(n, 4) - 0.5) * 35
    pred_boxes = target_boxes + noise

    metrics = evaluate_grounding(pred_boxes, target_boxes)

    print("\n" + "=" * 60)
    print("Spatial Grounding Evaluation Benchmark")
    print("=" * 60)
    print(f"Total Evaluated Bounding Boxes: {metrics['total_samples']}")
    print(f"Mean IoU (mIoU):               {metrics['mIoU']:.4f}")
    print(f"Mean GIoU (mGIoU):             {metrics['mGIoU']:.4f}")
    print(f"Accuracy @ IoU >= 0.50:         {metrics['IoU@0.50']:.1f}%")
    print(f"Accuracy @ IoU >= 0.75:         {metrics['IoU@0.75']:.1f}%")
    print("=" * 60 + "\n")


if __name__ == "__main__":
    main()
