import torch
import pytest
from edgevlm.core.loss_functions import MultiTaskVLMLoss, generalized_box_iou


def test_generalized_box_iou_perfect_overlap():
    boxes = torch.tensor([[10.0, 10.0, 50.0, 50.0], [100.0, 100.0, 200.0, 200.0]])
    giou = generalized_box_iou(boxes, boxes)
    assert torch.allclose(giou, torch.tensor([1.0, 1.0]), atol=1e-4)


def test_generalized_box_iou_disjoint_boxes():
    b1 = torch.tensor([[0.0, 0.0, 10.0, 10.0]])
    b2 = torch.tensor([[20.0, 20.0, 30.0, 30.0]])
    giou = generalized_box_iou(b1, b2)
    # Disjoint boxes have IoU=0 and GIoU < 0
    assert giou.item() < 0.0


def test_multi_task_vlm_loss_text_only():
    loss_fn = MultiTaskVLMLoss(lambda_ce=1.0)
    batch, seq_len, vocab_size = 2, 8, 100
    logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    total_loss, loss_dict = loss_fn(logits, labels)
    assert total_loss.item() > 0
    assert "loss_ce" in loss_dict
    assert loss_dict["loss_l1"] == 0.0
    assert loss_dict["loss_giou"] == 0.0

    total_loss.backward()
    assert logits.grad is not None


def test_multi_task_vlm_loss_with_grounding():
    loss_fn = MultiTaskVLMLoss(lambda_ce=1.0, lambda_l1=1.0, lambda_giou=2.0)
    batch, seq_len, vocab_size = 2, 8, 100
    logits = torch.randn(batch, seq_len, vocab_size, requires_grad=True)
    labels = torch.randint(0, vocab_size, (batch, seq_len))

    pred_boxes = torch.tensor([[0.1, 0.1, 0.5, 0.5]], requires_grad=True)
    target_boxes = torch.tensor([[0.15, 0.12, 0.55, 0.52]])

    total_loss, loss_dict = loss_fn(
        logits, labels, pred_boxes=pred_boxes, target_boxes=target_boxes
    )
    assert total_loss.item() > 0
    assert loss_dict["loss_l1"] > 0
    assert loss_dict["loss_giou"] > 0

    total_loss.backward()
    assert pred_boxes.grad is not None
    assert logits.grad is not None
