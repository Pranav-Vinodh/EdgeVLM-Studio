"""
Spatial Saliency Visual Token Pruner.

Filters out background/non-informative visual patches prior to Language Model processing
based on prompt-conditioned cross-attention saliency weights.
"""

from typing import Tuple, Optional, Dict, Any
import torch
import torch.nn as nn


class SpatialVisualTokenPruner(nn.Module):
    """
    Surgically discards non-informative visual patch tokens based on
    cross-attention saliency scores between text queries and visual features.
    Maintains spatial coherence by keeping selected tokens in raster-scan order.
    """

    def __init__(
        self,
        keep_ratio: float = 0.45,
        min_keep_patches: int = 32,
        pooling_strategy: str = "mean_cross_attn",
    ):
        super().__init__()
        self.keep_ratio = float(keep_ratio)
        self.min_keep_patches = int(min_keep_patches)
        self.pooling_strategy = pooling_strategy

    def compute_saliency_from_cross_attn(
        self, cross_attn_map: torch.Tensor
    ) -> torch.Tensor:
        """
        Computes visual patch saliency from cross-attention weights.
        cross_attn_map: [Batch, Heads, Text_Seq_Len, Visual_Seq_Len]
        Returns: [Batch, Visual_Seq_Len]
        """
        if cross_attn_map.dim() != 4:
            raise ValueError(
                f"Expected 4D cross-attention map [B, H, Nt, Nv], got {cross_attn_map.shape}"
            )

        if self.pooling_strategy == "max_cross_attn":
            # Max over text tokens and heads
            saliency, _ = torch.max(cross_attn_map, dim=2)  # [B, H, Nv]
            saliency, _ = torch.max(saliency, dim=1)        # [B, Nv]
        else:
            # Mean over heads and prompt text tokens:
            # S(v_i) = 1/(H * Nt) * sum_h sum_j A_{h, j, i}
            saliency = torch.mean(cross_attn_map, dim=(1, 2))  # [B, Nv]

        return saliency

    def forward(
        self,
        visual_embeds: torch.Tensor,
        saliency_weights: Optional[torch.Tensor] = None,
        cross_attn_map: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            visual_embeds: [Batch, Num_Patches, Embed_Dim]
            saliency_weights: [Batch, Num_Patches] (optional if cross_attn_map is given)
            cross_attn_map: [Batch, Heads, Text_Len, Num_Patches] (optional)

        Returns:
            pruned_visual_embeds: [Batch, Num_Keep, Embed_Dim]
            kept_indices: [Batch, Num_Keep]
            saliency: [Batch, Num_Patches]
        """
        batch_size, num_patches, embed_dim = visual_embeds.shape

        if saliency_weights is None:
            if cross_attn_map is not None:
                saliency = self.compute_saliency_from_cross_attn(cross_attn_map)
            else:
                # Fallback: compute L2 norm saliency over feature dimension
                saliency = torch.norm(visual_embeds, p=2, dim=-1)
        else:
            saliency = saliency_weights

        # Determine budget K
        num_keep = max(self.min_keep_patches, int(num_patches * self.keep_ratio))
        num_keep = min(num_keep, num_patches)

        # Fast path if keep_ratio >= 1.0
        if num_keep == num_patches:
            all_indices = torch.arange(num_patches, device=visual_embeds.device).expand(
                batch_size, num_patches
            )
            return visual_embeds, all_indices, saliency

        # Top-K patch selection by attention magnitude
        _, topk_indices = torch.topk(saliency, k=num_keep, dim=-1, sorted=True)

        # Sort indices along patch dimension to preserve natural 2D raster-scan order
        topk_indices, _ = torch.sort(topk_indices, dim=-1)

        # Gather retained visual tokens
        expanded_indices = topk_indices.unsqueeze(-1).expand(-1, -1, embed_dim)
        pruned_visual_embeds = torch.gather(visual_embeds, dim=1, index=expanded_indices)

        return pruned_visual_embeds, topk_indices, saliency

    @staticmethod
    def get_compression_stats(
        original_count: int, pruned_count: int
    ) -> Dict[str, float]:
        """Calculates token compression metrics."""
        reduction_count = original_count - pruned_count
        compression_ratio = (
            (100.0 * reduction_count / original_count) if original_count > 0 else 0.0
        )
        speedup_factor = (
            (float(original_count) / pruned_count) if pruned_count > 0 else 1.0
        )
        return {
            "original_tokens": original_count,
            "pruned_tokens": pruned_count,
            "tokens_eliminated": reduction_count,
            "compression_ratio_percent": compression_ratio,
            "theoretical_speedup_factor": speedup_factor,
        }
