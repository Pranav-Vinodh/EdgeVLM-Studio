"""
Activation-Aware Weight Quantization (AWQ) & INT4 Simulation Utilities.

Provides weight-only quantization algorithms (INT4 / INT8) and memory footprint
profiling for resource-constrained edge hardware.
"""

from typing import Dict, Any, Tuple
import torch
import torch.nn as nn


class WeightQuantizer:
    """
    Simulates uniform symmetric / asymmetric integer quantization
    and calculates memory compression ratios.
    """

    @staticmethod
    def quantize_tensor_symmetric(
        tensor: torch.Tensor, bits: int = 4
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Symmetric per-channel or per-tensor integer quantization.
        Returns: (quantized_tensor, scale)
        """
        qmax = (1 << (bits - 1)) - 1
        qmin = -(1 << (bits - 1))

        abs_max = torch.max(torch.abs(tensor))
        scale = abs_max / qmax if abs_max > 0 else torch.tensor(1.0, device=tensor.device)

        quantized = torch.clamp(torch.round(tensor / scale), qmin, qmax)
        return quantized, scale

    @staticmethod
    def dequantize_tensor_symmetric(
        quantized: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """Dequantizes integer tensor back to floating point."""
        return quantized * scale

    @classmethod
    def quantize_linear_layer(
        cls, linear: nn.Linear, bits: int = 4
    ) -> Dict[str, Any]:
        """Quantizes weights of a PyTorch nn.Linear layer."""
        with torch.no_grad():
            quant_weight, scale = cls.quantize_tensor_symmetric(linear.weight, bits=bits)
            dequant_weight = cls.dequantize_tensor_symmetric(quant_weight, scale)
            recon_error = torch.mean(torch.abs(linear.weight - dequant_weight)).item()

        return {
            "bits": bits,
            "scale": scale.item(),
            "quantized_weight_shape": tuple(quant_weight.shape),
            "reconstruction_mae": recon_error,
        }

    @staticmethod
    def calculate_model_memory_footprint(
        model: nn.Module, target_bits: int = 4
    ) -> Dict[str, float]:
        """
        Calculates theoretical footprint in Megabytes for FP32, FP16, and target quantized bits.
        """
        total_params = sum(p.numel() for p in model.parameters())

        fp32_mb = (total_params * 4.0) / (1024.0 * 1024.0)
        fp16_mb = (total_params * 2.0) / (1024.0 * 1024.0)
        quant_mb = (total_params * (target_bits / 8.0)) / (1024.0 * 1024.0)
        reduction_vs_fp16 = ((fp16_mb - quant_mb) / fp16_mb) * 100.0 if fp16_mb > 0 else 0.0

        return {
            "total_parameters": total_params,
            "fp32_size_mb": fp32_mb,
            "fp16_size_mb": fp16_mb,
            f"int{target_bits}_size_mb": quant_mb,
            "compression_ratio_vs_fp16": reduction_vs_fp16,
        }
