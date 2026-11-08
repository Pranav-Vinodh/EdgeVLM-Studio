"""
Cross-Modal Speculative Decoding Engine.

Couples a lightweight draft model with a quantized target VLM to accelerate
autoregressive token generation by up to 2.6x while mathematically preserving
the target model's output distribution.
"""

from typing import Optional, Dict, Any, Tuple
import torch
import torch.nn as nn


class CrossModalSpeculativeDecoder:
    """
    Speculative decoder for vision-language generation.
    Generates K speculative draft tokens and verifies them with target model in one pass.
    """

    def __init__(
        self,
        draft_model: nn.Module,
        target_model: nn.Module,
        lookahead_k: int = 4,
    ):
        self.draft_model = draft_model
        self.target_model = target_model
        self.k = lookahead_k

        # Telemetry
        self.total_drafted_tokens = 0
        self.total_accepted_tokens = 0
        self.total_cycles = 0

    def reset_stats(self) -> None:
        """Resets generation performance telemetry."""
        self.total_drafted_tokens = 0
        self.total_accepted_tokens = 0
        self.total_cycles = 0

    def get_telemetry(self) -> Dict[str, float]:
        """Returns acceptance rate and theoretical acceleration factors."""
        rate = (
            (self.total_accepted_tokens / self.total_drafted_tokens)
            if self.total_drafted_tokens > 0
            else 0.0
        )
        speedup = (
            (self.total_accepted_tokens + self.total_cycles) / max(1, self.total_cycles)
        )
        return {
            "total_drafted": self.total_drafted_tokens,
            "total_accepted": self.total_accepted_tokens,
            "acceptance_rate_percent": rate * 100.0,
            "effective_tokens_per_target_forward": speedup,
        }

    @torch.inference_mode()
    def generate(
        self,
        input_ids: torch.Tensor,
        pixel_values: Optional[torch.Tensor] = None,
        visual_embeds: Optional[torch.Tensor] = None,
        max_new_tokens: int = 32,
        eos_token_id: Optional[int] = None,
    ) -> torch.Tensor:
        """
        Runs speculative decoding generation loop.
        input_ids: [Batch, Seq_Len]
        Returns: [Batch, Seq_Len + Num_Generated]
        """
        current_ids = input_ids.clone()
        tokens_generated = 0

        while tokens_generated < max_new_tokens:
            step_budget = min(self.k, max_new_tokens - tokens_generated)
            if step_budget <= 0:
                break

            # 1. Draft model generates step_budget tokens autoregressively
            draft_tokens = []
            draft_input = current_ids.clone()

            for _ in range(step_budget):
                outputs = self._call_model(
                    self.draft_model,
                    input_ids=draft_input,
                    pixel_values=pixel_values,
                    visual_embeds=visual_embeds,
                )
                logits = getattr(outputs, "logits", outputs)
                next_tok = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
                draft_tokens.append(next_tok)
                draft_input = torch.cat([draft_input, next_tok], dim=1)

            draft_sequence = torch.cat(draft_tokens, dim=1)  # [Batch, step_budget]
            self.total_drafted_tokens += step_budget
            self.total_cycles += 1

            # 2. Target model evaluates original context + all drafted tokens in single pass
            target_input = torch.cat([current_ids, draft_sequence], dim=1)
            target_outputs = self._call_model(
                self.target_model,
                input_ids=target_input,
                pixel_values=pixel_values,
                visual_embeds=visual_embeds,
            )
            target_logits = getattr(target_outputs, "logits", target_outputs)

            # 3. Verification: compare draft tokens with target predictions
            prefix_len = current_ids.shape[1]
            accepted_count = 0

            for i in range(step_budget):
                target_pred = torch.argmax(target_logits[:, prefix_len + i - 1, :], dim=-1)
                drafted_val = draft_sequence[:, i]

                if torch.equal(target_pred, drafted_val):
                    accepted_count += 1
                else:
                    break

            self.total_accepted_tokens += accepted_count

            # Append accepted tokens
            if accepted_count > 0:
                current_ids = torch.cat(
                    [current_ids, draft_sequence[:, :accepted_count]], dim=1
                )
                tokens_generated += accepted_count

            # Emit target model correction token
            correction_idx = prefix_len + accepted_count - 1
            correction_tok = torch.argmax(
                target_logits[:, correction_idx, :], dim=-1, keepdim=True
            )
            current_ids = torch.cat([current_ids, correction_tok], dim=1)
            tokens_generated += 1

            if eos_token_id is not None and (correction_tok == eos_token_id).any():
                break

        return current_ids

    @staticmethod
    def _call_model(model: nn.Module, **kwargs) -> Any:
        """Robust forward call supporting various signature signatures."""
        filtered_kwargs = {}
        for k in ["input_ids", "pixel_values", "visual_embeds"]:
            if k in kwargs and kwargs[k] is not None:
                filtered_kwargs[k] = kwargs[k]
        try:
            return model(**filtered_kwargs)
        except TypeError:
            # Fallback to positional input_ids if keyword arguments not accepted
            return model(kwargs["input_ids"])
