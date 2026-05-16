# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Shared utilities for Confident Decoding (entropy-trough selection).

Provides:
- ``read_trough_config``: read trough parameters from vllm_config.
- ``compute_trough_layer_range``: compute candidate layer range for a model.
- ``TroughStateMixin``: base mixin adding trough-related state and methods.
- ``vectorized_entropy_select``: in-place entropy computation + layer selection.
"""

import math
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = __import__("vllm.logger", fromlist=["logger"]).logger


def read_trough_config(vllm_config: "VllmConfig") -> dict:
    """Read trough decoding config from vllm_config.additional_config
    or vllm_config.model_config.hf_overrides."""
    additional_config = getattr(vllm_config, "additional_config", {}) or {}
    hf_overrides = getattr(vllm_config.model_config, "hf_overrides", {}) or {}

    def _cfg(key: str, default):
        if key in additional_config:
            return additional_config[key]
        if isinstance(hf_overrides, dict) and key in hf_overrides:
            return hf_overrides[key]
        return default

    return {
        "enable_trough_decoding": bool(_cfg("enable_multi_layer_entropy_selection", False)),
        "trough_max_backtrack_layers": int(_cfg("trough_max_backtrack_layers", 0)),
        "trough_backtrack_ratio": float(_cfg("trough_backtrack_ratio", 0.0)),
        "trough_select_method": str(_cfg("select_method", "trough")),
        "trough_p": float(_cfg("p", 1.0)),
        "trough_log_interval": int(_cfg("trough_log_interval", 0)),
    }


def compute_trough_layer_range(num_layers: int, config: dict) -> tuple[int, int]:
    """Compute trough start layer and candidate layer count.

    Returns (trough_start_layer, candidate_layers).
    """
    max_backtrack = config["trough_max_backtrack_layers"]
    backtrack_ratio = config["trough_backtrack_ratio"]

    if max_backtrack > 0:
        candidate_layers = min(num_layers, max_backtrack)
    elif backtrack_ratio > 0:
        candidate_layers = max(1, int(math.ceil(num_layers * backtrack_ratio)))
    else:
        candidate_layers = num_layers

    start_layer = num_layers - candidate_layers
    return start_layer, candidate_layers


class TroughStateMixin:
    """Adds trough decoding state and logic to a CausalLM wrapper.

    Subclasses must provide:
    - ``model`` attribute (inner model, may be TroughModel variant).
    - ``lm_head`` attribute.
    - ``logits_processor`` attribute.
    - ``_trough_start_layer`` on the inner model (set by TroughModel).
    - ``compute_logits_override()`` class method that returns a reference
      implementation of logits computation with trough selection.
    """

    def init_trough_state(self, config: dict) -> None:
        self.trough_max_backtrack_layers = config["trough_max_backtrack_layers"]
        self.trough_backtrack_ratio = config["trough_backtrack_ratio"]
        self.trough_select_method = config["trough_select_method"]
        self.trough_p = config["trough_p"]
        self.trough_log_interval = config["trough_log_interval"]
        self._trough_call_count = 0
        self._trough_buffers: dict[int, torch.Tensor] = {}
        self._last_eager_buf: torch.Tensor | None = None
        self._last_seq_len: int = 0
        self._last_logits_indices: torch.Tensor | None = None

    def init_trough_buffer(
        self,
        output: torch.Tensor,
        trough_states: list[torch.Tensor],
    ) -> None:
        """Apply final norm to collected trough states and store in buffer."""
        if not trough_states:
            return
        normed_layers = [self.model.norm(hs, None) for hs in trough_states]
        self._last_eager_buf = torch.stack(normed_layers)
        self._trough_buffers[output.shape[0]] = self._last_eager_buf

    def compute_trough_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        """Compute logits using entropy-trough layer selection.

        Returns ``None`` if trough is disabled or no buffer is available.
        """
        raise NotImplementedError


def vectorized_entropy_select(
    layer_states: torch.Tensor,
    fallback_hidden_states: torch.Tensor,
    logits_processor,
    lm_head,
    select_method: str,
    trough_p: float,
    trough_max_backtrack_layers: int,
    trough_backtrack_ratio: float,
    trough_start_layer: int,
    total_model_layers: int,
    trough_log_interval: int,
    trough_call_count: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
    """Compute entropy trough layer selection.

    Operates **in-place** on ``layer_states`` logits to save memory:
    logits -> probs (in-place softmax), entropy computed in-place,
    logits rebuilt in-place before gather.

    Args:
        layer_states: ``[L, B, H]`` normed hidden states per candidate layer.
        fallback_hidden_states: ``[B, H]`` last-layer hidden states fallback.
        logits_processor: LogitsProcessor instance.
        lm_head: language model head (ParallelLMHead or equivalent).
        select_method: selection strategy (``trough``, ``last-m1``, etc.).
        trough_p: stochastic fallback probability to final layer.
        trough_max_backtrack_layers: explicit max backtrack (0=use ratio).
        trough_backtrack_ratio: ratio-based backtrack window.
        trough_start_layer: global model layer index of first candidate.
        total_model_layers: total number of layers in the model.
        trough_log_interval: log period (0=disabled).
        trough_call_count: current step count for logging.

    Returns:
        Tuple of (selected_logits ``[B, V]``, entropy ``[L, B]``,
        layer_states ``[L, B, V]``, L).
    """
    import torch.nn.functional as F

    L, B, H = layer_states.shape
    device = layer_states.device

    flat = layer_states.reshape(L * B, H)
    flat_logits = logits_processor(lm_head, flat)
    if flat_logits is None:
        flat_logits = logits_processor(lm_head, fallback_hidden_states)
        return flat_logits, torch.zeros(L, B, device=device), torch.zeros(
            L, B, V, device=device
        ) if (V := flat_logits.shape[-1]) else torch.zeros(L, B, 1, device=device), L
    V = flat_logits.shape[-1]
    all_logits = flat_logits.reshape(L, B, V)

    method = select_method

    if method.startswith("last-"):
        try:
            offset = int(method.split("-m")[-1])
        except (IndexError, ValueError):
            offset = 0
        target_model_layer = max(0, total_model_layers - 1 - offset)
        cand_idx = target_model_layer - trough_start_layer
        cand_idx = max(0, min(L - 1, cand_idx))
        selected_layer_idx = torch.full(
            (B,), cand_idx, device=device, dtype=torch.long
        )
        entropy = torch.zeros(L, B, device=device)
    else:
        # In-place softmax on logits to save memory.
        row_max = all_logits.max(dim=-1, keepdim=True).values
        all_logits.sub_(row_max)
        all_logits.exp_()
        Z = all_logits.sum(dim=-1, keepdim=True)
        all_logits.div_(Z)
        entropy = (all_logits * all_logits.log_()).sum(dim=-1)
        entropy.neg_()

        explicit = int(trough_max_backtrack_layers)
        if explicit > 0:
            max_backtrack = explicit
        elif explicit < 0:
            max_backtrack = L
        else:
            max_backtrack = int(L * trough_backtrack_ratio)
        min_layer = L - 1 - max(0, max_backtrack)

        selected_layer_idx = torch.full(
            (B,), L - 1, device=device, dtype=torch.long
        )
        frozen = torch.zeros(B, dtype=torch.bool, device=device)
        prev_entropy = entropy[L - 1]

        for l_idx in range(L - 2, min_layer - 1, -1):
            cur_entropy = entropy[l_idx]
            improves = cur_entropy < prev_entropy
            update_mask = improves & (~frozen)
            selected_layer_idx = torch.where(
                update_mask,
                torch.full_like(selected_layer_idx, l_idx),
                selected_layer_idx,
            )
            frozen = frozen | (~improves)
            prev_entropy = cur_entropy

        # Rebuild raw logits for gather (in-place).
        all_logits.exp_()
        all_logits.mul_(Z)
        all_logits.log_()
        all_logits.add_(row_max)

        if method == "trough-m2":
            selected_layer_idx = torch.clamp(selected_layer_idx - 2, 0, L - 1)
        elif method == "trough-m1":
            selected_layer_idx = torch.clamp(selected_layer_idx - 1, 0, L - 1)
        elif method == "trough-p1":
            selected_layer_idx = torch.clamp(selected_layer_idx + 1, 0, L - 1)
        elif method == "trough-p2":
            selected_layer_idx = torch.clamp(selected_layer_idx + 2, 0, L - 1)

    p = float(trough_p)
    if p < 1.0:
        rng = torch.rand(B, device=device)
        use_final = rng > p
        selected_layer_idx = torch.where(
            use_final,
            torch.full((B,), L - 1, device=device, dtype=torch.long),
            selected_layer_idx,
        )

    gather_idx = selected_layer_idx.unsqueeze(0).unsqueeze(-1).expand(1, B, V)
    selected_logits = all_logits.gather(0, gather_idx).squeeze(0)

    if B > 0 and trough_log_interval > 0 and trough_call_count % trough_log_interval == 0:
        with torch.no_grad():
            sel = selected_layer_idx
            backtrack_depth = (L - 1) - sel
            num_at_final = (sel == (L - 1)).sum().item()
            preview = min(B, 4)
            if method.startswith("trough"):
                final_entropy = entropy[L - 1]
                sample_pairs = [
                    (int(sel[i].item()), float(final_entropy[i].item()))
                    for i in range(preview)
                ]
            else:
                sample_pairs = [(int(sel[i].item()), 0.0) for i in range(preview)]
            logger.info(
                "[trough-decoding] step=%d tokens=%d layers=%d "
                "select_method=%s p=%.2f "
                "avg_selected_layer=%.2f min_selected_layer=%d "
                "avg_backtrack_depth=%.2f max_backtrack_depth=%d "
                "tokens_kept_at_final=%d/%d sample=%s",
                trough_call_count,
                B,
                L,
                method,
                p,
                sel.float().mean().item(),
                int(sel.min().item()),
                backtrack_depth.float().mean().item(),
                int(backtrack_depth.max().item()),
                num_at_final,
                B,
                sample_pairs,
            )

    return selected_logits, entropy, all_logits, L
