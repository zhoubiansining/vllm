# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Copyright 2024 The Qwen team.
# Copyright 2023 The vLLM team.
# Copyright 2022 EleutherAI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Inference-only Qwen3 model compatible with HuggingFace weights."""

from collections.abc import Iterable
from typing import Any

import torch
from torch import nn
from transformers import Qwen3Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.attention.encoder_only_attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import QKVParallelLinear, RowParallelLinear
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import ParallelLMHead
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsEagle, SupportsEagle3, SupportsLoRA, SupportsPP
from .qwen2 import Qwen2MLP as Qwen3MLP
from .qwen2 import Qwen2Model, Qwen2TroughModel
from .utils import AutoWeightsLoader, PPMissingLayer, extract_layer_index, maybe_prefix

logger = init_logger(__name__)


class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict,
        max_position: int = 4096 * 32,
        head_dim: int | None = None,
        rms_norm_eps: float = 1e-06,
        qkv_bias: bool = False,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        tp_size = get_tensor_model_parallel_world_size()
        self.total_num_heads = num_heads
        assert self.total_num_heads % tp_size == 0
        self.num_heads = self.total_num_heads // tp_size
        self.total_num_kv_heads = num_kv_heads
        if self.total_num_kv_heads >= tp_size:
            # Number of KV heads is greater than TP size, so we partition
            # the KV heads across multiple tensor parallel GPUs.
            assert self.total_num_kv_heads % tp_size == 0
        else:
            # Number of KV heads is less than TP size, so we replicate
            # the KV heads across multiple tensor parallel GPUs.
            assert tp_size % self.total_num_kv_heads == 0
        self.num_kv_heads = max(1, self.total_num_kv_heads // tp_size)
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=qkv_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position,
            rope_parameters=rope_parameters,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        attn_cls = (
            EncoderOnlyAttention
            if attn_type == AttentionType.ENCODER_ONLY
            else Attention
        )
        self.attn = attn_cls(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            prefix=f"{prefix}.attn",
            attn_type=attn_type,
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # Add qk-norm
        q_by_head = q.view(*q.shape[:-1], q.shape[-1] // self.head_dim, self.head_dim)
        q_by_head = self.q_norm(q_by_head)
        q = q_by_head.view(q.shape)
        k_by_head = k.view(*k.shape[:-1], k.shape[-1] // self.head_dim, self.head_dim)
        k_by_head = self.k_norm(k_by_head)
        k = k_by_head.view(k.shape)
        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        set_default_rope_theta(config, default_theta=1000000)
        dual_chunk_attention_config = getattr(
            config, "dual_chunk_attention_config", None
        )

        # By default, Qwen3 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen3-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            rms_norm_eps=config.rms_norm_eps,
            qkv_bias=getattr(config, "attention_bias", False),
            head_dim=getattr(config, "head_dim", None),
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
        )
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        # Self Attention
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
        )

        # Fully Connected
        hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)
        hidden_states = self.mlp(hidden_states)
        return hidden_states, residual


ALL_DECODER_LAYER_TYPES = {
    "attention": Qwen3DecoderLayer,
}


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3Model(Qwen2Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    }
)
class Qwen3TroughModel(Qwen2TroughModel):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(
            vllm_config=vllm_config, prefix=prefix, decoder_layer_type=Qwen3DecoderLayer
        )


class Qwen3ForCausalLM(
    nn.Module, SupportsLoRA, SupportsPP, SupportsEagle, SupportsEagle3
):
    packed_modules_mapping = {
        "qkv_proj": [
            "q_proj",
            "k_proj",
            "v_proj",
        ],
        "gate_up_proj": [
            "gate_proj",
            "up_proj",
        ],
    }

    embedding_modules = {
        "embed_tokens": "input_embeddings",
        "lm_head": "output_embeddings",
    }

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        self.config = config

        self.quant_config = quant_config

        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        hf_overrides = getattr(vllm_config.model_config, "hf_overrides", {}) or {}

        def _cfg_get(key: str, default: object) -> object:
            if key in additional_config:
                return additional_config[key]
            if isinstance(hf_overrides, dict) and key in hf_overrides:
                return hf_overrides[key]
            return default

        self.enable_trough_decoding = bool(
            _cfg_get("enable_multi_layer_entropy_selection", False)
        )
        if self.enable_trough_decoding and get_pp_group().world_size > 1:
            logger.warning(
                "Disabling trough decoding because pipeline parallelism is enabled; "
                "current implementation only supports PP=1 for correctness."
            )
            self.enable_trough_decoding = False

        if self.enable_trough_decoding:
            self.model = Qwen3TroughModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )
        else:
            self.model = Qwen3Model(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )

        if get_pp_group().is_last_rank:
            if config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=maybe_prefix(prefix, "lm_head"),
                )
        else:
            self.lm_head = PPMissingLayer()

        self.logits_processor = LogitsProcessor(config.vocab_size)

        self.make_empty_intermediate_tensors = (
            self.model.make_empty_intermediate_tensors
        )

        self.trough_max_backtrack_layers = int(
            _cfg_get("trough_max_backtrack_layers", 0)
        )
        self.trough_backtrack_ratio = float(_cfg_get("trough_backtrack_ratio", 0.0))
        self.trough_select_method = str(_cfg_get("select_method", "trough"))
        self.trough_p = float(_cfg_get("p", 1.0))
        self.trough_log_interval = int(_cfg_get("trough_log_interval", 0))
        self._trough_call_count = 0
        self._trough_buffers: dict = {}
        self._last_seq_len = 0

        compilation_config = getattr(vllm_config, "compilation_config", None)
        cg_sizes = (
            getattr(compilation_config, "cudagraph_capture_sizes", None)
            if compilation_config is not None
            else None
        )
        self._trough_captured_shapes: frozenset[int] = (
            frozenset(cg_sizes) if cg_sizes else frozenset()
        )

        if self.enable_trough_decoding:
            logger.info(
                "Qwen3 trough decoding init: enabled=%s, "
                "select_method=%s, p=%.2f, "
                "max_backtrack_layers=%d, backtrack_ratio=%.3f, trough_log_interval=%d",
                self.enable_trough_decoding,
                self.trough_select_method,
                self.trough_p,
                self.trough_max_backtrack_layers,
                self.trough_backtrack_ratio,
                self.trough_log_interval,
            )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.model.embed_input_ids(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        is_trough_model = isinstance(self.model, Qwen3TroughModel)

        output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds
        )

        if not (self.enable_trough_decoding and is_trough_model and get_pp_group().is_last_rank):
            return output

        if isinstance(output, tuple) and len(output) == 3:
            hidden_states, aux_hidden_states, trough_states = output
        else:
            hidden_states, trough_states = output
            aux_hidden_states = None
        if not trough_states:
            if aux_hidden_states:
                return hidden_states, aux_hidden_states
            return hidden_states

        normed_layers = [self.model.norm(hs, None) for hs in trough_states]
        normed_buf = torch.stack(normed_layers) if normed_layers else None
        self._trough_buffers[hidden_states.shape[0]] = normed_buf
        if aux_hidden_states:
            return hidden_states, aux_hidden_states
        return hidden_states

    def compute_logits(
        self,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor | None:
        if not self.enable_trough_decoding:
            return self.logits_processor(self.lm_head, hidden_states)

        from .trough_utils import vectorized_entropy_select

        self._trough_call_count += 1
        B = hidden_states.shape[0]
        assert isinstance(self.model, Qwen3TroughModel)
        layer_states = self._trough_buffers.get(self._last_seq_len)
        if layer_states is None:
            return self.logits_processor(self.lm_head, hidden_states)
        L_buf, S_buf, H_buf = layer_states.shape

        logits_indices = getattr(self, "_last_logits_indices", None)
        if logits_indices is not None:
            layer_states = layer_states[:, logits_indices]
        elif B != S_buf:
            layer_states = layer_states[:, -B:]

        selected_logits, _, _, _ = vectorized_entropy_select(
            layer_states=layer_states,
            fallback_hidden_states=hidden_states,
            logits_processor=self.logits_processor,
            lm_head=self.lm_head,
            select_method=self.trough_select_method,
            trough_p=self.trough_p,
            trough_max_backtrack_layers=self.trough_max_backtrack_layers,
            trough_backtrack_ratio=self.trough_backtrack_ratio,
            trough_start_layer=self.model._trough_start_layer,
            total_model_layers=len(self.model.layers),
            trough_log_interval=self.trough_log_interval,
            trough_call_count=self._trough_call_count,
        )
        if (
            self._last_seq_len not in self._trough_captured_shapes
            and self._last_seq_len in self._trough_buffers
        ):
            self._trough_buffers.pop(self._last_seq_len, None)
        self._last_logits_indices = None
        self._last_seq_len = 0
        return selected_logits

    def clear_trough_buffers(self) -> None:
        captured = self._trough_captured_shapes
        for key in list(self._trough_buffers.keys()):
            if key not in captured:
                self._trough_buffers.pop(key, None)

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        loader = AutoWeightsLoader(
            self,
            skip_prefixes=(["lm_head."] if self.config.tie_word_embeddings else None),
        )
        return loader.load_weights(weights)
