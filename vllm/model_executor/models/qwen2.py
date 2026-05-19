# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/qwen2/modeling_qwen2.py
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
"""Inference-only Qwen2 model compatible with HuggingFace weights."""

import math
from collections.abc import Iterable
from itertools import islice
from typing import Any

import torch
from torch import nn
from transformers import Qwen2Config

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import SiluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import (
    MergedColumnParallelLinear,
    QKVParallelLinear,
    RowParallelLinear,
)
from vllm.model_executor.layers.logits_processor import LogitsProcessor
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.model_executor.layers.rotary_embedding import get_rope
from vllm.model_executor.layers.vocab_parallel_embedding import (
    ParallelLMHead,
    VocabParallelEmbedding,
)
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader,
    maybe_remap_kv_scale_name,
)
from vllm.sequence import IntermediateTensors
from vllm.transformers_utils.config import is_interleaved, set_default_rope_theta
from vllm.v1.attention.backend import AttentionType

from .interfaces import (
    EagleModelMixin,
    SupportsEagle,
    SupportsEagle3,
    SupportsLoRA,
    SupportsPP,
)
from .utils import (
    AutoWeightsLoader,
    PPMissingLayer,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Qwen2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_act: str,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.gate_up_proj = MergedColumnParallelLinear(
            hidden_size,
            [intermediate_size] * 2,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.gate_up_proj",
        )
        self.down_proj = RowParallelLinear(
            intermediate_size,
            hidden_size,
            bias=False,
            quant_config=quant_config,
            prefix=f"{prefix}.down_proj",
        )
        if hidden_act != "silu":
            raise ValueError(
                f"Unsupported activation: {hidden_act}. Only silu is supported for now."
            )
        self.act_fn = SiluAndMul()

    def forward(self, x):
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Qwen2Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        rope_parameters: dict[str, Any],
        max_position: int = 4096 * 32,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
        attn_type: str = AttentionType.DECODER,
        dual_chunk_attention_config: dict[str, Any] | None = None,
        qk_norm: bool = False,
        rms_norm_eps: float = 1e-6,
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
        self.head_dim = hidden_size // self.total_num_heads
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = self.head_dim**-0.5
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.qk_norm = qk_norm

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=True,
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

        # QK Normalization support (used in BAGEL and some other models)
        if self.qk_norm:
            self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
            self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

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
            attn_type=attn_type,
            prefix=f"{prefix}.attn",
            **{
                "layer_idx": extract_layer_index(prefix),
                "dual_chunk_attention_config": dual_chunk_attention_config,
            }
            if dual_chunk_attention_config
            else {},
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        # Apply QK normalization if enabled (before RoPE)
        if self.qk_norm:
            # Reshape to apply per-head normalization
            # q shape: (total_tokens, q_size) -> (total_tokens, num_heads, head_dim)
            total_tokens = q.shape[0]
            q = q.view(total_tokens, self.num_heads, self.head_dim)
            k = k.view(total_tokens, self.num_kv_heads, self.head_dim)

            # Apply normalization
            q = self.q_norm(q)
            k = self.k_norm(k)

            # Reshape back
            q = q.view(total_tokens, self.q_size)
            k = k.view(total_tokens, self.kv_size)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Qwen2DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen2Config,
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

        # By default, Qwen2 uses causal attention as it is a decoder-only model.
        # You can override the HF config with `is_causal=False` to enable
        # bidirectional attention, which is used in some embedding models
        # (e.g. Alibaba-NLP/gte-Qwen2-7B-instruct)
        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

        # Check if QK normalization is enabled (used in BAGEL and some other models)
        qk_norm = getattr(config, "qk_norm", False)

        self.self_attn = Qwen2Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            max_position=config.max_position_embeddings,
            num_kv_heads=config.num_key_value_heads,
            cache_config=cache_config,
            quant_config=quant_config,
            rope_parameters=config.rope_parameters,
            prefix=f"{prefix}.self_attn",
            attn_type=attn_type,
            dual_chunk_attention_config=dual_chunk_attention_config,
            qk_norm=qk_norm,
            rms_norm_eps=config.rms_norm_eps,
        )
        self.mlp = Qwen2MLP(
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


def qwen_2_model_invariants(
    input_ids: torch.Tensor,
    positions: torch.Tensor,
    intermediate_tensors: IntermediateTensors | None = None,
    inputs_embeds: torch.Tensor | None = None,
):
    """Shape invariants for Qwen2Model Model, those are translated to
    runtime assertions for unbacked dynamic shapes and are compiled away for
    backed"""
    # All these should be equal.
    # input_ids.size()[0]
    # positions.size()[-1]
    # intermediate_tensors["hidden_states"].size()[0]
    # inputs_embeds.size()[0]
    torch._check(input_ids.size()[0] == positions.size()[-1])
    if intermediate_tensors is not None:
        torch._check(
            input_ids.size()[0] == intermediate_tensors["hidden_states"].size()[0]
        )

    if inputs_embeds is not None:
        torch._check(input_ids.size()[0] == inputs_embeds.size()[0])

    # Hidden dimensions should match (hidden_size)
    # intermediate_tensors["hidden_states"].size()[1]
    # inputs_embeds.size()[1]
    if inputs_embeds is not None and intermediate_tensors is not None:
        torch._check(
            inputs_embeds.size()[1] == intermediate_tensors["hidden_states"].size()[1]
        )


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        # positions is of shape (3, seq_len) if mrope is enabled for qwen2-vl,
        # otherwise (seq_len, ).
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    shape_invariants=qwen_2_model_invariants,
)
class Qwen2Model(nn.Module, EagleModelMixin):
    def __init__(
        self,
        *,
        vllm_config: VllmConfig,
        prefix: str = "",
        decoder_layer_type: type[nn.Module] = Qwen2DecoderLayer,
    ):
        super().__init__()

        config = vllm_config.model_config.hf_config.get_text_config()
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config

        # TODO (@robertgshaw2): see if this can be moved out
        if is_interleaved(vllm_config.model_config.hf_text_config):
            assert config.max_window_layers == config.num_hidden_layers, (
                "Sliding window for some but all layers is not supported. "
                "This model uses sliding window but `max_window_layers` = {} "
                "is less than `num_hidden_layers` = {}. Please open an issue "
                "to discuss this feature.".format(
                    config.max_window_layers,
                    config.num_hidden_layers,
                )
            )

        self.config = config
        self.quant_config = quant_config
        self.vocab_size = config.vocab_size

        if get_pp_group().is_first_rank or (
            config.tie_word_embeddings and get_pp_group().is_last_rank
        ):
            self.embed_tokens = VocabParallelEmbedding(
                config.vocab_size,
                config.hidden_size,
                quant_config=quant_config,
                prefix=f"{prefix}.embed_tokens",
            )
        else:
            self.embed_tokens = PPMissingLayer()

        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: decoder_layer_type(
                config=config,
                cache_config=cache_config,
                quant_config=quant_config,
                prefix=prefix,
            ),
            prefix=f"{prefix}.layers",
        )

        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )
        if get_pp_group().is_last_rank:
            self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        else:
            self.norm = PPMissingLayer()

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        return self.embed_tokens(input_ids)

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> torch.Tensor | IntermediateTensors:
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states = self._maybe_add_hidden_state([], 0, hidden_states, residual)
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )

        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )

        hidden_states, _ = self.norm(hidden_states, residual)

        if len(aux_hidden_states) > 0:
            return hidden_states, aux_hidden_states

        return hidden_states

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]
        params_dict = dict(self.named_parameters(remove_duplicate=False))
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            if "rotary_emb.inv_freq" in name:
                continue
            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache quantization scales
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = (
                    loaded_weight if loaded_weight.dim() == 0 else loaded_weight[0]
                )
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue
            for param_name, weight_name, shard_id in stacked_params_mapping:
                if weight_name not in name:
                    continue
                name = name.replace(weight_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name.endswith("scale"):
                    # Remapping the name of FP8 kv-scale.
                    name = maybe_remap_kv_scale_name(name, params_dict)
                    if name is None:
                        continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                if weight_loader == default_weight_loader:
                    weight_loader(param, loaded_weight)
                else:
                    weight_loader(param, loaded_weight, shard_id)
                break
            else:
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                if name is None:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                if name not in params_dict:
                    continue
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)
        return loaded_params


# =============================================================================
# Qwen2TroughModel — inner model with pre-allocated trough buffer
# =============================================================================


class _Qwen2TroughModelImpl(Qwen2Model):
    """Inner model that also collects normed hidden states for candidate layers
    into a pre-allocated buffer so that entropy-trough selection can run in
    ``compute_logits`` (outside the compiled graph).
    """

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "",
                 decoder_layer_type: type[nn.Module] = None):
        if decoder_layer_type is None:
            decoder_layer_type = Qwen2DecoderLayer
        super().__init__(vllm_config=vllm_config, prefix=prefix, decoder_layer_type=decoder_layer_type)

        additional_config = getattr(vllm_config, "additional_config", {}) or {}
        hf_overrides = getattr(vllm_config.model_config, "hf_overrides", {}) or {}

        def _cfg(key: str, default: object) -> object:
            if key in additional_config:
                return additional_config[key]
            if isinstance(hf_overrides, dict) and key in hf_overrides:
                return hf_overrides[key]
            return default

        num_layers = self.end_layer - self.start_layer
        max_backtrack = int(_cfg("trough_max_backtrack_layers", 0))
        backtrack_ratio = float(_cfg("trough_backtrack_ratio", 0.0))

        if max_backtrack > 0:
            candidate_layers = min(num_layers, max_backtrack)
        elif backtrack_ratio > 0:
            candidate_layers = max(1, int(math.ceil(num_layers * backtrack_ratio)))
        else:
            candidate_layers = num_layers

        self._trough_candidate_layers: int = candidate_layers
        self._trough_start_layer: int = self.start_layer + (
            num_layers - candidate_layers
        )

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: torch.Tensor | None = None,
    ) -> (
        torch.Tensor
        | IntermediateTensors
        | tuple[torch.Tensor, list[torch.Tensor]]
        | tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]
    ):
        if get_pp_group().is_first_rank:
            if inputs_embeds is not None:
                hidden_states = inputs_embeds
            else:
                hidden_states = self.embed_input_ids(input_ids)
            residual = None
        else:
            assert intermediate_tensors is not None
            hidden_states = intermediate_tensors["hidden_states"]
            residual = intermediate_tensors["residual"]

        aux_hidden_states: list[torch.Tensor] = []
        trough_states: list[torch.Tensor] = []
        for idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer)
        ):
            hidden_states, residual = layer(positions, hidden_states, residual)
            self._maybe_add_hidden_state(
                aux_hidden_states, idx + 1, hidden_states, residual
            )
            if (self.start_layer + idx) >= self._trough_start_layer:
                current_h = hidden_states + residual if residual is not None else hidden_states
                trough_states.append(current_h)

        if not get_pp_group().is_last_rank:
            return IntermediateTensors({"hidden_states": hidden_states, "residual": residual})
        hidden_states, _ = self.norm(hidden_states, residual)
        if aux_hidden_states:
            return hidden_states, aux_hidden_states, trough_states
        return hidden_states, trough_states


@support_torch_compile(
    dynamic_arg_dims={
        "input_ids": 0,
        "positions": -1,
        "intermediate_tensors": 0,
        "inputs_embeds": 0,
    },
    shape_invariants=qwen_2_model_invariants,
)
class Qwen2TroughModel(_Qwen2TroughModelImpl, EagleModelMixin):
    pass


class Qwen2ForCausalLM(
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

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config.get_text_config()
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
            self.model = Qwen2TroughModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )
        else:
            self.model = Qwen2Model(
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
                "Qwen2 trough decoding init: enabled=%s, "
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
        is_trough_model = isinstance(self.model, Qwen2TroughModel)

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
        assert isinstance(self.model, Qwen2TroughModel)
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
