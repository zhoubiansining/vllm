# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright 2025 The vLLM team.
# Copyright 2025 Google Inc. HuggingFace Inc. team. All rights reserved.
#
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
from collections.abc import Iterable
from itertools import islice

import math
import torch
from torch import nn
from transformers import Gemma3TextConfig

from vllm.compilation.decorators import support_torch_compile
from vllm.config import CacheConfig, VllmConfig
from vllm.distributed import get_pp_group, get_tensor_model_parallel_world_size
from vllm.logger import init_logger
from vllm.model_executor.layers.activation import GeluAndMul
from vllm.model_executor.layers.attention import (
    Attention,
    EncoderOnlyAttention,
)
from vllm.model_executor.layers.layernorm import GemmaRMSNorm
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
from vllm.v1.attention.backend import AttentionType

from .interfaces import SupportsLoRA, SupportsPP
from .utils import (
    AutoWeightsLoader,
    extract_layer_index,
    is_pp_missing_parameter,
    make_empty_intermediate_tensors_factory,
    make_layers,
    maybe_prefix,
)

logger = init_logger(__name__)


class Gemma3MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        hidden_activation: str,
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
        if hidden_activation != "gelu_pytorch_tanh":
            raise ValueError(
                "Gemma3 uses `gelu_pytorch_tanh` as the hidden activation "
                "function. Please set `hidden_act` and `hidden_activation` to "
                "`gelu_pytorch_tanh`."
            )
        self.act_fn = GeluAndMul(approximate="tanh")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gate_up, _ = self.gate_up_proj(x)
        x = self.act_fn(gate_up)
        x, _ = self.down_proj(x)
        return x


class Gemma3Attention(nn.Module):
    def __init__(
        self,
        config: Gemma3TextConfig,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        max_position_embeddings: int,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        attn_logits_soft_cap: float | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.config = config
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
        self.head_dim = head_dim
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        self.scaling = config.query_pre_attn_scalar**-0.5

        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.qkv_proj",
        )
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=config.attention_bias,
            quant_config=quant_config,
            prefix=f"{prefix}.o_proj",
        )

        self.q_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = GemmaRMSNorm(self.head_dim, eps=config.rms_norm_eps)

        layer_idx = extract_layer_index(prefix)
        layer_type = config.layer_types[layer_idx]
        self.is_sliding = layer_type == "sliding_attention"
        sliding_window = config.sliding_window if self.is_sliding else None

        # Initialize the rotary embedding.
        if layer_type in config.rope_parameters:
            # Transformers v5 rope config.
            rope_parameters = config.rope_parameters[layer_type]
        else:
            # Transformers v4 rope config.
            # Global attention. Use the values in config.json.
            rope_parameters = config.rope_parameters
            # Local attention. Override the values in config.json.
            if self.is_sliding:
                rope_parameters = dict(
                    rope_type="default", rope_theta=config.rope_local_base_freq
                )

        self.rotary_emb = get_rope(
            self.head_dim,
            max_position=max_position_embeddings,
            rope_parameters=rope_parameters,
            is_neox_style=True,
        )

        if getattr(config, "is_causal", True):
            attn_type = AttentionType.DECODER
        else:
            attn_type = AttentionType.ENCODER_ONLY

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
            logits_soft_cap=attn_logits_soft_cap,
            per_layer_sliding_window=sliding_window,
            prefix=f"{prefix}.attn",
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        **kwargs,
    ) -> torch.Tensor:
        qkv, _ = self.qkv_proj(hidden_states)
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)

        q = q.unflatten(-1, (self.num_heads, self.head_dim))
        q = self.q_norm(q)
        q = q.flatten(-2, -1)
        k = k.unflatten(-1, (self.num_kv_heads, self.head_dim))
        k = self.k_norm(k)
        k = k.flatten(-2, -1)

        q, k = self.rotary_emb(positions, q, k)
        attn_output = self.attn(q, k, v)
        output, _ = self.o_proj(attn_output)
        return output


class Gemma3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Gemma3TextConfig,
        cache_config: CacheConfig | None = None,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        self.hidden_size = config.hidden_size
        self.self_attn = Gemma3Attention(
            config=config,
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            head_dim=config.head_dim,
            max_position_embeddings=config.max_position_embeddings,
            cache_config=cache_config,
            quant_config=quant_config,
            attn_logits_soft_cap=None,
            prefix=f"{prefix}.self_attn",
        )
        self.hidden_size = config.hidden_size
        self.mlp = Gemma3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_activation=config.hidden_activation,
            quant_config=quant_config,
            prefix=f"{prefix}.mlp",
        )
        self.input_layernorm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.pre_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )
        self.post_feedforward_layernorm = GemmaRMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if residual is None:
            residual = hidden_states
            hidden_states = self.input_layernorm(hidden_states)
        else:
            hidden_states, residual = self.input_layernorm(hidden_states, residual)
        hidden_states = self.self_attn(
            positions=positions,
            hidden_states=hidden_states,
            **kwargs,
        )
        hidden_states = self.post_attention_layernorm(hidden_states)

        hidden_states, residual = self.pre_feedforward_layernorm(
            hidden_states, residual
        )
        hidden_states = self.mlp(hidden_states)
        hidden_states = self.post_feedforward_layernorm(hidden_states)
        return hidden_states, residual


@support_torch_compile
class Gemma3Model(nn.Module):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__()
        config = vllm_config.model_config.hf_config
        cache_config = vllm_config.cache_config
        quant_config = vllm_config.quant_config
        self.config = config
        self.quant_config = quant_config

        self.embed_tokens = VocabParallelEmbedding(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=f"{prefix}.embed_tokens",
        )
        self.start_layer, self.end_layer, self.layers = make_layers(
            config.num_hidden_layers,
            lambda prefix: Gemma3DecoderLayer(
                config, cache_config, quant_config, prefix=prefix
            ),
            prefix=f"{prefix}.layers",
        )
        self.norm = GemmaRMSNorm(config.hidden_size, eps=config.rms_norm_eps)

        # Normalize the embedding by sqrt(hidden_size)
        # The normalizer's data type should be downcasted to the model's
        # data type such as bfloat16, not float32.
        # See https://github.com/huggingface/transformers/pull/29402
        normalizer = self.config.hidden_size**0.5
        self.register_buffer("normalizer", torch.tensor(normalizer), persistent=False)
        self.make_empty_intermediate_tensors = make_empty_intermediate_tensors_factory(
            ["hidden_states", "residual"], config.hidden_size
        )

    def embed_input_ids(self, input_ids: torch.Tensor) -> torch.Tensor:
        # NOTE(woosuk): Only apply the normalizer to the output of
        # vocab embedding. Don't apply it to the vision embedding.
        return self.embed_tokens(input_ids) * self.normalizer

    def forward(
        self,
        input_ids: torch.Tensor | None,
        positions: torch.Tensor,
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
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
        for layer in islice(self.layers, self.start_layer, self.end_layer):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **kwargs,
            )
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
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
        params_dict = dict(self.named_parameters())
        loaded_params: set[str] = set()
        for name, loaded_weight in weights:
            # Revert +1 during llama.cpp conversion
            # see: https://github.com/ggml-org/llama.cpp/blob/be7c3034108473beda214fd1d7c98fd6a7a3bdf5/convert_hf_to_gguf.py#L3397-L3400
            if (
                self.quant_config
                and self.quant_config.get_name() == "gguf"
                and name.endswith("norm.weight")
            ):
                loaded_weight -= 1

            if self.quant_config is not None and (
                scale_name := self.quant_config.get_cache_scale(name)
            ):
                # Loading kv cache scales for compressed-tensors quantization
                param = params_dict[scale_name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                loaded_weight = loaded_weight[0]
                weight_loader(param, loaded_weight)
                loaded_params.add(scale_name)
                continue

            # Check if this is a scale parameter that needs remapping first
            if name.endswith((".k_scale", ".v_scale", ".q_scale", ".prob_scale")):
                # Try to remap the scale name first
                remapped_name = maybe_remap_kv_scale_name(name, params_dict)
                if remapped_name is not None and remapped_name in params_dict:
                    # Successfully remapped, use the remapped name
                    param = params_dict[remapped_name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                    loaded_params.add(remapped_name)
                    continue
                # If remapping failed, continue with normal processing

            for param_name, shard_name, shard_id in stacked_params_mapping:
                if shard_name not in name:
                    continue
                name = name.replace(shard_name, param_name)
                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue
                if is_pp_missing_parameter(name, self):
                    continue
                param = params_dict[name]
                weight_loader = param.weight_loader
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
                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader", default_weight_loader)
                weight_loader(param, loaded_weight)
            loaded_params.add(name)

        return loaded_params


# =============================================================================
# Gemma3TroughModel — inner model with pre-allocated trough buffer
# =============================================================================


class _Gemma3TroughModelImpl(Gemma3Model):
    def __init__(self, *, vllm_config: VllmConfig, prefix: str = ""):
        super().__init__(vllm_config=vllm_config, prefix=prefix)

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
        intermediate_tensors: IntermediateTensors | None,
        inputs_embeds: torch.Tensor | None = None,
        **kwargs,
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
        trough_states: list[torch.Tensor] = []
        for layer_idx, layer in enumerate(
            islice(self.layers, self.start_layer, self.end_layer),
            start=self.start_layer,
        ):
            hidden_states, residual = layer(
                positions,
                hidden_states,
                residual,
                **kwargs,
            )
            if layer_idx >= self._trough_start_layer:
                current_h = hidden_states + residual if residual is not None else hidden_states
                trough_states.append(current_h)
        if not get_pp_group().is_last_rank:
            return IntermediateTensors(
                {"hidden_states": hidden_states, "residual": residual}
            )
        hidden_states, _ = self.norm(hidden_states, residual)
        return hidden_states, trough_states


@support_torch_compile
class Gemma3TroughModel(_Gemma3TroughModelImpl):
    pass


class Gemma3ForCausalLM(nn.Module, SupportsLoRA, SupportsPP):
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
        config = vllm_config.model_config.hf_config
        quant_config = vllm_config.quant_config

        super().__init__()
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
            self.model = Gemma3TroughModel(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )
        else:
            self.model = Gemma3Model(
                vllm_config=vllm_config, prefix=maybe_prefix(prefix, "model")
            )

        self.lm_head = ParallelLMHead(
            config.vocab_size,
            config.hidden_size,
            quant_config=quant_config,
            prefix=maybe_prefix(prefix, "lm_head"),
        )
        if config.tie_word_embeddings:
            self.lm_head = self.lm_head.tie_weights(self.model.embed_tokens)

        self.logits_processor = LogitsProcessor(
            config.vocab_size, soft_cap=config.final_logit_softcapping
        )
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
                "Gemma3 trough decoding init: enabled=%s, "
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
        **kwargs,
    ) -> torch.Tensor | IntermediateTensors:
        is_trough_model = isinstance(self.model, Gemma3TroughModel)

        output = self.model(
            input_ids, positions, intermediate_tensors, inputs_embeds, **kwargs
        )

        if not (self.enable_trough_decoding and is_trough_model and get_pp_group().is_last_rank):
            return output

        if isinstance(output, tuple) and len(output) == 2:
            hidden_states, trough_states = output
        else:
            return output
        if not trough_states:
            return hidden_states

        normed_layers = [self.model.norm(hs, None) for hs in trough_states]
        normed_buf = torch.stack(normed_layers) if normed_layers else None
        self._trough_buffers[hidden_states.shape[0]] = normed_buf
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
        assert isinstance(self.model, Gemma3TroughModel)
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
