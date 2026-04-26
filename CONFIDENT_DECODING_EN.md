# Confident Decoding Technical Summary

Confident Decoding is a logits-layer selection method for autoregressive inference. Instead of always using logits from the final model layer, it searches among a set of near-final candidate layers and selects the output layer where the model appears most confident. The default implementation uses entropy trough selection: it scans prediction entropy from the back to the front and selects the first entropy valley closest to the final layer.

## Algorithm Idea

Standard decoding computes logits from the final-layer hidden states. Confident Decoding is based on the observation that, for some token positions, a high intermediate layer may already produce a lower-entropy and more confident prediction distribution than the final layer. Additional layers may sometimes make the distribution less decisive. Therefore, inference can compute logits at several near-final layers and select the logits from the layer with higher confidence.

The default `trough` strategy works as follows:

1. Collect intermediate states from candidate layers during model forward.
2. Apply the final norm to these states in the eager wrapper, producing normed hidden states for each candidate layer.
3. Compute logits for all candidate layers in `compute_logits` using a batched `lm_head` call.
4. Compute prediction entropy for every token and every candidate layer.
5. Scan from the final candidate layer backward and find the first point where entropy stops decreasing, i.e. the entropy valley closest to the final layer.
6. Use logits from that selected layer as the output logits for the current token.

If `p < 1.0`, each token uses the selected Confident Decoding layer with probability `p`, and falls back to the final-layer logits with probability `1-p`. This provides stochastic mixing with standard decoding.

## Technical Core

### Candidate Layer Collection

The implementation only collects several layers near the end of the model. The collection window is controlled by `trough_max_backtrack_layers` or `trough_backtrack_ratio`. The compiled inner model forward collects intermediate states, but does not run extra norm or logits computation inside the compiled region.

This design is necessary because vLLM's CUDA graph capture covers the outer wrapper call. Mutating Python attributes or dynamic tensor buffers directly inside compiled forward can cause stale state during CUDA graph replay. The stable design is:

- The inner trough model only collects raw candidate-layer states, similar to vLLM's existing `aux_hidden_states` mechanism.
- The outer CausalLM wrapper reads these states in eager logic and applies the final norm.
- `compute_logits` uses `_last_seq_len`, set by the model runner, to retrieve the buffer corresponding to the current CUDA graph batch shape and avoid mixing buffers from different shapes.

### Logits and Entropy Computation

`compute_logits` reshapes candidate hidden states into `[L * B, H]`, computes logits with one batched `lm_head` call, then reshapes them back to `[L, B, V]`.

- `L`: number of candidate layers.
- `B`: number of tokens that need logits in this step.
- `H`: hidden size.
- `V`: vocabulary size.

Entropy is computed from the softmax distribution:

```text
entropy = -sum(p * log(p))
```

The default `trough` selection scans backward from `L-1`. It only continues backtracking while entropy keeps decreasing. Once entropy stops decreasing, the selected layer for that token is frozen. Therefore, it selects the first entropy valley closest to the final layer, not the global minimum-entropy layer.

### Supported Strategies

`select_method` supports the following values:

| Method | Meaning |
| --- | --- |
| `trough` | Default strategy. Selects the first entropy valley closest to the final layer. |
| `trough-m1` | Starts from the `trough` selection and shifts 1 layer toward shallower layers, clamped at boundaries. |
| `trough-m2` | Starts from the `trough` selection and shifts 2 layers toward shallower layers, clamped at boundaries. |
| `trough-p1` | Starts from the `trough` selection and shifts 1 layer toward deeper layers, clamped at boundaries. |
| `trough-p2` | Starts from the `trough` selection and shifts 2 layers toward deeper layers, clamped at boundaries. |
| `last-m1` | Does not compute entropy. Selects the second-to-last original model layer. |
| `last-m2` | Does not compute entropy. Selects the third-to-last original model layer. |
| `last-m4` | Does not compute entropy. Selects the fifth-to-last original model layer. |
| `last-m8` | Does not compute entropy. Selects the ninth-to-last original model layer. |

For `last-mk`, the offset is based on the original model's total layer count, not the local candidate-window index. The implementation converts the original model layer index into the local candidate-buffer index.

## User Guide

Confident Decoding is enabled through `--additional-config`. Minimal example:

```bash
vllm serve /path/to/model \
  --additional-config '{"enable_multi_layer_entropy_selection": true}'
```

Full example:

```bash
vllm serve /workspace/ckpt/Qwen3.5-9B \
  --port 8000 \
  --tensor-parallel-size 4 \
  --max-model-len 262144 \
  --reasoning-parser qwen3 \
  --language-model-only \
  --additional-config '{
    "enable_multi_layer_entropy_selection": true,
    "select_method": "trough",
    "p": 1.0,
    "trough_max_backtrack_layers": 10,
    "trough_log_interval": 200
  }'
```

### Configuration

| Parameter | Type | Default | Description |
| --- | --- | --- | --- |
| `enable_multi_layer_entropy_selection` | bool | `false` | Global switch for Confident Decoding. |
| `select_method` | str | `"trough"` | Layer selection strategy. |
| `p` | float | `1.0` | Probability of using selected-layer logits. `0.0` is equivalent to standard final-layer decoding. |
| `trough_max_backtrack_layers` | int | `0` | Maximum number of layers to backtrack. `>0` uses this value directly; `<0` means unlimited. |
| `trough_backtrack_ratio` | float | `0.0` | Used when `trough_max_backtrack_layers == 0`; backtrack window is `num_candidate_layers * ratio`. |
| `trough_log_interval` | int | `0` | Periodically logs selection statistics. `0` disables logging. |

### Common Configurations

Standard Confident Decoding:

```json
{
  "enable_multi_layer_entropy_selection": true,
  "select_method": "trough",
  "p": 1.0,
  "trough_max_backtrack_layers": 10
}
```

Test the candidate path while keeping standard final-layer output:

```json
{
  "enable_multi_layer_entropy_selection": true,
  "select_method": "trough",
  "p": 0.0
}
```

Always select the ninth-to-last layer:

```json
{
  "enable_multi_layer_entropy_selection": true,
  "select_method": "last-m8",
  "p": 1.0
}
```

## Supported Models

The current implementation supports the following model paths:

- Qwen3.5: `model_executor/models/qwen3_5.py`, including `Qwen3_5ForCausalLM` and `Qwen3_5ForConditionalGeneration`.
- Qwen3.5 MoE: reuses the Qwen3.5 CausalLMBase implementation.
- GPT-OSS: `model_executor/models/gpt_oss.py`.
- Gemma4: `model_executor/models/gemma4.py`, with the multimodal entry point in `model_executor/models/gemma4_mm.py` forwarding to the language-model wrapper.
- GLM5.1 / DeepSeek-V2 family: `model_executor/models/deepseek_v2.py`.

## Logging and Validation

When `trough_log_interval` is enabled, logs periodically report:

- Current step and number of tokens.
- Number of candidate layers.
- `select_method` and `p`.
- Average selected layer and shallowest selected layer.
- Average and maximum backtrack depth.
- Number of tokens kept at the final layer.

Example:

```text
[trough-decoding] step=3000 tokens=8 layers=10 select_method=trough p=1.00 avg_selected_layer=8.25 min_selected_layer=6 avg_backtrack_depth=0.75 max_backtrack_depth=3 tokens_kept_at_final=5/8 sample=[...]
```

Suggested validation:

- With `p=0.0`, outputs should match standard final-layer decoding.
- With `select_method=last-mk`, fixed layer selection can be verified directly.
- With `trough_log_interval>0`, logs should show whether tokens are selecting non-final layers.

## Limitations and Notes

- The current implementation only supports pipeline parallel size 1; the feature is disabled when pipeline parallelism is enabled to preserve correctness.
- Confident Decoding computes additional `lm_head` and entropy operations for multiple candidate layers, so it adds overhead compared with standard decoding.
- The `trough` strategy depends on confidence dynamics across layers. Some models or checkpoints may frequently select the final layer, which is valid and does not necessarily indicate an implementation bug.
- `p=0.0` is an important regression-test setting and should be equivalent to standard final-layer decoding.
