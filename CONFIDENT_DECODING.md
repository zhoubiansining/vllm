# Confident Decoding 技术说明

Confident Decoding 是一种面向自回归推理阶段的 logits 层选择方法。它不总是使用模型最后一层的 logits，而是在靠近最后层的一组候选层中寻找模型最“自信”的输出层。当前实现以 entropy trough selection 为默认策略：从后往前扫描候选层的预测熵，并选择最靠近最终层的第一个熵谷。

## 算法思想

标准 decoding 使用最终层 hidden states 计算 logits。Confident Decoding 的核心假设是：在某些 token 位置，模型的中间高层已经给出比最终层更低熵、更确定的预测分布；继续经过后续层可能会带来犹豫或分布扩散。因此，推理时可以在多个靠近输出端的层上计算 logits，并按置信度选择最终用于采样或贪心解码的 logits。

在默认的 `trough` 策略中，算法流程为：

1. 在模型 forward 中收集候选层的中间状态。
2. 在 wrapper 的 eager 区域对这些状态执行最终 norm，得到每个候选层的 normed hidden states。
3. 在 `compute_logits` 中批量计算所有候选层 logits。
4. 对每个 token、每个候选层计算预测分布 entropy。
5. 从最后候选层向前扫描，找到第一个 entropy 停止下降的位置，即最靠近最终层的 entropy valley。
6. 使用该层 logits 作为当前 token 的输出 logits。

如果设置 `p < 1.0`，每个 token 会以概率 `p` 使用 Confident Decoding 选择层，以概率 `1-p` 回退到最终层 logits，相当于和标准 decoding 做随机混合。

## 技术核心

### 候选层收集

当前实现只收集靠近模型末端的若干层，起始层由 `trough_max_backtrack_layers` 或 `trough_backtrack_ratio` 决定。模型的 compiled inner forward 负责收集中间状态，但不在 compiled 区域内执行额外 norm 或 logits 计算。

这样设计的原因是 vLLM 的 CUDA graph capture 会覆盖外层 wrapper 调用，直接在 compiled forward 中修改 Python 属性或动态 tensor buffer 容易出现 replay 时状态不更新的问题。当前稳定方案是：

- inner trough model 只收集候选层原始中间状态，行为类似 vLLM 现有的 `aux_hidden_states`。
- outer CausalLM wrapper 在 eager 逻辑中读取这些状态，并执行最终 norm。
- `compute_logits` 根据 runner 设置的 `_last_seq_len` 读取对应 batch size 的缓存，避免不同 CUDA graph shape 的 buffer 混用。

### logits 与 entropy 计算

`compute_logits` 将候选层 hidden states reshape 为 `[L * B, H]`，一次性通过 `lm_head` 计算 logits，再恢复为 `[L, B, V]`。其中：

- `L`：候选层数量。
- `B`：本轮需要 logits 的 token 数量。
- `H`：hidden size。
- `V`：vocab size。

Entropy 使用 softmax 后的分布计算：

```text
entropy = -sum(p * log(p))
```

默认 `trough` 选择逻辑从 `L-1` 向前扫描，只有 entropy 继续下降时才继续回溯；一旦 entropy 不再下降，该 token 的选择被冻结。因此选到的是“距离最终层最近的第一个熵谷”，而不是全局最小熵层。

### 支持策略

`select_method` 支持以下取值：

| 方法 | 含义 |
| --- | --- |
| `trough` | 默认策略，选择最靠近最终层的第一个 entropy valley。 |
| `trough-m1` | 在 `trough` 选择层基础上向更浅层偏移 1 层，越界时截断。 |
| `trough-m2` | 在 `trough` 选择层基础上向更浅层偏移 2 层，越界时截断。 |
| `trough-p1` | 在 `trough` 选择层基础上向更深层偏移 1 层，越界时截断。 |
| `trough-p2` | 在 `trough` 选择层基础上向更深层偏移 2 层，越界时截断。 |
| `last-m1` | 不计算 entropy，直接选择原模型倒数第 2 层。 |
| `last-m2` | 不计算 entropy，直接选择原模型倒数第 3 层。 |
| `last-m4` | 不计算 entropy，直接选择原模型倒数第 5 层。 |
| `last-m8` | 不计算 entropy，直接选择原模型倒数第 9 层。 |

`last-mk` 的偏移基准是原始模型的总层数，而不是候选层窗口的局部索引。实现会将原始模型层号转换为候选层 buffer 中的局部索引。

## 用户使用方式

Confident Decoding 通过 `--additional-config` 开启。最小配置如下：

```bash
vllm serve /path/to/model \
  --additional-config '{"enable_multi_layer_entropy_selection": true}'
```

完整示例：

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

### 配置项

| 参数 | 类型 | 默认值 | 说明 |
| --- | --- | --- | --- |
| `enable_multi_layer_entropy_selection` | bool | `false` | 全局开关，开启 Confident Decoding。 |
| `select_method` | str | `"trough"` | 层选择策略。 |
| `p` | float | `1.0` | 使用选择层 logits 的概率；`0.0` 等价于标准最终层 decoding。 |
| `trough_max_backtrack_layers` | int | `0` | 最大回溯层数；`>0` 时优先使用该值，`<0` 表示不限制。 |
| `trough_backtrack_ratio` | float | `0.0` | 当 `trough_max_backtrack_layers == 0` 时，使用 `候选层数 * ratio` 作为回溯窗口。 |
| `trough_log_interval` | int | `0` | 周期性打印选择统计；`0` 表示关闭。 |

### 常用配置

标准 Confident Decoding：

```json
{
  "enable_multi_layer_entropy_selection": true,
  "select_method": "trough",
  "p": 1.0,
  "trough_max_backtrack_layers": 10
}
```

只测试候选机制但保持标准最终层输出：

```json
{
  "enable_multi_layer_entropy_selection": true,
  "select_method": "trough",
  "p": 0.0
}
```

固定选择倒数第 9 层：

```json
{
  "enable_multi_layer_entropy_selection": true,
  "select_method": "last-m8",
  "p": 1.0
}
```

## 支持模型

当前已接入以下模型路径：

- Qwen3.5：`model_executor/models/qwen3_5.py`，入口包括 `Qwen3_5ForCausalLM` 和 `Qwen3_5ForConditionalGeneration`。
- Qwen3.5 MoE：复用 Qwen3.5 CausalLMBase 逻辑。
- GPT-OSS：`model_executor/models/gpt_oss.py`。
- Gemma4：`model_executor/models/gemma4.py`，多模态入口在 `model_executor/models/gemma4_mm.py` 中转发到语言模型 wrapper。
- GLM5.1 / DeepSeek-V2 系：`model_executor/models/deepseek_v2.py`。

## 日志与验证

开启 `trough_log_interval` 后，日志会周期性输出：

- 当前 step 与 token 数。
- 候选层数量。
- `select_method` 与 `p`。
- 平均选择层、最浅选择层。
- 平均/最大回溯深度。
- 保持最终层的 token 数。

示例：

```text
[trough-decoding] step=3000 tokens=8 layers=10 select_method=trough p=1.00 avg_selected_layer=8.25 min_selected_layer=6 avg_backtrack_depth=0.75 max_backtrack_depth=3 tokens_kept_at_final=5/8 sample=[...]
```

验证建议：

- 设置 `p=0.0` 时，输出应与标准 decoding 对齐。
- 设置 `select_method=last-mk` 时，可验证固定层选择是否生效。
- 设置 `trough_log_interval>0` 时，可观察不同 token 是否选择了非最终层。

## 限制与注意事项

- 当前实现只支持 pipeline parallel size 为 1；启用 pipeline parallelism 时会关闭该功能以保证正确性。
- Confident Decoding 会额外计算多个候选层的 `lm_head` 和 entropy，因此相比标准 decoding 有额外开销。
- `trough` 策略依赖模型不同层的置信度变化；某些模型或 checkpoint 可能长期选择最终层，这是合法现象，不一定代表实现错误。
- `p=0.0` 是重要的回归测试配置，应等价于标准最终层 decoding。
