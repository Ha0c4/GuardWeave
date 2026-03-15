# GuardWeave 性能与轻量化说明

[English](PERFORMANCE.md) | [简体中文](PERFORMANCE.zh-CN.md)

这份文档把 README 里的 `lightweight` 说法拆成部署者真正关心的成本项。

## 测量口径

- Base model：`Qwen/Qwen2.5-7B-Instruct`
- 生成上限：`max_new_tokens = 128`
- 公开攻击协议：`wambosec/prompt-injections` + 本地 `attack_strategies.json`
- 下面的 `无防御`、`纯启发式`、`本地 judge` 三列，使用的是同一轮 full-scan 代表性结果：`90` 条有效攻击 + `50` 条 benign。
- `远程 judge` 使用的是已经公开存档的 `gemini-2.5-flash` 10 轮 suite 汇总，base model 和攻击筛选协议保持一致。
- 当前公开 benchmark 记录的是整轮请求时延，而不是 first-token latency。所以这里公开的运行时指标统一使用 benign 流量上的中位整轮延迟开销。

## 部署总览

| 方案 | 恶意样本违规率 | benign 误拒率 | 中位延迟开销 | benign 中位额外 prompt token | benign 中位额外模型/API 调用 |
| --- | ---: | ---: | ---: | ---: | ---: |
| 无防御 | `100.00%` | `0.00%` | `0.00s` | `0` | `0` |
| 纯启发式 | `54.44%` | `2.00%` | `3.58s` | `+148` | `0` |
| 本地 judge（`Qwen/Qwen2.5-3B`） | `37.78%` | `10.00%` | `1.45s` | `+148` | `+1` |
| 远程 judge（`gemini-2.5-flash`） | `7.67%` | `9.00%` | `3.26s` | `+148` | `+1` |

## 1. 计算成本

- `无防御`：只有一次 `7B` base-model 生成。
- `纯启发式`：仍然只有一次 `7B` base-model 生成，外加本地规则式 pre/post 检查。
- `本地 judge`：一次 `7B` base-model 生成，再叠加一个 `3B` 分类 judge 层做风险/输出判定。
- `远程 judge`：一次 `7B` base-model 生成，再叠加远程 risk/output judge 调用。

本地 judge 这条路径的模型体量仍然显著小于被保护的主模型：

- Judge 规模：`3B`
- Base model 规模：`7B`
- 比例：约 `42.9%`

## 2. 延迟

这里的运行时指标是 benign 流量相对于无防御基线的中位整轮延迟开销。

| 方案 | 中位延迟开销 |
| --- | ---: |
| 无防御 | `0.00s` |
| 纯启发式 | `3.58s` |
| 本地 judge | `1.45s` |
| 远程 judge | `3.26s` |

说明：

- `纯启发式` 不一定就是“最快的防御模式”。在这轮代表性实验里，它放行了更多请求进入完整 defended prompt 路径，而不是更早 pre-refuse。
- `本地 judge` 虽然增加了分类判断，但在这轮代表性结果里，中位延迟仍低于纯启发式。
- `远程 judge` 则是在同样 defended generation 的基础上，再额外叠加网络/API 往返。

## 3. Token / API 开销

在当前 benchmark 里，所有防御模式的 benign 流量 base-model prompt 中位额外 token 都是 `+148`。这些 token 主要来自：

- 注入的 defense policy block
- 分层 threat instruction
- 在高风险时把用户输入包装成不可信内容

带 judge 的模式，在 benign 流量上的中位额外 judge 调用数都是 `+1`。

可以这样理解：

- `纯启发式`：改 prompt 结构，但不增加额外模型/API 调用。
- `本地 judge`：增加本地分类调用，但不引入外部 API 账单。
- `远程 judge`：把这部分额外调用变成外部 API 流量，同时引入凭证、网络和费用要求。

## 4. 依赖足迹

包的安装方式是分层设计的，默认安装保持极简。

| 安装方式 | 声明的必需依赖数 | 能力范围 |
| --- | ---: | --- |
| `pip install guardweave` | `0` | 核心库、CLI、纯启发式路径、远程 REST judge 集成 |
| `pip install guardweave[train]` | `7` 个可选 ML 依赖 | 本地 judge 训练与本地分类 judge 运行 |

当前可选训练栈包括：

- `accelerate`
- `datasets`
- `numpy`
- `peft`
- `pyyaml`
- `torch`
- `transformers`

## 结论

- 最轻的可用部署是 `纯启发式`：没有额外模型/API 调用，没有额外运行时必需依赖，而且恶意样本违规率确实下降。
- 当前公开 benchmark 里，本地 `3B` judge 是比较好的本地折中：拦截效果明显强于纯启发式，而 benign 流量的中位额外调用仍然只有 `+1`。
- 最强的公开拦截结果仍然是远程 Gemini judge，但它的代价是外部 API 延迟、凭证和费用。
