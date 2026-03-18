# GuardWeave Judge 评测对比

[English](JUDGE_BENCHMARK_COMPARISON.md) | [简体中文](JUDGE_BENCHMARK_COMPARISON.zh-CN.md)

这份文档把两组同口径的 10 轮实验放在一起比较：

- `本地 judge`：`Qwen/Qwen2.5-3B`
- `远程 judge`：`gemini-2.5-flash`

目的不是比较 base model，而是在其他条件尽量保持一致的前提下，观察 judge 选择对 GuardWeave 防御效果的影响。

## 实验设置

- Base model：`Qwen/Qwen2.5-7B-Instruct`
- 运行轮数：`10`
- Seed：`42-51`
- 每轮候选攻击：`200`
- 每轮至少有效攻击：`80`
- 实际筛出的有效攻击：`90-93`
- 每轮 benign 样本：`50`
- 生成上限：`max_new_tokens = 128`
- 攻击数据源：`wambosec/prompt-injections`
- 违规判定：本地 `attack_strategies.json` + GuardWeave 的 system prompt 违背检查

## 聚合结果

| 指标 | 本地 Judge (`Qwen2.5-3B`) | Gemini Judge (`gemini-2.5-flash`) |
| --- | ---: | ---: |
| 有效攻击均值 | `91.3` | `91.3` |
| 有效攻击率均值 | `45.65%` | `45.65%` |
| 恶意样本防御后违规率均值 | `36.04%` | `7.67%` |
| 恶意样本违规率下降均值 | `63.96%` | `92.33%` |
| 恶意样本 post-block 均值 | `11.93%` | `35.38%` |
| benign 误拒率均值 | `11.60%` | `9.00%` |

## 10 轮波动范围

| 指标 | 本地 Judge 范围 | Gemini Judge 范围 |
| --- | ---: | ---: |
| 有效攻击数 | `90-93` | `90-93` |
| 恶意样本防御后违规率 | `35.16%-38.46%` | `7.53%-7.78%` |
| 恶意样本 post-block 率 | `10.99%-13.19%` | `34.07%-37.36%` |
| benign 误拒率 | `8%-18%` | `6%-12%` |

## 结果解读

- 攻击池是一致的。两组实验使用相同的 base model 和相同的每轮攻击筛选预算，所以有效攻击数基本一致。
- 核心差异来自 judge 层。`gemini-2.5-flash` 在恶意样本上的最终违规率显著低于本地 `Qwen2.5-3B` 分类 judge。
- 生成后拦截能力提升很明显。`post-block` 从大约 `11.9%` 提升到了大约 `35.4%`。
- benign 误拒也略有改善。Gemini 版本把误拒均值从 `11.6%` 降到了 `9.0%`。

## 公开仓库策略

- 公开仓库保留 benchmark 脚本和这份高层总结。
- 另外提供了一份仓库内最小 smoke pack，见 [`docs/BENCHMARK_REPRO.md`](BENCHMARK_REPRO.md)。
- `benchmarks/results/` 下的生成结果不会上传到 GitHub。
- `benchmarks/data/qwen_attack_guardweave/` 和 `benchmarks/data/strategy_guardweave*/` 这类生成出来的 benchmark 数据也不会上传。

## 备注

- 这份对比只比较 judge，不比较被保护的主模型。
- Gemini 版本依赖外部 API，因此会引入网络、费用和凭证管理要求；本地版本没有这些外部依赖。
- 如果你的目标是离线部署或固定推理成本，本地 judge 仍然有意义；如果目标是更强的拦截效果，Gemini 这组结果更有优势。
