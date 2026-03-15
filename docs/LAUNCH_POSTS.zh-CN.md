# 发布文案

下面这些文案可以直接拿去发 GitHub、X、Reddit、Hacker News 或中文社区。

## 短版

GuardWeave 现在公开了，一个轻量的 prompt injection 防御层，可放在商用 API 和本地 LLM 前后。

- 纯启发式模式只依赖 Python 标准库
- 本地 `Qwen2.5-3B` judge 对 `Qwen2.5-7B-Instruct` 的恶意样本违规率平均压低了 `63.96%`
- `gemini-2.5-flash` judge 版本平均压低了 `92.33%`

仓库：

https://github.com/Ha0c4/GuardWeave

## 技术版

我把 GuardWeave 开源了，这是一个风险自适应的防御层，可以直接套在商用 API、OpenAI-compatible 本地壳，以及本地 Hugging Face 模型前后。

我想解决的问题不是“再做一个只在 benchmark 里看起来不错的原型”，而是做一个更容易接入真实部署链路的 gating layer。

当前仓库已经包含：

- 生成前风险评分
- 生成后输出校验
- 只有在 system prompt 改动时才刷新的 judge 派生 regex
- 本地分类 judge 的训练与评测接口
- 同一 `7B` base model 上本地 `3B` judge 和远程 `gemini-2.5-flash` judge 的对比实验

目前最值得展示的一组结果：

- 本地 `Qwen2.5-3B` judge 保护 `Qwen2.5-7B-Instruct`：恶意样本违规率平均下降 `63.96%`
- `gemini-2.5-flash` judge：恶意样本违规率平均下降 `92.33%`

仓库：

https://github.com/Ha0c4/GuardWeave
