# GuardWeave Threat Model / 边界说明

[English](THREAT_MODEL.md) | [简体中文](THREAT_MODEL.zh-CN.md)

这份文档定义了 GuardWeave 主要防什么、哪些能力是条件成立的、以及哪些事情本来就不在项目目标里。

## In Scope

- 用户输入路径上的直接 prompt injection
- 多轮持续试探导致的风险升级
- 分块抽取、续写式抽取、编码后抽取等 stepwise disclosure / exfiltration
- 输出回放或 policy 泄漏，即模型开始回显隐藏 policy / system prompt 内容
- 由 GuardWeave 同时控制生成前 gating 和生成后校验的 base-model 包装场景

## Partially Covered / Conditional

- Retrieved context 或 tool output 注入
  - 只有在调用方明确把这些内容当作不可信输入送进同一条防御路径时，GuardWeave 才能覆盖到。
- 本地 / 远程 judge 辅助验证
  - judge 路径是否有效，取决于 judge 质量、可用性、延迟预算和凭证配置。
- 与 system prompt 绑定的 regex 生成
  - 这会增强拦截，但效果仍然依赖于绑定的 system prompt 是否足够具体，以及 judge 路径是否开启。

## Out of Scope

- 完整 agent sandbox
- 工具权限隔离
- 数据溯源或文档真实性证明
- 模型权重级安全
- 超出当前 benchmark 协议的广义 jailbreak / safety claim
- 托管 API 提供商侧的底层安全保证

## 假设前提

- 应用侧会在调用主模型前绑定一个明确的 system prompt。
- 调用方自己决定哪些内容是 trusted，哪些内容必须作为 untrusted input 处理。
- 最终负责生成回答的仍然是 base model；GuardWeave 是防御层，不是替代模型。

## 非目标

- 不声称提供“通用 LLM 安全”
- 不替代 OS / 进程级 sandbox
- 不替代应用层认证与授权
- 不证明检索文档的来源可信度

## 实用理解

如果你的问题是：

- `请求路径上的 prompt injection`：GuardWeave 直接面向这个问题。
- `模型输出里的 policy 泄漏`：GuardWeave 直接面向这个问题。
- `agent/tool 权限隔离`：GuardWeave 不是主要控制点。
- `agent 平台的端到端系统安全`：你仍然需要在外层补 sandbox、权限设计和 provenance 控制。
