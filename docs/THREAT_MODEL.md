# GuardWeave Threat Model

[English](THREAT_MODEL.md) | [简体中文](THREAT_MODEL.zh-CN.md)

This document defines what GuardWeave is meant to defend, where the guarantees become conditional, and what is explicitly out of scope.

## In Scope

- Direct prompt injection in user-supplied text
- Multi-turn probing that escalates over repeated attempts
- Chunked extraction attempts such as stepwise disclosure, continuation prompts, or encoded exfiltration
- Output replay or policy leakage where the protected model starts echoing hidden policy/system-prompt content
- Base-model wrappers where GuardWeave controls both the pre-generation gate and the post-generation verifier

## Partially Covered / Conditional

- Retrieved context or tool output injection
  - GuardWeave can help only if the caller explicitly routes those payloads through the same untrusted-input path or wraps them before generation.
- Local/remote judge-assisted verification
  - Judge-backed checks are conditional on model quality, availability, latency budget, and credentials.
- Regex generation tied to the system prompt
  - Regex-derived profiles strengthen blocking, but they still depend on how informative the bound system prompt is and whether the judge path is enabled.

## Out of Scope

- Full agent sandboxing
- Tool permission isolation
- Data provenance or document authenticity
- Model-weight security
- Jailbreak or safety claims beyond the current benchmark protocol
- Provider-side guarantees for hosted APIs

## Assumptions

- The application binds a concrete system prompt before calling the main model.
- The caller decides which content is trusted and which content must be treated as untrusted input.
- The protected base model is still the final generator; GuardWeave is a defense layer, not a replacement model.

## Non-Goals

- Claiming universal LLM security
- Replacing OS/process sandboxing
- Replacing application-layer authentication or authorization
- Proving provenance or trustworthiness of retrieved documents

## Practical Reading

If your problem is:

- `prompt injection on the request path`: GuardWeave is directly aimed at this.
- `policy leakage in model outputs`: GuardWeave is directly aimed at this.
- `agent/tool privilege separation`: GuardWeave is not the primary control.
- `end-to-end system security for an agent platform`: you will still need sandboxing, permission design, and provenance controls around it.
