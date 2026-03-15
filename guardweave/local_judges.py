"""Local classifier judges and artifact helpers."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .core import Policy

try:
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
except Exception as exc:  # pragma: no cover - handled at runtime
    torch = None  # type: ignore[assignment]
    AutoModelForSequenceClassification = None  # type: ignore[assignment]
    AutoTokenizer = None  # type: ignore[assignment]
    _TRANSFORMERS_IMPORT_ERROR = exc
else:
    _TRANSFORMERS_IMPORT_ERROR = None

try:
    from peft import PeftConfig, PeftModel
except Exception:
    PeftConfig = None  # type: ignore[assignment]
    PeftModel = None  # type: ignore[assignment]


JUDGE_ARTIFACT_CONFIG = "guardweave_judge_config.json"
TASK_RISK = "risk"
TASK_OUTPUT = "output"
DEFAULT_JUDGE_VERSION = 1

DEFAULT_RISK_TEMPLATE = (
    "TASK: risk_judge\n"
    "POLICY:\n{policy_text}\n\n"
    "CONTEXT:\n{context}\n\n"
    "USER_INPUT:\n{user_text}\n"
)

DEFAULT_OUTPUT_TEMPLATE = (
    "TASK: output_judge\n"
    "POLICY:\n{policy_text}\n\n"
    "USER_INPUT:\n{user_text}\n\n"
    "ASSISTANT_OUTPUT:\n{output_text}\n"
)


def _require_transformers() -> None:
    if _TRANSFORMERS_IMPORT_ERROR is not None:
        raise RuntimeError(
            "Local classifier judges require transformers and torch. "
            "Install GuardWeave with training extras."
        ) from _TRANSFORMERS_IMPORT_ERROR


def render_policy_text(policy: Policy) -> str:
    return "\n".join(f"- {item}" for item in policy.prohibited) + (f"\nNotes: {policy.notes}" if policy.notes else "")


def _default_template(task: str) -> str:
    if task == TASK_RISK:
        return DEFAULT_RISK_TEMPLATE
    if task == TASK_OUTPUT:
        return DEFAULT_OUTPUT_TEMPLATE
    raise ValueError(f"Unsupported judge task: {task}")


def render_judge_text(
    task: str,
    *,
    user_text: str,
    policy_text: str,
    context: str = "",
    output_text: str = "",
    template: Optional[str] = None,
) -> str:
    """Render one text example for local judge training or inference."""
    active_template = template or _default_template(task)
    return active_template.format(
        user_text=user_text or "",
        policy_text=policy_text or "",
        context=context or "",
        output_text=output_text or "",
    )


def _infer_device(device: str) -> str:
    if device != "auto":
        return device
    _require_transformers()
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(device: str, dtype: str = "auto") -> Any:
    _require_transformers()
    value = str(dtype or "auto").strip().lower()
    if value == "float16":
        return torch.float16
    if value == "bfloat16":
        return torch.bfloat16
    if value == "float32":
        return torch.float32
    if device in {"cuda", "mps"}:
        return torch.float16
    return torch.float32


@dataclass
class JudgeArtifactConfig:
    """Metadata stored alongside a trained local classifier judge."""

    task: str
    label_names: List[str]
    positive_label: str
    threshold: float = 0.5
    max_length: int = 256
    text_template: str = ""
    base_model_name_or_path: str = ""
    finetune_method: str = "full"
    version: int = DEFAULT_JUDGE_VERSION
    extra: Dict[str, Any] = field(default_factory=dict)

    def save(self, output_dir: str | Path) -> str:
        target = Path(output_dir) / JUDGE_ARTIFACT_CONFIG
        target.write_text(json.dumps(asdict(self), indent=2, ensure_ascii=False), encoding="utf-8")
        return str(target)

    @classmethod
    def load(cls, output_dir: str | Path) -> "JudgeArtifactConfig":
        target = Path(output_dir) / JUDGE_ARTIFACT_CONFIG
        data = json.loads(target.read_text(encoding="utf-8"))
        return cls(**data)


class LocalSequenceClassificationJudgeBase:
    """Base class for locally loaded Hugging Face sequence-classification judges."""

    def __init__(self, model_dir: str | Path, *, device: str = "auto", dtype: str = "auto") -> None:
        _require_transformers()
        self.model_dir = str(model_dir)
        self.artifact = JudgeArtifactConfig.load(self.model_dir)
        self.device = _infer_device(device)
        preferred_dtype = dtype if dtype != "auto" else str(self.artifact.extra.get("load_dtype", "auto"))
        self.torch_dtype = _resolve_torch_dtype(self.device, preferred_dtype)
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_dir, use_fast=True)
        self.model = self._load_model()
        if getattr(self.tokenizer, "pad_token", None) is None and getattr(self.tokenizer, "eos_token", None) is not None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

    def _load_model(self) -> Any:
        label2id = {label: idx for idx, label in enumerate(self.artifact.label_names)}
        id2label = {idx: label for label, idx in label2id.items()}
        model_path = Path(self.model_dir)
        if (model_path / "adapter_config.json").exists():
            if PeftConfig is None or PeftModel is None:
                raise RuntimeError("Loading a LoRA-trained judge requires peft to be installed.")
            peft_cfg = PeftConfig.from_pretrained(self.model_dir)
            base_model = AutoModelForSequenceClassification.from_pretrained(
                peft_cfg.base_model_name_or_path,
                num_labels=len(self.artifact.label_names),
                label2id=label2id,
                id2label=id2label,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
            model = PeftModel.from_pretrained(base_model, self.model_dir)
        else:
            model = AutoModelForSequenceClassification.from_pretrained(
                self.model_dir,
                torch_dtype=self.torch_dtype,
                low_cpu_mem_usage=True,
            )
        model.to(self.device)
        model.eval()
        return model

    @property
    def positive_index(self) -> int:
        try:
            return self.artifact.label_names.index(self.artifact.positive_label)
        except ValueError as exc:
            raise RuntimeError(
                f"Positive label {self.artifact.positive_label!r} missing from {self.artifact.label_names!r}."
            ) from exc

    def score_text(self, text: str) -> float:
        """Return the positive-class probability for one rendered text input."""
        encoded = self.tokenizer(
            text,
            truncation=True,
            max_length=int(self.artifact.max_length),
            padding=True,
            return_tensors="pt",
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        with torch.no_grad():
            logits = self.model(**encoded).logits
        if logits.shape[-1] == 1:
            score = torch.sigmoid(logits[0][0]).item()
        else:
            score = torch.softmax(logits, dim=-1)[0][self.positive_index].item()
        return float(score)


class LocalSequenceRiskJudge(LocalSequenceClassificationJudgeBase):
    """Risk judge backed by a local sequence-classification checkpoint."""

    def risk(self, user_text: str, context: str, policy: Policy) -> float:
        text = render_judge_text(
            TASK_RISK,
            user_text=user_text,
            context=context,
            policy_text=render_policy_text(policy),
            template=self.artifact.text_template or None,
        )
        return self.score_text(text)


class LocalSequenceOutputJudge(LocalSequenceClassificationJudgeBase):
    """Output judge backed by a local sequence-classification checkpoint."""

    def violates(self, user_text: str, output_text: str, policy: Policy) -> tuple[bool, str]:
        text = render_judge_text(
            TASK_OUTPUT,
            user_text=user_text,
            output_text=output_text,
            policy_text=render_policy_text(policy),
            template=self.artifact.text_template or None,
        )
        score = self.score_text(text)
        violates = score >= float(self.artifact.threshold)
        return violates, f"local_sequence_classifier:p={score:.3f}"
