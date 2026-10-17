from dataclasses import dataclass, field
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


@dataclass(repr=False)
class Account:
    provider: str
    alias: str
    key: str = field(repr=False)
    models: list[str] = field(default_factory=list)
    quota_group: str = ""
    account_id: str = field(default="", repr=False)
    free_verified: bool = False


@dataclass
class Deployment:
    account: Account = field(repr=False)
    model: str
    capabilities: set[str] = field(default_factory=lambda: {"chat"})
    context: int = 0
    max_output: int = 0
    listed: bool = False
    access: str = "unknown"
    free: str = "unknown"
    evidence: str = ""
    checked_at: float = 0
    available_after: float = 0
    available_reason: str | None = None
    priority: int = 0
    model_classes: set[str] = field(default_factory=lambda: {"general"})
    max_input: int = 0
    limit_source: str = "unknown"
    tokenizer_file: str | None = None

    @property
    def id(self):
        return f"{self.account.alias}/{self.model}"

    def public(self):
        return dict(id=self.id, provider=self.account.provider, account=self.account.alias,
                    model=self.model, capabilities=sorted(self.capabilities), context=self.context,
                    max_output=self.max_output, max_input=self.max_input, limit_source=self.limit_source,
                    tokenizer_configured=bool(self.tokenizer_file), listed=self.listed, access=self.access,
                    priority=self.priority, model_classes=sorted(self.model_classes),
                    free=self.free, evidence=self.evidence, checked_at=self.checked_at)


def default_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ntg_response",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {"answer": {"type": "string"}},
                "required": ["answer"],
                "additionalProperties": False,
            },
        },
    }


def tool_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ntg_tool_step",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "tool": {"type": ["string", "null"], "enum": [
                        "read_file", "write_file", "list_files", None,
                    ]},
                    "arguments": {
                        "type": "object",
                        "properties": {
                            "path": {"type": ["string", "null"]},
                            "content": {"type": ["string", "null"]},
                        },
                        "required": ["path", "content"],
                        "additionalProperties": False,
                    },
                    "answer": {"type": ["string", "null"]},
                },
                "required": ["tool", "arguments", "answer"],
                "additionalProperties": False,
            },
        },
    }

def review_response_format(entrypoint="calculator.py"):
    from .project import validate_entrypoint
    entrypoint = validate_entrypoint(entrypoint)
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ntg_review", "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "approved": {"type": "boolean"},
                    "issues": {
                        "type": "array", "maxItems": 20,
                        "items": {
                            "type": "object",
                            "properties": {
                                "severity": {"type": "string", "enum": ["high", "medium", "low"]},
                                "file": {"type": "string", "enum": [entrypoint]},
                                "description": {"type": "string", "minLength": 1, "maxLength": 1000},
                            },
                            "required": ["severity", "file", "description"],
                            "additionalProperties": False,
                        },
                    },
                },
                "required": ["approved", "issues"],
                "additionalProperties": False,
            },
        },
    }


def repair_response_format(anchor_ids):
    return {
        "type": "json_schema",
        "json_schema": {"name": "ntg_repair", "strict": True, "schema": {
            "type": "object", "properties": {"edits": {
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": {"type": "object", "properties": {
                    "anchor": {"type": "string", "enum": list(anchor_ids)},
                    "new": {"type": "string"},
                }, "required": ["anchor", "new"], "additionalProperties": False},
            }}, "required": ["edits"], "additionalProperties": False,
        }},
    }


def plan_response_format():
    return {
        "type": "json_schema",
        "json_schema": {
            "name": "ntg_plan", "strict": True,
            "schema": {
                "type": "object",
                "properties": {"tasks": {
                    "type": "array", "minItems": 1, "maxItems": 6,
                    "items": {
                        "type": "object",
                        "properties": {
                            "id": {"type": "string", "pattern": "^T[1-6]$"},
                            "description": {"type": "string", "minLength": 1, "maxLength": 2000},
                            "research_questions": {"type": "array", "maxItems": 3, "uniqueItems": True,
                                "items": {"type": "string", "minLength": 1, "maxLength": 300}},
                            "dependencies": {
                                "type": "array", "maxItems": 5,
                                "items": {"type": "string", "pattern": "^T[1-6]$"},
                            },
                        },
                        "required": ["id", "description", "dependencies", "research_questions"],
                        "additionalProperties": False,
                    },
                }},
                "required": ["tasks"], "additionalProperties": False,
            },
        },
    }


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "auto"
    model_class: Literal["general", "fast", "reasoning", "coding"] = "general"
    preferred_provider: str | None = None
    preferred_model: str | None = None
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0, le=2)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] = Field(default_factory=default_response_format)
    required_capabilities: set[str] = Field(default_factory=set)
    # Caller can increase the conservative local input estimate, never reduce it.
    input_tokens: int | None = Field(default=None, ge=0)
    deadline_seconds: float = Field(default=45, gt=0, le=120)
    stream: bool = False

    @field_validator("response_format", mode="before")
    @classmethod
    def default_structured_output(cls, value):
        return default_response_format() if value is None else value


class RoutingError(Exception):
    def __init__(self, code: str, status: int = 503, history=None):
        self.code, self.status, self.history = code, status, history or []
        super().__init__(code)

    def public(self):
        return {"error": {"message": self.code.replace("_", " "), "type": "routing_error",
                          "code": self.code}, "routing": {"history": self.history}}


def research_response_format():
    return {"type": "json_schema", "json_schema": {
        "name": "ntg_research", "strict": True, "schema": {
            "type": "object", "additionalProperties": False,
            "required": ["findings", "open_questions"],
            "properties": {
                "findings": {"type": "array", "maxItems": 6, "items": {
                    "type": "object", "additionalProperties": False,
                    "required": ["question_index", "summary", "source_ids"],
                    "properties": {
                        "question_index": {"type": "integer", "minimum": 0, "maximum": 2},
                        "summary": {"type": "string", "minLength": 1, "maxLength": 600},
                        "source_ids": {"type": "array", "minItems": 1, "maxItems": 3,
                                       "uniqueItems": True, "items": {"type": "string", "enum": ["S1", "S2", "S3"]}},
                    }}},
                "open_questions": {"type": "array", "maxItems": 3,
                                   "items": {"type": "string", "minLength": 1, "maxLength": 300}},
            }}}}
