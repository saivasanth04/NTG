from dataclasses import dataclass, field
from typing import Any

from pydantic import BaseModel, ConfigDict, Field


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

    @property
    def id(self):
        return f"{self.account.alias}/{self.model}"

    def public(self):
        return dict(id=self.id, provider=self.account.provider, account=self.account.alias,
                    model=self.model, capabilities=sorted(self.capabilities), context=self.context,
                    max_output=self.max_output, listed=self.listed, access=self.access,
                    free=self.free, evidence=self.evidence, checked_at=self.checked_at)


class ChatRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: str = "auto"
    messages: list[dict[str, Any]] = Field(min_length=1)
    max_tokens: int = Field(default=512, ge=1, le=32768)
    temperature: float | None = Field(default=None, ge=0, le=2)
    tools: list[dict[str, Any]] | None = None
    tool_choice: Any = None
    response_format: dict[str, Any] | None = None
    required_capabilities: set[str] = Field(default_factory=set)
    # Caller can increase the conservative local input estimate, never reduce it.
    input_tokens: int | None = Field(default=None, ge=0)
    deadline_seconds: float = Field(default=45, gt=0, le=120)
    stream: bool = False


class RoutingError(Exception):
    def __init__(self, code: str, status: int = 503, history=None):
        self.code, self.status, self.history = code, status, history or []
        super().__init__(code)

    def public(self):
        return {"error": {"message": self.code.replace("_", " "), "type": "routing_error",
                          "code": self.code}, "routing": {"history": self.history}}
