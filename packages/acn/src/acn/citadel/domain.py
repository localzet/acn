from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum

from acn.controller.domain import AdaptiveAction, DecisionParameters
from acn.versioning.domain import Metadata


class AuditDecision(StrEnum):
    ALLOWED = "allowed"
    DENIED = "denied"
    OVERRIDE_APPROVED = "override_approved"


@dataclass(frozen=True, slots=True)
class OverrideApproval:
    approved_by: str
    reason: str
    ticket_id: str | None = None


@dataclass(frozen=True, slots=True)
class CitadelActionRequest:
    action: AdaptiveAction
    actor: str
    branch_name: str
    parameters: DecisionParameters = field(default_factory=dict)
    current_commit_id: str | None = None
    override: OverrideApproval | None = None


@dataclass(frozen=True, slots=True)
class CitadelValidationResult:
    allowed: bool
    decision: AuditDecision
    reasons: tuple[str, ...]
    requires_override: bool = False


@dataclass(frozen=True, slots=True)
class AuditLogRecord:
    id: str
    action: str
    actor: str
    branch_name: str
    decision: AuditDecision
    reasons: tuple[str, ...]
    parameters: Metadata = field(default_factory=dict)
    override_by: str | None = None
    override_reason: str | None = None
    override_ticket_id: str | None = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
