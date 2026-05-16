from typing import Protocol
from uuid import uuid4

from acn.citadel.domain import AuditDecision, AuditLogRecord, OverrideApproval
from acn.versioning.domain import Metadata


class AuditLogRepository(Protocol):
    def record(
        self,
        *,
        action: str,
        actor: str,
        branch_name: str,
        decision: AuditDecision,
        reasons: tuple[str, ...],
        parameters: Metadata | None = None,
        override: OverrideApproval | None = None,
    ) -> AuditLogRecord: ...

    def list_records(self) -> tuple[AuditLogRecord, ...]: ...


class InMemoryAuditLogRepository:
    def __init__(self) -> None:
        self._records: list[AuditLogRecord] = []

    def record(
        self,
        *,
        action: str,
        actor: str,
        branch_name: str,
        decision: AuditDecision,
        reasons: tuple[str, ...],
        parameters: Metadata | None = None,
        override: OverrideApproval | None = None,
    ) -> AuditLogRecord:
        record = AuditLogRecord(
            id=f"audit_{uuid4().hex}",
            action=action,
            actor=actor,
            branch_name=branch_name,
            decision=decision,
            reasons=reasons,
            parameters=parameters or {},
            override_by=override.approved_by if override is not None else None,
            override_reason=override.reason if override is not None else None,
            override_ticket_id=override.ticket_id if override is not None else None,
        )
        self._records.append(record)
        return record

    def list_records(self) -> tuple[AuditLogRecord, ...]:
        return tuple(self._records)
