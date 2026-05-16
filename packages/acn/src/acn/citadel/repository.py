from uuid import uuid4

from sqlalchemy import select
from sqlalchemy.orm import Session

from acn.citadel.audit import AuditLogRepository
from acn.citadel.domain import AuditDecision, AuditLogRecord, OverrideApproval
from acn.citadel.models import CitadelAuditLogModel
from acn.versioning.domain import Metadata


class SqlAlchemyAuditLogRepository(AuditLogRepository):
    def __init__(self, session: Session) -> None:
        self._session = session

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
        model = CitadelAuditLogModel(
            id=f"audit_{uuid4().hex}",
            action=action,
            actor=actor,
            branch_name=branch_name,
            decision=decision.value,
            reasons=list(reasons),
            parameters=parameters or {},
            override_by=override.approved_by if override is not None else None,
            override_reason=override.reason if override is not None else None,
            override_ticket_id=override.ticket_id if override is not None else None,
        )
        self._session.add(model)
        self._session.flush()
        return _audit_record(model)

    def list_records(self) -> tuple[AuditLogRecord, ...]:
        records = self._session.scalars(
            select(CitadelAuditLogModel).order_by(CitadelAuditLogModel.created_at)
        ).all()
        return tuple(_audit_record(record) for record in records)


def _audit_record(model: CitadelAuditLogModel) -> AuditLogRecord:
    return AuditLogRecord(
        id=model.id,
        action=model.action,
        actor=model.actor,
        branch_name=model.branch_name,
        decision=AuditDecision(model.decision),
        reasons=tuple(model.reasons),
        parameters=dict(model.parameters),
        override_by=model.override_by,
        override_reason=model.override_reason,
        override_ticket_id=model.override_ticket_id,
        created_at=model.created_at,
    )
