import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from typing import Any

from fastapi import APIRouter, WebSocket
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

router = APIRouter(prefix="/api/v1", tags=["dashboard"])


class OverrideRequest(BaseModel):
    model_config = ConfigDict(extra="ignore", populate_by_name=True)

    decision_id: str = Field(validation_alias=AliasChoices("decision_id", "decisionId"))
    approved_by: str = Field(validation_alias=AliasChoices("approved_by", "approvedBy"))
    reason: str
    ticket_id: str | None = Field(
        default=None,
        validation_alias=AliasChoices("ticket_id", "ticketId"),
    )


@router.get("/dashboard/snapshot")
async def dashboard_snapshot() -> dict[str, Any]:
    return _empty_snapshot()


@router.get("/dashboard/events")
async def dashboard_events() -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        yield f"data: {json.dumps({'type': 'snapshot', 'payload': _empty_snapshot()})}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


@router.websocket("/dashboard/ws")
async def dashboard_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "snapshot", "payload": _empty_snapshot()})
    await websocket.close()


@router.post("/overrides")
async def submit_override(payload: OverrideRequest) -> dict[str, Any]:
    return {
        "status": "accepted",
        "decisionId": payload.decision_id,
        "approvedBy": payload.approved_by,
        "ticketId": payload.ticket_id,
    }


def _empty_snapshot() -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    return {
        "commitGraph": {"nodes": [], "edges": []},
        "branchGraph": {"nodes": [], "edges": []},
        "metricsTimeline": [],
        "experiments": [],
        "controllerDecisions": [],
        "rollbackHistory": [],
        "liveLogs": [
            {
                "id": "api_bootstrap",
                "level": "info",
                "source": "api",
                "message": "Dashboard snapshot endpoint is available.",
                "createdAt": now,
            }
        ],
    }
