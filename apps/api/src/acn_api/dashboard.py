import asyncio
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime
from itertools import pairwise
from typing import Any

from fastapi import APIRouter, Request, WebSocket
from pydantic import AliasChoices, BaseModel, ConfigDict, Field
from starlette.responses import StreamingResponse

from acn.experiments.visual_demo import VisualDemoSnapshot, visual_demo_session

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
async def dashboard_snapshot(request: Request) -> dict[str, Any]:
    return _load_snapshot(request)


@router.get("/dashboard/events")
async def dashboard_events(request: Request) -> StreamingResponse:
    async def events() -> AsyncIterator[str]:
        for _ in range(3600):
            event = {"type": "snapshot", "payload": _load_snapshot(request)}
            yield f"data: {json.dumps(event)}\n\n"
            await asyncio.sleep(0.75)

    return StreamingResponse(events(), media_type="text/event-stream")


@router.websocket("/dashboard/ws")
async def dashboard_websocket(websocket: WebSocket) -> None:
    await websocket.accept()
    await websocket.send_json({"type": "snapshot", "payload": _load_snapshot(websocket)})
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


def _load_snapshot(request: Request | WebSocket) -> dict[str, Any]:
    settings = getattr(request.app.state, "settings", None)
    telemetry_path = getattr(settings, "dashboard_telemetry_path", None)
    if telemetry_path is None or not telemetry_path.exists():
        return _visual_demo_dashboard_snapshot(visual_demo_session.snapshot())

    with telemetry_path.open(encoding="utf-8") as stream:
        payload = json.load(stream)
    if not isinstance(payload, dict):
        return _visual_demo_dashboard_snapshot(visual_demo_session.snapshot())
    return payload


def _visual_demo_dashboard_snapshot(snapshot: VisualDemoSnapshot) -> dict[str, Any]:
    now = datetime.now(UTC).isoformat()
    checkpoints = snapshot["checkpoints"]
    decisions = snapshot["decisions"]
    events = snapshot["events"]
    current_commit_id = snapshot["activeCheckpointId"]
    best_checkpoint = max(checkpoints, key=lambda item: item["accuracy"], default=None)
    checkpoint_edges = [
        {"parentId": previous["id"], "childId": current["id"]}
        for previous, current in pairwise(checkpoints)
    ]
    rollback_decisions = [decision for decision in decisions if decision["action"] == "rollback"]
    return {
        "commitGraph": {
            "nodes": [
                {
                    "id": checkpoint["id"],
                    "branchId": "visual-demo",
                    "checkpointId": checkpoint["id"],
                    "message": (
                        "stable checkpoint" if checkpoint["stable"] else "candidate checkpoint"
                    ),
                    "createdAt": checkpoint["createdAt"],
                    "metrics": {
                        "validation_loss": checkpoint["validationLoss"],
                        "accuracy": checkpoint["accuracy"],
                        "stable": checkpoint["stable"],
                    },
                }
                for checkpoint in checkpoints
            ],
            "edges": checkpoint_edges,
        },
        "branchGraph": {
            "nodes": [
                {
                    "id": "visual-demo",
                    "name": snapshot["currentBranch"],
                    "headCommitId": current_commit_id,
                    "baseCommitId": checkpoints[0]["id"] if checkpoints else None,
                    "status": snapshot["status"],
                }
            ],
            "edges": [],
        },
        "metricsTimeline": [
            {
                "timestamp": metric["timestamp"],
                "stageId": metric["stage"],
                "trainLoss": metric["trainLoss"],
                "validationLoss": metric["validationLoss"],
                "trainAccuracy": None,
                "validationAccuracy": metric["accuracy"],
                "forgettingScore": max(0.0, 1.0 - metric["accuracy"]),
                "oldClassRetention": metric["accuracy"],
                "newClassAdaptation": metric["accuracy"],
            }
            for metric in snapshot["metrics"]
        ],
        "experiments": [
            {
                "id": "exp_visual_demo",
                "name": "Visual adaptive classifier demo",
                "status": snapshot["status"],
                "branchName": snapshot["currentBranch"],
                "currentStageId": snapshot["stage"],
                "currentCommitId": current_commit_id,
                "bestCommitId": best_checkpoint["id"] if best_checkpoint else None,
                "updatedAt": now,
            }
        ],
        "controllerDecisions": [
            {
                "id": decision["id"],
                "action": decision["action"],
                "confidence": 0.92 if decision["status"] != "denied" else 0.5,
                "branchName": snapshot["currentBranch"],
                "commitId": current_commit_id,
                "reasons": [decision["reason"]],
                "createdAt": decision["createdAt"],
                "status": decision["status"],
            }
            for decision in decisions
        ],
        "rollbackHistory": [
            {
                "id": decision["id"],
                "branchName": snapshot["currentBranch"],
                "fromCommitId": current_commit_id or "unknown",
                "toCommitId": snapshot["activeCheckpointId"] or "unknown",
                "actor": "controller" if snapshot["autoMode"] else "operator",
                "createdAt": decision["createdAt"],
                "reason": decision["reason"],
            }
            for decision in rollback_decisions
            if decision["status"] in {"approved", "executed"}
        ],
        "liveLogs": [
            {
                "id": event["id"],
                "level": event["level"],
                "source": "visual-demo",
                "message": event["message"],
                "createdAt": event["createdAt"],
            }
            for event in reversed(events[-120:])
        ],
    }
