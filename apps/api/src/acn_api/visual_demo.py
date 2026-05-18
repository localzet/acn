import json
import time
from collections.abc import Iterator

from fastapi import APIRouter
from pydantic import BaseModel, ConfigDict
from starlette.responses import StreamingResponse

from acn.experiments.visual_demo import VisualDemoSnapshot, visual_demo_session

router = APIRouter(prefix="/api/v1/demo", tags=["visual-demo"])


class StartDemoRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    auto_mode: bool = True


class AutoModeRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    enabled: bool


class DecisionRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    decision_id: str


class PredictRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    image_data_url: str


@router.get("/state")
async def demo_state() -> VisualDemoSnapshot:
    return visual_demo_session.snapshot()


@router.post("/start")
async def start_demo(payload: StartDemoRequest) -> VisualDemoSnapshot:
    return visual_demo_session.start(auto_mode=payload.auto_mode)


@router.post("/pause")
async def pause_demo() -> VisualDemoSnapshot:
    return visual_demo_session.pause()


@router.post("/resume")
async def resume_demo() -> VisualDemoSnapshot:
    return visual_demo_session.resume()


@router.post("/rollback")
async def rollback_demo() -> VisualDemoSnapshot:
    return visual_demo_session.rollback()


@router.post("/auto-mode")
async def set_auto_mode(payload: AutoModeRequest) -> VisualDemoSnapshot:
    return visual_demo_session.set_auto_mode(payload.enabled)


@router.post("/approve")
async def approve_decision(payload: DecisionRequest) -> VisualDemoSnapshot:
    return visual_demo_session.approve(payload.decision_id)


@router.post("/reject")
async def reject_decision(payload: DecisionRequest) -> VisualDemoSnapshot:
    return visual_demo_session.reject(payload.decision_id)


@router.post("/predict")
async def predict(payload: PredictRequest) -> dict[str, str | float]:
    return visual_demo_session.predict_data_url(payload.image_data_url)


@router.get("/events")
async def demo_events() -> StreamingResponse:
    def events() -> Iterator[str]:
        for _ in range(3600):
            snapshot = visual_demo_session.snapshot()
            yield f"data: {json.dumps({'type': 'snapshot', 'payload': snapshot})}\n\n"
            time.sleep(0.75)

    return StreamingResponse(events(), media_type="text/event-stream")
