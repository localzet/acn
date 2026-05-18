from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from acn.config.logging import configure_logging
from acn.config.settings import Settings, get_settings
from acn_api.dashboard import router as dashboard_router
from acn_api.visual_demo import router as visual_demo_router


def create_app(settings: Settings | None = None) -> FastAPI:
    resolved_settings = settings or get_settings()
    configure_logging(resolved_settings)
    app = FastAPI(
        title="Adaptive Core Network API",
        version="0.1.0",
        docs_url="/docs",
        redoc_url="/redoc",
    )
    app.state.settings = resolved_settings
    app.add_middleware(
        CORSMiddleware,
        allow_origin_regex=resolved_settings.cors_allow_origin_regex,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    app.include_router(dashboard_router)
    app.include_router(visual_demo_router)

    @app.get("/health", tags=["system"])
    async def health() -> dict[str, str]:
        return {"status": "ok", "environment": resolved_settings.env}

    return app


app = create_app()
