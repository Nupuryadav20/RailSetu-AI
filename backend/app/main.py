
from fastapi import FastAPI
import time
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi import Request
from fastapi.responses import JSONResponse
from backend.app.api.dashboard import router as dashboard_router
from backend.app.core.config import settings
from backend.app.core.logging_config import setup_logging
logger = setup_logging()
logger.info("RailSetu AI backend starting...")
app = FastAPI(
    title=settings.APP_NAME,
    description=(
        "AI-Powered Railway Predictive Maintenance "
        "and Automatic Block Planning System"
    ),
    version=settings.APP_VERSION,
    debug=settings.DEBUG,
)
origins = [o.strip() for o in settings.CORS_ORIGINS.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins if origins else ["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(dashboard_router)
@app.middleware("http")
async def log_requests(request: Request, call_next):
    start_time = time.perf_counter()

    try:
        response = await call_next(request)

        duration = round(
            (time.perf_counter() - start_time) * 1000, 2
        )

        logger.info(
            "%s %s | Status: %s | Time: %s ms",
            request.method,
            request.url.path,
            response.status_code,
            duration,
        )

        return response

    except Exception:
        logger.exception(
            "Unhandled error: %s %s",
            request.method,
            request.url.path,
        )
        raise


@app.exception_handler(Exception)
async def global_exception_handler(
    request: Request,
    exc: Exception,
):
    logger.exception(
        "Application error at %s",
        request.url.path,
    )

    return JSONResponse(
        status_code=500,
        content={
            "detail": "Internal server error. Please try again later."
        },
    )

logger.info("Dashboard API routes registered successfully.")
@app.get("/")
def home():
    return {
        "project": settings.APP_NAME,
        "version": settings.APP_VERSION,
        "environment": settings.APP_ENV,
        "status": "Running",
        "message": "Railway Maintenance and Block Planning System",
    }


@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "service": settings.APP_NAME,
    }
