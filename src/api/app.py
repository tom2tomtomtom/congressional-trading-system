"""
Main FastAPI application for the Congressional Trading Intelligence API.

This is the entry point for the REST API that provides:
- Congressional member data
- Stock trade disclosures
- AI-powered analysis (conviction scores, timing analysis)
- Story generation for journalists
- API key authentication with tiered rate limiting
"""

import time
from contextlib import asynccontextmanager
from datetime import datetime
from typing import Callable

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse

from .config import get_settings, API_DESCRIPTION, API_TAGS_METADATA
from .database import init_db, close_db
from .routers import members_router, trades_router, analysis_router, auth_router
from .schemas import HealthCheck, APIInfo, ErrorResponse


settings = get_settings()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager for startup/shutdown events."""
    # Startup
    print(f"🚀 Starting {settings.app_name} v{settings.app_version}")
    init_db()
    yield
    # Shutdown
    print("👋 Shutting down API...")
    close_db()


# Create FastAPI application
app = FastAPI(
    title=settings.app_name,
    description=API_DESCRIPTION,
    version=settings.app_version,
    openapi_tags=API_TAGS_METADATA,
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    lifespan=lifespan,
)


# CORS Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=settings.cors_allow_credentials,
    allow_methods=settings.cors_allow_methods,
    allow_headers=settings.cors_allow_headers,
)


# Request timing middleware
@app.middleware("http")
async def add_process_time_header(request: Request, call_next: Callable) -> Response:
    """Add X-Process-Time header to all responses."""
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(round(process_time * 1000, 2))
    return response


# Exception handlers
@app.exception_handler(404)
async def not_found_handler(request: Request, exc) -> JSONResponse:
    """Handle 404 errors."""
    return JSONResponse(
        status_code=404,
        content={"error": "not_found", "message": "The requested resource was not found"},
    )


@app.exception_handler(500)
async def internal_error_handler(request: Request, exc) -> JSONResponse:
    """Handle 500 errors."""
    return JSONResponse(
        status_code=500,
        content={"error": "internal_error", "message": "An internal server error occurred"},
    )


# Include routers
app.include_router(members_router, prefix="/api")
app.include_router(trades_router, prefix="/api")
app.include_router(analysis_router, prefix="/api")
app.include_router(auth_router, prefix="/api")


# Root endpoints
@app.get(
    "/",
    response_model=APIInfo,
    tags=["health"],
    summary="API Information",
    description="Get basic information about the API.",
)
async def root() -> APIInfo:
    """Return API information."""
    return APIInfo(
        name=settings.app_name,
        version=settings.app_version,
        description="REST API for analyzing congressional trading patterns",
        documentation_url="/docs",
        endpoints={
            "members": "/api/members",
            "trades": "/api/trades",
            "analysis": "/api/analysis",
            "auth": "/api/auth",
            "health": "/health",
            "docs": "/docs",
            "redoc": "/redoc",
        },
    )


@app.get(
    "/health",
    response_model=HealthCheck,
    tags=["health"],
    summary="Health Check",
    description="Check if the API is healthy and running.",
)
async def health_check() -> HealthCheck:
    """Return health status."""
    return HealthCheck(
        status="healthy",
        service="congressional-trading-api",
        version=settings.app_version,
        timestamp=datetime.utcnow(),
    )


@app.get(
    "/api",
    response_model=APIInfo,
    tags=["health"],
    summary="API Root",
    description="Get information about available API endpoints.",
)
async def api_root() -> APIInfo:
    """Return API endpoint information."""
    return APIInfo(
        name=settings.app_name,
        version=settings.app_version,
        description="REST API for analyzing congressional trading patterns",
        documentation_url="/docs",
        endpoints={
            "members": "/api/members - List and search congressional members",
            "members/{id}": "/api/members/{id} - Get member details",
            "members/{id}/trades": "/api/members/{id}/trades - Get member trades",
            "members/{id}/score": "/api/members/{id}/score - Get member swamp score",
            "trades": "/api/trades - List and search trades",
            "trades/{id}": "/api/trades/{id} - Get trade details",
            "trades/{id}/conviction": "/api/trades/{id}/conviction - Get conviction analysis",
            "trades/{id}/timing": "/api/trades/{id}/timing - Get timing analysis",
            "analysis/leaderboard": "/api/analysis/leaderboard - Get rankings",
            "analysis/stats": "/api/analysis/stats - Get aggregate statistics",
            "analysis/stories/generate": "/api/analysis/stories/generate - Generate story",
            "auth/keys": "/api/auth/keys - Manage API keys",
            "auth/usage": "/api/auth/usage - Get usage statistics",
        },
    )


def custom_openapi():
    """Generate custom OpenAPI schema."""
    if app.openapi_schema:
        return app.openapi_schema

    openapi_schema = get_openapi(
        title=settings.app_name,
        version=settings.app_version,
        description=API_DESCRIPTION,
        routes=app.routes,
        tags=API_TAGS_METADATA,
    )

    # Add security scheme
    openapi_schema["components"]["securitySchemes"] = {
        "ApiKeyAuth": {
            "type": "apiKey",
            "in": "header",
            "name": "X-API-Key",
            "description": "API key for authentication. Get one at /api/auth/keys",
        }
    }

    # Add example responses
    openapi_schema["info"]["x-logo"] = {
        "url": "https://example.com/logo.png",
        "altText": "Congressional Trading API",
    }

    app.openapi_schema = openapi_schema
    return app.openapi_schema


app.openapi = custom_openapi


def main():
    """Run the API server."""
    import uvicorn

    uvicorn.run(
        "src.api.app:app",
        host=settings.host,
        port=settings.port,
        reload=settings.debug,
        log_level="info" if not settings.debug else "debug",
    )


if __name__ == "__main__":
    main()
