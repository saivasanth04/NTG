import secrets
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from .config import environment
from .engine import RoutingService
from .types import ChatRequest, RoutingError


def create_app(service=None, token=None):
    token = token or environment().get("NTG_GATEWAY_KEY", "")
    if len(token) < 32 or token.startswith("replace-with-"):
        raise ValueError("Set NTG_GATEWAY_KEY to a random secret of at least 32 characters")

    @asynccontextmanager
    async def lifespan(app):
        app.state.router = service or await RoutingService.from_env()
        yield

    app = FastAPI(title="NTG routing foundation", lifespan=lifespan,
                  docs_url=None, redoc_url=None, openapi_url=None)
    if service:
        app.state.router = service

    async def authenticate(request: Request):
        auth = request.headers.get("authorization", "")
        if not secrets.compare_digest(auth.encode(), f"Bearer {token}".encode()):
            raise RoutingError("unauthorized", 401)

    @app.exception_handler(RoutingError)
    async def routing_error(request, exc):
        return JSONResponse(exc.public(), status_code=exc.status)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request, exc):
        # FastAPI's default validation errors echo input, including sensitive messages.
        return JSONResponse({"error": {"code": "invalid_request", "message": "Request validation failed"}}, status_code=400)

    @app.exception_handler(Exception)
    async def internal_error(request, exc):
        return JSONResponse({"error": {"code": "internal_error", "message": "Internal routing error"}}, status_code=500)

    @app.middleware("http")
    async def body_limit(request, call_next):
        # Bound even chunked request bodies; no prompt data in rejection responses.
        size = 0
        chunks = []
        async for chunk in request.stream():
            size += len(chunk)
            if size > 1_048_576:
                return JSONResponse({"error": {"code": "request_too_large"}}, status_code=413)
            chunks.append(chunk)
        request._body = b"".join(chunks)
        return await call_next(request)

    @app.get("/healthz")
    async def health():
        return {"status": "running"}

    @app.get("/v1/models", dependencies=[Depends(authenticate)])
    async def models(request: Request):
        return {"object": "list", "data": [{"id": "auto", "object": "model", "owned_by": "ntg"}] + [
            {"id": d.id, "object": "model", "owned_by": d.account.provider}
            for d in request.app.state.router.deployments]}

    @app.post("/v1/chat/completions", dependencies=[Depends(authenticate)])
    async def chat(body: ChatRequest, request: Request):
        return await request.app.state.router.chat(body)

    @app.get("/status", dependencies=[Depends(authenticate)])
    async def status(request: Request):
        return request.app.state.router.status()

    return app
