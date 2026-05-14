from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from app.api.chat import router as chat_router
from app.api.ingest import router as ingest_router
from app.api.shrimp_prices import router as shrimp_router
from app.core.config import settings
from app.debug.tracer import configure_logging
from app.dependencies import _init_redis, _close_redis

configure_logging()

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage app lifecycle: startup and shutdown."""
    # Startup
    try:
        await _init_redis()
    except Exception as e:
        import logging
        log = logging.getLogger(__name__)
        log.error(f"Failed to initialize Redis: {e}")
        raise
    
    yield
    
    # Shutdown
    await _close_redis()

app = FastAPI(
    title=settings.PROJECT_NAME,
    openapi_url=f"{settings.API_V1_STR}/openapi.json",
    lifespan=lifespan
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Routes
app.include_router(chat_router, prefix=settings.API_V1_STR)
app.include_router(ingest_router, prefix=settings.API_V1_STR)
app.include_router(shrimp_router, prefix=settings.API_V1_STR)

# Debug Dashboard Routes
@app.get("/debug-dashboard")
async def get_dashboard():
    from fastapi.responses import HTMLResponse
    import os
    file_path = "app/debug/dashboard.html"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return "Dashboard file not found."

@app.get("/3d-flow")
async def get_3d_flow_dashboard():
    from fastapi.responses import HTMLResponse
    import os
    file_path = "app/debug/3d_flow.html"
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return HTMLResponse(content=f.read())
    return "3D Flow Dashboard file not found."

@app.get("/api/v1/debug/requests")
async def get_debug_logs():
    from app.debug.request_tracker import tracker as debug_tracker
    return debug_tracker.get_logs()

@app.delete("/api/v1/debug/requests")
async def clear_debug_logs():
    from app.debug.request_tracker import tracker as debug_tracker
    debug_tracker.clear_logs()
    return {"message": "Logs cleared"}

@app.get("/")
async def root():
    return {"message": "Multi-Agent Chatbot API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
