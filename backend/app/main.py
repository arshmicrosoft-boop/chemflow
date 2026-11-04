import os
import secrets
import traceback

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from .routes import router
from .database import init_db

PASSWORD_REQUIRED = "Password required"

app = FastAPI(title="ChemFlow API", description="Full-stack scientific research platform backend")

# Added before CORSMiddleware so it runs inside it: an unhandled error would
# otherwise leave without CORS headers and reach the browser as "Failed to fetch".
@app.middleware("http")
async def report_errors(request: Request, call_next):
    try:
        return await call_next(request)
    except Exception as e:
        traceback.print_exc()
        return JSONResponse(status_code=500, content={"detail": str(e)})

# A hosted app is public, so APP_PASSWORD keeps strangers from running research
# or using up the free model allowance. Left unset, as locally, nothing is asked.
@app.middleware("http")
async def require_password(request: Request, call_next):
    password = os.environ.get("APP_PASSWORD", "")
    given = request.headers.get("X-App-Password", "")
    if password and request.url.path != "/health" and request.method != "OPTIONS" and \
            not secrets.compare_digest(given.encode(), password.encode()):
        return JSONResponse(status_code=401, content={"detail": PASSWORD_REQUIRED})
    return await call_next(request)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[o.strip() for o in os.environ.get("ALLOWED_ORIGINS", "*").split(",") if o.strip()],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

init_db()
app.include_router(router)
